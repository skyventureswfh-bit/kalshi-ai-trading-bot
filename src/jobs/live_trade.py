"""
Runtime-aware live-trade decision loop for short-dated markets.

This module adds a small W5 foundation inside the existing trading cycle:
1. Scout ranked live-trade events.
2. Run focus-aware specialist analysis on a short list.
3. Synthesize one trade intent.
4. Execute only if the existing live-trade guardrails allow it.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

import httpx
from json_repair import repair_json

from src.clients.kalshi_client import KalshiClient
from src.clients.model_router import ModelRouter
from src.config.settings import settings
from src.data.live_trade_research import LiveTradeResearchService
from src.jobs.execute import execute_position
from src.strategies.quick_flip_scalping import (
    QuickFlipConfig,
    QuickFlipOpportunity,
    QuickFlipScalpingStrategy,
    manage_live_quick_flip_positions,
)
from src.utils.database import (
    DatabaseManager,
    LiveTradeDecision,
    LiveTradeRuntimeState,
    Market,
    Position,
)
from src.utils.kalshi_normalization import (
    get_balance_dollars,
    get_market_tick_size,
    get_portfolio_value_dollars,
)
from src.utils.logging_setup import get_trading_logger
from src.utils.source_health import record_research_payload_snapshots


SCOUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "selected_events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event_ticker": {"type": "string"},
                    "priority": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["event_ticker", "priority", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "selected_events"],
    "additionalProperties": False,
}

SPECIALIST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "action": {"type": "string", "enum": ["TRADE", "WATCH", "SKIP"]},
        "market_ticker": {"type": "string"},
        "side": {"type": "string", "enum": ["YES", "NO"]},
        "fair_yes_probability": {"type": "number"},
        "confidence": {"type": "number"},
        "edge_pct": {"type": "number"},
        "position_size_pct": {"type": "number"},
        "hold_minutes": {"type": "integer"},
        "limit_price": {"type": "number"},
        "execution_style": {
            "type": "string",
            "enum": ["QUICK_FLIP", "LIVE_TRADE", "NONE"],
        },
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": [
        "summary",
        "action",
        "market_ticker",
        "side",
        "fair_yes_probability",
        "confidence",
        "edge_pct",
        "position_size_pct",
        "hold_minutes",
        "limit_price",
        "execution_style",
        "risk_flags",
        "reasoning",
    ],
    "additionalProperties": False,
}

FINAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "action": {"type": "string", "enum": ["BUY", "SKIP"]},
        "event_ticker": {"type": "string"},
        "market_ticker": {"type": "string"},
        "side": {"type": "string", "enum": ["YES", "NO"]},
        "fair_yes_probability": {"type": "number"},
        "confidence": {"type": "number"},
        "edge_pct": {"type": "number"},
        "position_size_pct": {"type": "number"},
        "hold_minutes": {"type": "integer"},
        "limit_price": {"type": "number"},
        "execution_style": {
            "type": "string",
            "enum": ["QUICK_FLIP", "LIVE_TRADE", "NONE"],
        },
        "reasoning": {"type": "string"},
    },
    "required": [
        "summary",
        "action",
        "event_ticker",
        "market_ticker",
        "side",
        "fair_yes_probability",
        "confidence",
        "edge_pct",
        "position_size_pct",
        "hold_minutes",
        "limit_price",
        "execution_style",
        "reasoning",
    ],
    "additionalProperties": False,
}


def _response_format(name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, *, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _parse_json_payload(raw_text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    try:
        repaired = repair_json(str(raw_text))
        parsed = json.loads(repaired)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _extract_router_metadata(model_router: ModelRouter) -> Dict[str, Optional[str]]:
    provider = getattr(model_router, "default_provider", None)
    if provider == "codex" and getattr(model_router, "codex_client", None) is not None:
        metadata = model_router.codex_client.last_request_metadata
        return {
            "provider": "codex",
            "model": metadata.actual_model or metadata.requested_model,
        }
    if provider == "openai" and getattr(model_router, "openai_client", None) is not None:
        metadata = model_router.openai_client.last_request_metadata
        return {
            "provider": "openai",
            "model": metadata.actual_model or metadata.requested_model,
        }
    if getattr(model_router, "openrouter_client", None) is not None:
        metadata = model_router.openrouter_client.last_request_metadata
        return {
            "provider": "openrouter",
            "model": metadata.actual_model or metadata.requested_model,
        }
    return {"provider": provider, "model": None}


def _event_rank(event: Dict[str, Any]) -> tuple[float, float, float, float]:
    live_bonus = 1.0 if event.get("is_live_candidate") else 0.0
    live_score = _safe_float(event.get("live_score"), 0.0)
    volume = _safe_float(event.get("volume_24h"), 0.0)
    spread_penalty = -_safe_float(event.get("avg_yes_spread"), 1.0)
    return (live_bonus, live_score, volume, spread_penalty)


def _heuristic_shortlist(
    events: Sequence[Dict[str, Any]],
    *,
    shortlist_size: int,
) -> List[Dict[str, Any]]:
    ranked = sorted(events, key=_event_rank, reverse=True)
    return list(ranked[:shortlist_size])


def _market_side_entry_price(market: Dict[str, Any], side: str) -> float:
    normalized_side = str(side or "YES").upper()
    if normalized_side == "NO":
        no_bid = _safe_float(market.get("no_bid"), 0.0)
        no_ask = _safe_float(market.get("no_ask"), 0.0)
        if no_bid > 0 and no_ask > 0:
            return round((no_bid + no_ask) / 2.0, 4)
        return round(_clamp(1.0 - _safe_float(market.get("yes_midpoint"), 0.5), lo=0.01, hi=0.99), 4)
    return round(_clamp(_safe_float(market.get("yes_midpoint"), 0.5), lo=0.01, hi=0.99), 4)


def _market_title(market: Dict[str, Any]) -> str:
    return str(market.get("title") or market.get("ticker") or "").strip()


def _build_quick_flip_config(*, hold_minutes: int) -> QuickFlipConfig:
    max_hold_minutes = max(
        1,
        min(
            hold_minutes or int(getattr(settings.trading, "quick_flip_max_hold_minutes", 30) or 30),
            int(getattr(settings.trading, "quick_flip_max_hold_minutes", 30) or 30),
        ),
    )
    return QuickFlipConfig(
        min_entry_price=settings.trading.quick_flip_min_entry_price,
        max_entry_price=settings.trading.quick_flip_max_entry_price,
        min_profit_margin=settings.trading.quick_flip_min_profit_margin,
        max_position_size=settings.trading.quick_flip_max_position_size,
        max_concurrent_positions=settings.trading.quick_flip_max_concurrent_positions,
        capital_per_trade=settings.trading.quick_flip_capital_per_trade,
        confidence_threshold=settings.trading.quick_flip_confidence_threshold,
        max_hold_minutes=max_hold_minutes,
        max_market_checks=settings.trading.quick_flip_max_market_checks,
        target_opportunity_buffer=settings.trading.quick_flip_target_opportunity_buffer,
        min_market_volume=settings.trading.quick_flip_min_market_volume,
        max_hours_to_expiry=settings.trading.quick_flip_max_hours_to_expiry,
        max_bid_ask_spread=settings.trading.quick_flip_max_bid_ask_spread,
        min_orderbook_depth_contracts=settings.trading.quick_flip_min_top_of_book_size,
        min_net_profit_per_trade=settings.trading.quick_flip_min_net_profit,
        min_net_roi=settings.trading.quick_flip_min_net_roi,
        recent_trade_window_seconds=settings.trading.quick_flip_recent_trade_window_seconds,
        min_recent_trade_count=settings.trading.quick_flip_min_recent_trade_count,
        max_target_vs_recent_trade_gap=settings.trading.quick_flip_max_target_vs_recent_trade_gap,
        min_recent_range_ticks=settings.trading.quick_flip_min_recent_range_ticks,
        min_recent_price_position=settings.trading.quick_flip_min_recent_price_position,
        max_entry_vs_recent_last_gap=settings.trading.quick_flip_max_entry_vs_recent_last_gap,
        maker_entry_timeout_seconds=settings.trading.quick_flip_maker_entry_timeout_seconds,
        maker_entry_poll_seconds=settings.trading.quick_flip_maker_entry_poll_seconds,
        maker_entry_reprice_seconds=settings.trading.quick_flip_maker_entry_reprice_seconds,
        dynamic_exit_reprice_seconds=settings.trading.quick_flip_dynamic_exit_reprice_seconds,
        stop_loss_pct=settings.trading.quick_flip_stop_loss_pct,
        ev_gate_enabled=settings.trading.quick_flip_ev_gate_enabled,
        ev_confidence_margin=settings.trading.quick_flip_ev_confidence_margin,
        max_last_trade_age_seconds=settings.trading.quick_flip_max_last_trade_age_seconds,
        min_bid_ask_size_ratio=settings.trading.quick_flip_min_bid_ask_size_ratio,
    )


def _resolve_quick_flip_runtime_label() -> str:
    if bool(getattr(settings.trading, "live_trading_enabled", False)):
        return "live"
    if bool(getattr(settings.trading, "shadow_mode_enabled", False)):
        return "shadow"
    return "paper"


async def _execute_quick_flip_paper_intent(
    *,
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    selected_event: Dict[str, Any],
    selected_market: Dict[str, Any],
    final_intent: Dict[str, Any],
    quantity: int,
) -> Dict[str, Any]:
    runtime_label = _resolve_quick_flip_runtime_label()
    hold_minutes = max(_safe_int(final_intent.get("hold_minutes"), 0), 1)
    entry_price = _clamp(_safe_float(final_intent.get("limit_price"), 0.5), lo=0.01, hi=0.99)
    config = _build_quick_flip_config(hold_minutes=hold_minutes)
    strategy = QuickFlipScalpingStrategy(
        db_manager=db_manager,
        kalshi_client=kalshi_client,
        xai_client=None,
        config=config,
        disable_ai=True,
    )
    runtime_shadow_mode = runtime_label == "shadow"

    market_info: Dict[str, Any] = {}
    try:
        market_response = await kalshi_client.get_market(str(selected_market.get("ticker") or ""))
        market_info = market_response.get("market", {}) if isinstance(market_response, dict) else {}
    except Exception:
        market_info = {}

    tick_size = get_market_tick_size(market_info or {}, entry_price)
    min_exit_price = strategy._minimum_profitable_exit_price(
        entry_price=entry_price,
        quantity=quantity,
        tick_size=tick_size,
        market_info=market_info or None,
    )
    edge_target = entry_price + max(_safe_float(final_intent.get("edge_pct"), 0.0), tick_size)
    exit_price = strategy._round_up_to_valid_tick(
        price=max(min_exit_price, edge_target),
        tick_size=tick_size,
        market_info=market_info or None,
    )
    if exit_price > 0.95:
        return {
            "executed": False,
            "status": "skipped",
            "summary": "Quick-flip target could not clear the profit floor within valid price bands.",
            "error": "quick_flip_exit_unreachable",
            "quantity": quantity,
        }

    profit_estimate = strategy._estimate_trade_profit(
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
    )
    opportunity = QuickFlipOpportunity(
        market_id=str(selected_market.get("ticker") or ""),
        market_title=_market_title(selected_market),
        side=str(final_intent.get("side") or "YES"),
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        expected_profit=max(float(profit_estimate.get("net_profit", 0.0)), 0.0),
        confidence_score=_safe_float(final_intent.get("confidence"), 0.0),
        movement_indicator=str(
            final_intent.get("reasoning")
            or final_intent.get("summary")
            or f"Live-trade synth quick flip for {selected_event.get('event_ticker') or selected_market.get('ticker')}"
        ),
        max_hold_time=config.max_hold_minutes,
        tick_size=tick_size,
    )

    executed = await strategy._execute_single_quick_flip(
        opportunity,
        shadow_mode=runtime_shadow_mode,
    )
    if not executed:
        return {
            "executed": False,
            "status": "error",
            "summary": f"Quick-flip {runtime_label} entry did not fill for the live-trade intent.",
            "error": "quick_flip_entry_failed",
            "quantity": quantity,
            "payload": {
                "entry_price": entry_price,
                "target_exit_price": exit_price,
                "expected_profit": opportunity.expected_profit,
                "execution_mode": runtime_label,
            },
        }

    sell_result = await strategy._place_immediate_sell_order(opportunity)
    payload = {
        "entry_price": entry_price,
        "target_exit_price": exit_price,
        "expected_profit": opportunity.expected_profit,
        "execution_mode": runtime_label,
        "exit_order": sell_result,
    }
    if sell_result.get("success"):
        summary = (
            f"Quick-flip {runtime_label} position opened and closed immediately."
            if sell_result.get("filled")
            else f"Quick-flip {runtime_label} position opened with a resting exit order."
        )
        return {
            "executed": True,
            "status": "executed",
            "summary": summary,
            "quantity": quantity,
            "payload": payload,
        }

    return {
        "executed": True,
        "status": "executed",
        "summary": (
            f"Quick-flip {runtime_label} entry filled, but the exit order did not post cleanly."
        ),
        "error": "quick_flip_exit_order_failed",
        "quantity": quantity,
        "payload": payload,
    }


def _trim_research_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    news = payload.get("news") or {}
    trimmed_news = {
        "article_count": _safe_int(news.get("article_count"), 0),
        "articles": [
            {
                "title": item.get("title"),
                "source": item.get("source"),
                "published": item.get("published"),
                "relevance": item.get("relevance"),
            }
            for item in (news.get("articles") or [])[:3]
        ],
    }
    event = payload.get("event") or {}
    trimmed_event = {
        "event_ticker": event.get("event_ticker"),
        "title": event.get("title"),
        "category": event.get("category"),
        "focus_type": event.get("focus_type"),
        "hours_to_expiry": event.get("hours_to_expiry"),
        "live_score": event.get("live_score"),
        "market_count": event.get("market_count"),
        "markets": [
            {
                "ticker": market.get("ticker"),
                "title": market.get("title"),
                "yes_midpoint": market.get("yes_midpoint"),
                "yes_bid": market.get("yes_bid"),
                "yes_ask": market.get("yes_ask"),
                "no_bid": market.get("no_bid"),
                "no_ask": market.get("no_ask"),
                "yes_spread": market.get("yes_spread"),
                "volume_24h": market.get("volume_24h"),
                "hours_to_expiry": market.get("hours_to_expiry"),
            }
            for market in (event.get("markets") or [])[:5]
        ],
    }
    return {
        "event": trimmed_event,
        "news": trimmed_news,
        "microstructure": payload.get("microstructure") or {},
        "cross_market_context": payload.get("cross_market_context"),
        "sports_context": payload.get("sports_context"),
        "bitcoin_context": payload.get("bitcoin_context"),
        "crypto_context": payload.get("crypto_context"),
        "macro_context": payload.get("macro_context"),
        "weather_context": payload.get("weather_context"),
    }


def _normalize_specialist_payload(
    payload: Optional[Dict[str, Any]],
    *,
    event: Dict[str, Any],
) -> Dict[str, Any]:
    markets = event.get("markets") or []
    default_market = markets[0] if markets else {}
    action = str((payload or {}).get("action", "SKIP")).upper()
    if action not in {"TRADE", "WATCH", "SKIP"}:
        action = "SKIP"
    side = str((payload or {}).get("side", "YES")).upper()
    if side not in {"YES", "NO"}:
        side = "YES"
    market_ticker = str((payload or {}).get("market_ticker") or default_market.get("ticker") or "")
    chosen_market = next(
        (market for market in markets if str(market.get("ticker")) == market_ticker),
        default_market,
    )
    limit_price = _safe_float(
        (payload or {}).get("limit_price"),
        _market_side_entry_price(chosen_market or {}, side),
    )
    execution_style = str((payload or {}).get("execution_style", "NONE")).upper()
    if execution_style not in {"QUICK_FLIP", "LIVE_TRADE", "NONE"}:
        execution_style = "NONE"
    # Fail-closed fair value: when the model omits its probability estimate,
    # fall back to the market midpoint, which yields zero edge at the EV gate.
    market_midpoint = _clamp(
        _safe_float((chosen_market or {}).get("yes_midpoint"), 0.5), lo=0.01, hi=0.99
    )
    fair_yes_probability = _clamp(
        _safe_float((payload or {}).get("fair_yes_probability"), market_midpoint),
        lo=0.01,
        hi=0.99,
    )
    normalized = {
        "summary": str((payload or {}).get("summary", "") or ""),
        "action": action,
        "market_ticker": market_ticker,
        "side": side,
        "fair_yes_probability": fair_yes_probability,
        "confidence": _clamp(_safe_float((payload or {}).get("confidence"), 0.0), lo=0.0, hi=1.0),
        "edge_pct": _safe_float((payload or {}).get("edge_pct"), 0.0),
        "position_size_pct": _clamp(_safe_float((payload or {}).get("position_size_pct"), 1.0), lo=0.0, hi=100.0),
        "hold_minutes": max(_safe_int((payload or {}).get("hold_minutes"), 0), 0),
        "limit_price": _clamp(limit_price, lo=0.01, hi=0.99),
        "execution_style": execution_style,
        "risk_flags": list((payload or {}).get("risk_flags") or []),
        "reasoning": str((payload or {}).get("reasoning", "") or ""),
    }
    if action != "TRADE":
        normalized["execution_style"] = "NONE"
    return normalized


def _normalize_final_payload(
    payload: Optional[Dict[str, Any]],
    *,
    candidates: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if not payload:
        best = max(
            candidates,
            key=lambda item: (
                _safe_float(item.get("confidence"), 0.0),
                _safe_float(item.get("edge_pct"), 0.0),
            ),
            default=None,
        )
        if best is None:
            return {
                "summary": "No live-trade candidate cleared the specialist bar.",
                "action": "SKIP",
                "event_ticker": "",
                "market_ticker": "",
                "side": "YES",
                "fair_yes_probability": 0.5,
                "confidence": 0.0,
                "edge_pct": 0.0,
                "position_size_pct": 0.0,
                "hold_minutes": 0,
                "limit_price": 0.0,
                "execution_style": "NONE",
                "reasoning": "No specialist candidate was strong enough to trade.",
            }
        return {
            "summary": best.get("summary") or "Best available specialist candidate selected heuristically.",
            "action": "BUY",
            "event_ticker": best.get("event_ticker", ""),
            "market_ticker": best.get("market_ticker", ""),
            "side": best.get("side", "YES"),
            "fair_yes_probability": _clamp(
                _safe_float(best.get("fair_yes_probability"), 0.5), lo=0.01, hi=0.99
            ),
            "confidence": _clamp(_safe_float(best.get("confidence"), 0.0), lo=0.0, hi=1.0),
            "edge_pct": _safe_float(best.get("edge_pct"), 0.0),
            "position_size_pct": _clamp(_safe_float(best.get("position_size_pct"), 1.0), lo=0.0, hi=100.0),
            "hold_minutes": max(_safe_int(best.get("hold_minutes"), 0), 0),
            "limit_price": _clamp(_safe_float(best.get("limit_price"), 0.5), lo=0.01, hi=0.99),
            "execution_style": best.get("execution_style", "NONE"),
            "reasoning": best.get("reasoning", ""),
        }

    action = str(payload.get("action", "SKIP")).upper()
    if action not in {"BUY", "SKIP"}:
        action = "SKIP"
    side = str(payload.get("side", "YES")).upper()
    if side not in {"YES", "NO"}:
        side = "YES"
    execution_style = str(payload.get("execution_style", "NONE")).upper()
    if execution_style not in {"QUICK_FLIP", "LIVE_TRADE", "NONE"}:
        execution_style = "NONE"
    normalized = {
        "summary": str(payload.get("summary", "") or ""),
        "action": action,
        "event_ticker": str(payload.get("event_ticker", "") or ""),
        "market_ticker": str(payload.get("market_ticker", "") or ""),
        "side": side,
        "fair_yes_probability": _clamp(
            _safe_float(payload.get("fair_yes_probability"), 0.5), lo=0.01, hi=0.99
        ),
        "confidence": _clamp(_safe_float(payload.get("confidence"), 0.0), lo=0.0, hi=1.0),
        "edge_pct": _safe_float(payload.get("edge_pct"), 0.0),
        "position_size_pct": _clamp(_safe_float(payload.get("position_size_pct"), 0.0), lo=0.0, hi=100.0),
        "hold_minutes": max(_safe_int(payload.get("hold_minutes"), 0), 0),
        "limit_price": _clamp(_safe_float(payload.get("limit_price"), 0.5), lo=0.01, hi=0.99),
        "execution_style": execution_style,
        "reasoning": str(payload.get("reasoning", "") or ""),
    }
    if action != "BUY":
        normalized["execution_style"] = "NONE"
    # fair_yes_disagreement and member_probabilities must survive
    # normalization: the EV gate demands extra edge on contested calls, and
    # settlement scoring needs each member's probability claim.
    for extra_key in (
        "debate_transcript",
        "step_results",
        "elapsed_seconds",
        "selected_candidate",
        "fair_yes_disagreement",
        "member_probabilities",
    ):
        if extra_key in payload:
            normalized[extra_key] = payload[extra_key]
    return normalized


def _candidate_priority(item: Dict[str, Any]) -> tuple[float, float, float]:
    return (
        _safe_float(item.get("confidence"), 0.0),
        _safe_float(item.get("edge_pct"), 0.0),
        -_safe_float(item.get("hold_minutes"), 0.0),
    )


def _selected_candidate_for_debate(candidates: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return max(candidates, key=_candidate_priority, default=None)


def _candidate_price_cents(value: Any, *, default: int) -> int:
    numeric = _safe_float(value, float(default))
    if numeric <= 1.0:
        numeric *= 100.0
    return int(round(_clamp(numeric, lo=1.0, hi=99.0)))


def _candidate_market_data(candidate: Dict[str, Any]) -> Dict[str, Any]:
    side = str(candidate.get("side") or "YES").upper()
    limit_price = _clamp(_safe_float(candidate.get("limit_price"), 0.5), lo=0.01, hi=0.99)
    if side == "NO":
        yes_probability = _clamp(1.0 - limit_price, lo=0.01, hi=0.99)
        no_probability = limit_price
    else:
        yes_probability = limit_price
        no_probability = _clamp(1.0 - limit_price, lo=0.01, hi=0.99)

    risk_flags = [str(item) for item in (candidate.get("risk_flags") or []) if str(item).strip()]
    fair_yes = _safe_float(candidate.get("fair_yes_probability"), 0.0)
    entry_fee_estimate = 0.07 * limit_price * (1.0 - limit_price)
    summary_lines = [
        "Entry-only live-trade loop candidate. Emit BUY with side YES or NO, or SKIP. Do not emit SELL.",
        f"Focus type: {candidate.get('focus_type') or 'general'}",
        f"Execution style: {candidate.get('execution_style') or 'NONE'}",
        f"Target hold minutes: {max(_safe_int(candidate.get('hold_minutes'), 0), 0)}",
        f"Specialist fair YES probability: {fair_yes:.2f}" if fair_yes > 0 else "Specialist fair YES probability: not provided",
        f"Specialist edge estimate: {_safe_float(candidate.get('edge_pct'), 0.0):.4f}",
        f"Specialist confidence: {_safe_float(candidate.get('confidence'), 0.0):.2f}",
        (
            f"Estimated Kalshi taker fee at entry: ~{entry_fee_estimate * 100:.1f}c/contract. "
            "Your edge must clearly exceed fees; if it does not, SKIP."
        ),
    ]
    if risk_flags:
        summary_lines.append(f"Risk flags: {', '.join(risk_flags[:5])}")
    if candidate.get("summary"):
        summary_lines.append(f"Specialist summary: {candidate.get('summary')}")

    return {
        "ticker": candidate.get("market_ticker"),
        "title": candidate.get("market_title") or candidate.get("title") or candidate.get("market_ticker") or "Live-trade candidate",
        "yes_price": _candidate_price_cents(candidate.get("yes_price"), default=_candidate_price_cents(yes_probability, default=50)),
        "no_price": _candidate_price_cents(candidate.get("no_price"), default=_candidate_price_cents(no_probability, default=50)),
        "volume": max(_safe_float(candidate.get("volume"), _safe_float(candidate.get("volume_24h"), 0.0)), 0.0),
        "days_to_expiry": max(
            _safe_float(candidate.get("hours_to_expiry"), _safe_float(candidate.get("hold_minutes"), 0.0) / 60.0) / 24.0,
            0.01,
        ),
        "rules": "\n".join(summary_lines),
        "news_summary": str(candidate.get("reasoning") or candidate.get("summary") or ""),
    }


def _pooled_fair_yes_probability(
    debate_result: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    skill_weights: Optional[Dict[str, float]] = None,
) -> tuple[float, Optional[float], List[Dict[str, Any]]]:
    """
    Pool the specialist's fair YES probability with the debate researchers'
    probability estimates in log-odds space. The specialist saw the full
    research payload, so it carries the largest weight; bull/bear act as
    adversarial correctors.

    Each member's base weight is scaled by its realized-skill multiplier
    (``skill_weights``, from settled-trade Brier scores; missing roles get
    1.0), so members that have demonstrated accuracy gradually earn more
    influence and persistently wrong ones lose it.

    Pooling uses disagreement-aware extremization (agreeing members earn the
    configured extremize correction; disagreeing members fall back to plain
    pooling). Returns ``(probability, disagreement, members)`` where
    disagreement is the member std dev (None when only one usable estimate
    existed) and ``members`` records every pooled estimate for settlement
    scoring.

    Non-pooled debate roles that still emitted a probability (e.g. the risk
    manager) are appended as ``pooled: False`` observers with weight 0: they
    never move this trade's pooled probability, but settlement scoring sees
    their claims, so they accrue the per-category skill history that future
    weighting draws on. The trader never emits a probability (confidence is
    certainty about the action, not a forecast) and is never scored here.
    """
    from src.agents.ensemble import extract_role_probability
    from src.utils.probability_engine import pool_probabilities_adaptive

    multipliers = skill_weights or {}
    role_models = settings.ensemble.get_role_model_map()
    estimates: List[tuple[float, float]] = []
    members: List[Dict[str, Any]] = []

    def _add_member(role: str, probability: float, base_weight: float) -> None:
        weight = base_weight * float(multipliers.get(role, 1.0))
        if weight <= 0:
            return
        estimates.append((probability, weight))
        members.append(
            {
                "role": role,
                "probability": probability,
                "weight": weight,
                "model": role_models.get(role),
                "pooled": True,
            }
        )

    specialist_fair = _safe_float(candidate.get("fair_yes_probability"), 0.0)
    if 0.0 < specialist_fair < 1.0:
        _add_member("specialist", specialist_fair, 0.5)

    step_results = debate_result.get("step_results") or {}
    for role, weight in (("bull_researcher", 0.25), ("bear_researcher", 0.25)):
        result = step_results.get(role) or {}
        if "error" in result:
            continue
        probability = _safe_float(result.get("probability"), -1.0)
        if 0.0 < probability < 1.0:
            _add_member(role, probability, weight)

    pooled_roles = {member["role"] for member in members}
    for role, result in step_results.items():
        role_name = str(role)
        if role_name in pooled_roles:
            continue
        observed = extract_role_probability(
            role_name, result if isinstance(result, dict) else {}
        )
        if observed is None or not (0.0 < observed < 1.0):
            continue
        members.append(
            {
                "role": role_name,
                "probability": observed,
                "weight": 0.0,
                "model": role_models.get(role_name),
                "pooled": False,
            }
        )

    extremize = float(getattr(settings.ensemble, "extremize_factor", 1.2) or 1.2)
    pooled = pool_probabilities_adaptive(estimates, extremize=extremize)
    if pooled is None:
        fallback = _clamp(specialist_fair if specialist_fair > 0 else 0.5, lo=0.01, hi=0.99)
        return fallback, None, members
    disagreement = pooled.disagreement if pooled.num_members >= 2 else None
    return pooled.probability, disagreement, members


def _normalize_sports_label(value: Any) -> str:
    """Lowercase and collapse non-alphanumerics for fuzzy team matching."""
    lowered = str(value or "").lower()
    cleaned = "".join(ch if ch.isalnum() else " " for ch in lowered)
    return " ".join(cleaned.split())


def _looks_like_spread_or_total(label: str) -> bool:
    """
    True when a normalized market label looks like a spread/total/score market
    rather than a plain game-winner â€” those must not receive a moneyline win
    probability. Conservative (any numeric token or wagering keyword skips).
    """
    tokens = label.split()
    if any(tok.isdigit() for tok in tokens):
        return True
    keywords = {
        "by", "margin", "points", "point", "spread", "total", "over", "under",
        "combined", "handicap", "plus", "minus", "odd", "even", "ot", "overtime",
    }
    return any(tok in keywords for tok in tokens)


def _team_matches_label(team: Any, label: str) -> bool:
    """
    True when ``label`` (already normalized) unambiguously names this team via
    its full display name, its distinctive nickname token, or its abbreviation.
    """
    if not isinstance(team, dict):
        return False
    label_tokens = label.split()
    display = _normalize_sports_label(team.get("display_name") or "")
    if display and display in label:
        return True
    if display:
        nickname = display.split()[-1] if display.split() else ""
        if len(nickname) >= 4 and nickname in label_tokens:
            return True
    abbr = _normalize_sports_label(team.get("abbreviation") or "")
    if abbr and len(abbr) >= 3 and abbr in label_tokens:
        return True
    return False


def _polymarket_trade_age_seconds(value: Any) -> Optional[float]:
    """Age in seconds of a Polymarket last-trade ISO timestamp; None if absent."""
    if not value:
        return None
    try:
        observed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - observed).total_seconds())


def _top_of_book_bid_size(orderbook: Dict[str, Any], side: str) -> float:
    """
    Total quantity resting at the best bid for one orderbook side.

    Accepts both ``{side}_dollars`` (price in dollars) and ``{side}``
    (price in cents) level arrays, mirroring the quick-flip normalizer.
    """
    raw_levels = orderbook.get(f"{side.lower()}_dollars", orderbook.get(side.lower(), [])) or []
    levels: List[tuple[float, float]] = []
    for level in raw_levels:
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            continue
        try:
            price = float(level[0])
            size = float(level[1])
        except (TypeError, ValueError):
            continue
        if price > 1.0:
            price = price / 100.0
        if price <= 0 or size <= 0:
            continue
        levels.append((price, size))
    if not levels:
        return 0.0
    best_price = max(price for price, _ in levels)
    return sum(size for price, size in levels if abs(price - best_price) < 1e-9)


def _debate_final_payload(
    debate_result: Dict[str, Any],
    *,
    candidate: Dict[str, Any],
    skill_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    raw_action = str(debate_result.get("action") or "SKIP").upper()
    action = "BUY" if raw_action in {"BUY", "SELL"} else "SKIP"
    side = str(debate_result.get("side") or candidate.get("side") or "YES").upper()
    if side not in {"YES", "NO"}:
        side = str(candidate.get("side") or "YES").upper()
    limit_price = _candidate_price_cents(
        debate_result.get("limit_price"),
        default=_candidate_price_cents(candidate.get("limit_price"), default=50),
    ) / 100.0
    execution_style = str(candidate.get("execution_style") or "NONE").upper()
    if action != "BUY":
        execution_style = "NONE"

    step_results = debate_result.get("step_results") or {}
    fair_yes_probability, fair_yes_disagreement, member_probabilities = (
        _pooled_fair_yes_probability(
            debate_result, candidate, skill_weights=skill_weights
        )
    )
    selected_candidate = {
        "event_ticker": candidate.get("event_ticker"),
        "market_ticker": candidate.get("market_ticker"),
        "focus_type": candidate.get("focus_type"),
        "execution_style": candidate.get("execution_style"),
        "hold_minutes": candidate.get("hold_minutes"),
    }
    summary = (
        f"Debate selected {candidate.get('market_ticker') or candidate.get('title') or 'the candidate'} for entry."
        if action == "BUY"
        else str(candidate.get("summary") or "Debate skipped the strongest live-trade specialist candidate.")
    )
    return {
        "summary": summary,
        "action": action,
        "event_ticker": str(candidate.get("event_ticker") or ""),
        "market_ticker": str(candidate.get("market_ticker") or ""),
        "side": side,
        "fair_yes_probability": fair_yes_probability,
        "fair_yes_disagreement": fair_yes_disagreement,
        "member_probabilities": member_probabilities,
        "confidence": _clamp(_safe_float(debate_result.get("confidence"), candidate.get("confidence")), lo=0.0, hi=1.0),
        "edge_pct": _safe_float(candidate.get("edge_pct"), 0.0),
        "position_size_pct": _clamp(
            _safe_float(debate_result.get("position_size_pct"), candidate.get("position_size_pct")),
            lo=0.0,
            hi=100.0,
        ),
        "hold_minutes": max(_safe_int(candidate.get("hold_minutes"), 0), 0),
        "limit_price": limit_price,
        "execution_style": execution_style,
        "reasoning": str(debate_result.get("reasoning") or candidate.get("reasoning") or ""),
        "debate_transcript": str(debate_result.get("debate_transcript") or ""),
        "step_results": step_results,
        "elapsed_seconds": debate_result.get("elapsed_seconds"),
        "selected_candidate": selected_candidate,
    }


@dataclass
class LiveTradeLoopSummary:
    run_id: str
    events_scanned: int = 0
    shortlisted_events: int = 0
    specialist_candidates: int = 0
    executed_positions: int = 0
    skipped_reason: Optional[str] = None


# Topic strings that the dashboard SSE hub understands. Keep in sync with the
# Node-side `internalRefreshBodySchema` in `server/src/app.ts`.
_NOTIFY_TOPICS = ("live-trade-decisions", "runtime-state", "feedback")


def _resolve_notify_url() -> Optional[str]:
    """Return the configured Node refresh hook URL, if any.

    We intentionally keep this an env-driven setting (rather than baking it into
    `settings.trading`) because the Python loop and the Node dashboard often
    live in different process trees -- in dev / CI the Node server may be off
    entirely, in which case we want the loop to skip notifications silently.
    """
    raw = (os.getenv("LIVE_TRADE_NOTIFY_URL") or "").strip()
    if not raw:
        return None
    return raw


def _resolve_notify_token() -> Optional[str]:
    raw = (os.getenv("LIVE_TRADE_INTERNAL_REFRESH_TOKEN") or "").strip()
    if not raw:
        return None
    return raw


class LiveTradeRefreshNotifier:
    """Best-effort fire-and-forget POST to the Node SSE hub.

    The Node side already serves a lightweight cursor poll as a fallback, so
    every method here is safe to fail silently: the dashboard will simply
    refresh on the next poll tick instead of immediately. We log a single
    warning per failure (de-duped on consecutive identical warnings) so an
    unreachable Node server does not flood logs while the Python loop runs.
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        token: Optional[str] = None,
        timeout_seconds: float = 0.5,
        client_factory: Optional[Callable[..., httpx.AsyncClient]] = None,
        logger=None,
    ) -> None:
        self._url = url if url is not None else _resolve_notify_url()
        self._token = token if token is not None else _resolve_notify_token()
        self._timeout_seconds = max(0.05, float(timeout_seconds))
        self._client_factory = client_factory
        self._logger = logger or get_trading_logger("live_trade_notify")
        self._last_warning: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self._url) and bool(self._token)

    async def notify(self, topic: str = "live-trade-decisions") -> bool:
        """POST a refresh notification to the Node hub.

        Returns True when the request was sent and accepted, False otherwise.
        Network / HTTP failures are swallowed so the trading loop never crashes
        on a missing dashboard.
        """
        if not self.enabled:
            return False
        normalized_topic = topic if topic in _NOTIFY_TOPICS else "live-trade-decisions"
        try:
            if self._client_factory is not None:
                client = self._client_factory(timeout=self._timeout_seconds)
            else:
                client = httpx.AsyncClient(timeout=self._timeout_seconds)
            try:
                response = await client.post(
                    self._url,  # type: ignore[arg-type]
                    json={"topic": normalized_topic},
                    headers={"x-internal-token": self._token or ""},
                )
            finally:
                await client.aclose()
        except Exception as exc:  # pragma: no cover - defensive: must never raise
            self._warn_once(f"live-trade refresh notify failed: {exc}")
            return False

        status_code = getattr(response, "status_code", None)
        if status_code is None or int(status_code) >= 400:
            self._warn_once(
                f"live-trade refresh notify rejected with status {status_code}"
            )
            return False
        # Reset the de-dupe key on success so the next failure surfaces.
        self._last_warning = None
        return True

    def _warn_once(self, message: str) -> None:
        if message == self._last_warning:
            return
        self._last_warning = message
        try:
            self._logger.warning(message)
        except Exception:
            # The logger should never break the trading loop either.
            pass


class LiveTradeDecisionLoop:
    """Runtime-aware W5 decision loop for the existing trading cycle."""

    def __init__(
        self,
        *,
        db_manager: DatabaseManager,
        kalshi_client: KalshiClient,
        model_router: Optional[ModelRouter] = None,
        research_service: Optional[LiveTradeResearchService] = None,
        execute_position_fn: Optional[Callable[..., Awaitable[bool]]] = None,
        guardrail_fn: Optional[Callable[..., Awaitable[tuple[bool, Optional[str]]]]] = None,
        quick_flip_executor_fn: Optional[Callable[..., Awaitable[Dict[str, Any]]]] = None,
        manage_quick_flip_positions_each_cycle: bool = False,
        shortlist_size: int = 3,
        refresh_notifier: Optional[LiveTradeRefreshNotifier] = None,
    ) -> None:
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.model_router = model_router or ModelRouter(db_manager=db_manager)
        self.research_service = research_service or LiveTradeResearchService(
            kalshi_client=kalshi_client,
            model_router=self.model_router,
        )
        self.execute_position_fn = execute_position_fn or execute_position
        self.guardrail_fn = guardrail_fn
        self.quick_flip_executor_fn = quick_flip_executor_fn or _execute_quick_flip_paper_intent
        self.manage_quick_flip_positions_each_cycle = manage_quick_flip_positions_each_cycle
        self.shortlist_size = max(1, shortlist_size)
        self.logger = get_trading_logger("live_trade_loop")
        self._owns_model_router = model_router is None
        self._owns_research_service = research_service is None
        self._runtime_state: Optional[LiveTradeRuntimeState] = None
        self.refresh_notifier = refresh_notifier or LiveTradeRefreshNotifier(
            logger=self.logger
        )
        self._refresh_notify_tasks: set[asyncio.Task] = set()
        # Cached settlement-calibration reliability slopes keyed by market
        # type ("" = global): key -> (slope, expires_at).
        self._calibration_slope_cache: Dict[str, tuple[float, float]] = {}
        # Cached per-role skill-weight multipliers: (weights, expires_at).
        self._model_skill_cache: Dict[str, tuple[Dict[str, float], float]] = {}
        # Deterministic weather-model probabilities harvested from the
        # specialist research payloads: market_ticker -> probability entry
        # (+ "cached_at" epoch seconds). Consumed by the EV gate.
        self._weather_model_probs: Dict[str, Dict[str, Any]] = {}
        # De-vigged sportsbook implied win probabilities harvested from the
        # research payloads: market_ticker -> probability entry (+ "cached_at").
        # Consumed by the EV gate the same way as weather.
        self._sports_model_probs: Dict[str, Dict[str, Any]] = {}
        # Cross-market (Polymarket) YES prices harvested from the research
        # payloads: market_ticker -> entry (+ "cached_at"). Pooled into the EV
        # gate as a second external prior when CROSS_MARKET_POOL_ENABLED.
        self._cross_market_probs: Dict[str, Dict[str, Any]] = {}

    async def _get_calibration_slope(self, market_type: Optional[str] = None) -> float:
        """
        Reliability slope from realized live-trade settlements, cached for
        30 minutes. 1.0 means no shrink (perfectly calibrated or not enough
        data); lower values shrink model probabilities toward 0.5 before the
        EV gate, so an overconfident strategy automatically trades less.

        When ``market_type`` is provided and that category has accumulated
        enough settled samples, the category-specific slope is used â€”
        categories where the models are systematically overconfident (e.g.
        economics) shrink harder than ones where they are sharp.
        """
        if not bool(getattr(settings.trading, "calibration_shrink_enabled", True)):
            return 1.0

        cache_key = (market_type or "").strip().lower()
        now = datetime.now(timezone.utc).timestamp()
        cached = self._calibration_slope_cache.get(cache_key)
        if cached is not None:
            slope, expires_at = cached
            if expires_at > now:
                return slope

        from src.utils.probability_engine import (
            MIN_CALIBRATION_SAMPLES,
            calibration_shrink_slope,
        )

        slope = 1.0
        try:
            # Rebuild calibration rows from closed trades so the feedback loop
            # runs autonomously (the CLI/operator refresh is manual-only).
            try:
                await self.db_manager.refresh_settlement_calibration()
            except Exception as refresh_exc:
                self.logger.debug(
                    "Settlement calibration refresh skipped", error=str(refresh_exc)
                )
            samples: List[tuple[float, int]] = []
            if cache_key:
                samples = await self.db_manager.get_calibration_samples(
                    strategy="live_trade", market_type=cache_key, limit=500
                )
                if len(samples) < MIN_CALIBRATION_SAMPLES:
                    samples = await self.db_manager.get_calibration_samples(
                        market_type=cache_key, limit=500
                    )
            if len(samples) < MIN_CALIBRATION_SAMPLES:
                samples = await self.db_manager.get_calibration_samples(
                    strategy="live_trade", limit=500
                )
            if len(samples) < MIN_CALIBRATION_SAMPLES:
                samples = await self.db_manager.get_calibration_samples(limit=500)
            slope = calibration_shrink_slope(samples)
        except Exception as exc:
            self.logger.debug("Calibration slope lookup failed", error=str(exc))

        self._calibration_slope_cache[cache_key] = (slope, now + 1800.0)
        return slope

    async def _get_model_skill_weights(
        self, market_type: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Per-role pooling-weight multipliers from realized Brier scores,
        cached for 30 minutes per category. Category-level evidence (the
        candidate's focus type, normalized like settlement calibration)
        refines each role's global multiplier via hierarchical shrinkage,
        so "sharp on weather, dull on sports" shows up in the weights
        instead of averaging away. Roles without enough settled
        observations (or when skill weighting is disabled) get no entry â€”
        pooling treats a missing role as multiplier 1.0, so this fails
        open to the configured base weights.
        """
        if not bool(getattr(settings.trading, "model_skill_weighting_enabled", True)):
            return {}

        from src.utils.database import normalize_market_type

        cache_key = normalize_market_type(market_type)
        now = datetime.now(timezone.utc).timestamp()
        cached = self._model_skill_cache.get(cache_key)
        if cached is not None:
            weights, expires_at = cached
            if expires_at > now:
                return weights

        from src.utils.probability_engine import (
            category_skill_weight_multipliers,
            skill_weight_multipliers,
        )

        weights: Dict[str, float] = {}
        try:
            global_summary = await self.db_manager.get_model_skill_summary()
            if cache_key != "unknown":
                category_summary = await self.db_manager.get_model_skill_summary(
                    market_type=cache_key
                )
                weights = category_skill_weight_multipliers(
                    global_summary, category_summary
                )
            else:
                weights = skill_weight_multipliers(global_summary)
            if weights:
                self.logger.info(
                    "Applying realized-skill pooling weights",
                    market_type=cache_key,
                    weights=weights,
                )
        except Exception as exc:
            self.logger.debug("Model skill weight lookup failed", error=str(exc))

        self._model_skill_cache[cache_key] = (weights, now + 1800.0)
        return weights

    def _harvest_weather_model_probabilities(self, payload: Dict[str, Any]) -> None:
        """
        Pull deterministic per-bucket weather probabilities out of a research
        payload so the EV gate can pool them with the LLM estimate later in
        the same run (and shortly after â€” entries expire in 30 minutes).
        """
        try:
            weather_context = payload.get("weather_context") or {}
            signals = weather_context.get("signals") or {}
            probabilities = signals.get("market_probabilities") or {}
            if not isinstance(probabilities, dict):
                return
            now = datetime.now(timezone.utc).timestamp()
            for ticker, entry in probabilities.items():
                if not isinstance(entry, dict):
                    continue
                model_prob = _safe_float(entry.get("model_yes_probability"), -1.0)
                if not (0.0 <= model_prob <= 1.0):
                    continue
                self._weather_model_probs[str(ticker)] = {**entry, "cached_at": now}
        except Exception as exc:  # telemetry only â€” never break the loop
            self.logger.debug("Weather probability harvest failed", error=str(exc))

    def _weather_model_entry(
        self, market_ticker: str, *, max_age_seconds: float = 1800.0
    ) -> Optional[Dict[str, Any]]:
        """Fresh deterministic weather estimate for a ticker, if we have one."""
        entry = self._weather_model_probs.get(str(market_ticker or ""))
        if not entry:
            return None
        cached_at = _safe_float(entry.get("cached_at"), 0.0)
        if datetime.now(timezone.utc).timestamp() - cached_at > max_age_seconds:
            return None
        return entry

    def _harvest_sports_model_probabilities(self, payload: Dict[str, Any]) -> None:
        """
        Bind de-vigged sportsbook win probabilities to the per-team game-winner
        market tickers so the EV gate can pool the sharpest public prior with
        the LLM estimate (same mechanism as weather).

        Fails closed: needs both de-vigged moneyline legs and an unambiguous
        team->ticker match (exactly one of the two teams named in the market's
        YES label, and no spread/total wording). In-game events are skipped by
        default because ESPN's scoreboard odds block is a stale pre-game line.
        """
        try:
            if not bool(getattr(settings.sports, "enabled", True)):
                return
            sports_context = payload.get("sports_context") or {}
            signals = sports_context.get("signals") or {}
            odds = signals.get("odds") or {}
            if not isinstance(odds, dict):
                return
            home_prob = _safe_float(odds.get("home_implied_win_probability"), -1.0)
            away_prob = _safe_float(odds.get("away_implied_win_probability"), -1.0)
            if not (0.0 < home_prob < 1.0) or not (0.0 < away_prob < 1.0):
                return  # need both de-vigged legs

            quality = 0.85  # title->team mapping is fuzzier than weather's per-ticker map
            if bool(signals.get("is_live")):
                quality = min(
                    quality,
                    float(getattr(settings.sports, "in_game_max_quality", 0.0) or 0.0),
                )
            if quality <= 0.0:
                return  # in-game / floored -> pre-game line is stale, do not pool

            teams = (
                ("home", odds.get("home_team") or {}, home_prob),
                ("away", odds.get("away_team") or {}, away_prob),
            )
            now = datetime.now(timezone.utc).timestamp()
            event = payload.get("event") or {}
            for market in (event.get("markets") or []):
                ticker = str(market.get("ticker") or "").strip()
                if not ticker:
                    continue
                label = _normalize_sports_label(
                    f"{market.get('yes_sub_title') or ''} {market.get('title') or ''}"
                )
                if not label or _looks_like_spread_or_total(label):
                    continue
                matched = [
                    (side_name, prob)
                    for side_name, team, prob in teams
                    if _team_matches_label(team, label)
                ]
                if len(matched) != 1:
                    continue  # 0 or 2 matches -> ambiguous, skip (fail-closed)
                side_name, model_prob = matched[0]
                self._sports_model_probs[ticker] = {
                    "model_yes_probability": _clamp(model_prob, lo=0.01, hi=0.99),
                    "quality": quality,
                    "source": odds.get("provider") or "espn_odds",
                    "matched_side": side_name,
                    "cached_at": now,
                }
        except Exception as exc:  # telemetry only â€” never break the loop
            self.logger.debug("Sports probability harvest failed", error=str(exc))

    def _sports_model_entry(
        self, market_ticker: str, *, max_age_seconds: float = 1800.0
    ) -> Optional[Dict[str, Any]]:
        """Fresh de-vigged sportsbook estimate for a ticker, if we have one."""
        entry = self._sports_model_probs.get(str(market_ticker or ""))
        if not entry:
            return None
        cached_at = _safe_float(entry.get("cached_at"), 0.0)
        if datetime.now(timezone.utc).timestamp() - cached_at > max_age_seconds:
            return None
        return entry

    def _harvest_cross_market_probabilities(self, payload: Dict[str, Any]) -> None:
        """
        Harvest matched Polymarket YES prices from the research payload so the
        EV gate can pool an independent market's price as a second prior. Only
        fresh, liquid, high-confidence matches are kept; everything else is
        skipped (fail-closed). Gated by ``cross_market_pool_enabled``.
        """
        try:
            if not bool(getattr(settings.trading, "cross_market_pool_enabled", False)):
                return
            context = payload.get("cross_market_context") or {}
            matches = context.get("matches") or []
            if not isinstance(matches, list):
                return
            min_conf = float(
                getattr(settings.trading, "cross_market_min_mapping_confidence", 0.65) or 0.65
            )
            min_volume = float(
                getattr(settings.trading, "cross_market_min_volume_usd", 2000.0) or 0.0
            )
            max_age = float(
                getattr(settings.trading, "cross_market_max_age_seconds", 900.0) or 0.0
            )
            now = datetime.now(timezone.utc).timestamp()
            for match in matches:
                if not isinstance(match, dict):
                    continue
                ticker = str(match.get("kalshi_ticker") or "").strip()
                if not ticker:
                    continue
                yes_price = _safe_float(match.get("polymarket_yes_price"), -1.0)
                if not (0.0 < yes_price < 1.0):
                    continue
                confidence = _safe_float(match.get("mapping_confidence"), 0.0)
                if confidence < min_conf:
                    continue
                volume = _safe_float(match.get("polymarket_volume_usd"), 0.0)
                if min_volume > 0 and volume < min_volume:
                    continue
                # Staleness guard when a last-trade timestamp is available; a
                # missing timestamp falls back to the 5-min snapshot cache +
                # volume floor (parity convention: missing telemetry skips the
                # check rather than failing the harvest).
                if max_age > 0:
                    age = _polymarket_trade_age_seconds(match.get("polymarket_last_trade_at"))
                    if age is not None and age > max_age:
                        continue
                self._cross_market_probs[ticker] = {
                    "polymarket_yes_probability": _clamp(yes_price, lo=0.01, hi=0.99),
                    "mapping_confidence": confidence,
                    "volume_usd": volume,
                    "cached_at": now,
                }
        except Exception as exc:  # telemetry only â€” never break the loop
            self.logger.debug("Cross-market probability harvest failed", error=str(exc))

    def _cross_market_entry(
        self, market_ticker: str, *, max_age_seconds: float = 1800.0
    ) -> Optional[Dict[str, Any]]:
        """Fresh matched Polymarket price for a ticker, if we have one."""
        entry = self._cross_market_probs.get(str(market_ticker or ""))
        if not entry:
            return None
        cached_at = _safe_float(entry.get("cached_at"), 0.0)
        if datetime.now(timezone.utc).timestamp() - cached_at > max_age_seconds:
            return None
        return entry

    def _resolve_runtime_mode(self) -> str:
        if bool(getattr(settings.trading, "live_trading_enabled", False)):
            return "live"
        if bool(getattr(settings.trading, "shadow_mode_enabled", False)):
            return "shadow"
        return "paper"

    def _is_live_execution_mode(self) -> bool:
        return self._resolve_runtime_mode() == "live"

    def _new_decision_record(self, **kwargs: Any) -> LiveTradeDecision:
        runtime_mode = self._resolve_runtime_mode()
        is_live_trade = runtime_mode == "live"
        return LiveTradeDecision(
            created_at=datetime.now(timezone.utc),
            runtime_mode=runtime_mode,
            paper_trade=not is_live_trade,
            live_trade=is_live_trade,
            **kwargs,
        )

    async def _notify_refresh(self, topic: str = "live-trade-decisions") -> None:
        """Best-effort push notification to the Node SSE hub.

        Schedule `LiveTradeRefreshNotifier.notify` without waiting on network
        I/O. The cursor-poll fallback in `liveStreamHub.ts` keeps the dashboard
        correct even if every push notification is dropped.
        """
        if not self.refresh_notifier.enabled:
            return

        task = asyncio.create_task(self.refresh_notifier.notify(topic))
        self._refresh_notify_tasks.add(task)

        def _discard_notify_task(done: asyncio.Task) -> None:
            self._refresh_notify_tasks.discard(done)
            try:
                done.result()
            except Exception:
                # Notifier already swallows network errors, but defend against
                # surprises (logger misconfig, asyncio cancellation, etc.).
                pass

        task.add_done_callback(_discard_notify_task)

    def _resolve_exchange_env(self) -> Optional[str]:
        exchange_env = getattr(getattr(settings, "api", None), "kalshi_env", None)
        if isinstance(exchange_env, str) and exchange_env.strip():
            return exchange_env.strip().lower()
        return None

    async def close(self) -> None:
        if self._refresh_notify_tasks:
            pending = list(self._refresh_notify_tasks)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            self._refresh_notify_tasks.clear()
        if self._owns_research_service:
            await self.research_service.close()
        if self._owns_model_router:
            await self.model_router.close()

    async def _manage_quick_flip_positions_for_cycle(self) -> None:
        if not self.manage_quick_flip_positions_each_cycle:
            return

        results = await manage_live_quick_flip_positions(
            db_manager=self.db_manager,
            kalshi_client=self.kalshi_client,
            config=_build_quick_flip_config(
                hold_minutes=int(getattr(settings.trading, "quick_flip_max_hold_minutes", 30) or 30)
            ),
        )
        if any(
            [
                int(results.get("positions_closed", 0)),
                int(results.get("positions_voided", 0)),
                int(results.get("positions_synced", 0)),
                int(results.get("orders_cancelled", 0)),
                int(results.get("orders_adjusted", 0)),
                int(results.get("losses_cut", 0)),
            ]
        ):
            self.logger.info(
                "Quick-flip manager processed standalone live-trade loop positions",
                results=results,
            )

    async def _hydrate_runtime_state(self) -> LiveTradeRuntimeState:
        if self._runtime_state is not None:
            return self._runtime_state

        existing = await self.db_manager.get_live_trade_runtime_state()
        if existing is not None:
            self._runtime_state = LiveTradeRuntimeState(
                strategy=str(existing.get("strategy") or "live_trade"),
                worker=str(existing.get("worker") or "decision_loop"),
                heartbeat_at=str(existing.get("heartbeat_at") or datetime.now(timezone.utc).isoformat()),
                runtime_mode=str(existing.get("runtime_mode") or self._resolve_runtime_mode()),
                exchange_env=existing.get("exchange_env") or self._resolve_exchange_env(),
                run_id=existing.get("run_id"),
                loop_status=str(existing.get("loop_status") or "idle"),
                last_started_at=existing.get("last_started_at"),
                last_completed_at=existing.get("last_completed_at"),
                last_step=existing.get("last_step"),
                last_step_at=existing.get("last_step_at"),
                last_step_status=existing.get("last_step_status"),
                last_summary=existing.get("last_summary"),
                last_healthy_at=existing.get("last_healthy_at"),
                last_healthy_step=existing.get("last_healthy_step"),
                latest_execution_at=existing.get("latest_execution_at"),
                latest_execution_status=existing.get("latest_execution_status"),
                error=existing.get("error"),
            )
        else:
            self._runtime_state = LiveTradeRuntimeState(
                heartbeat_at=datetime.now(timezone.utc).isoformat(),
                runtime_mode=self._resolve_runtime_mode(),
                exchange_env=self._resolve_exchange_env(),
            )
        return self._runtime_state

    async def _persist_runtime_state(
        self,
        *,
        run_id: Optional[str] = None,
        loop_status: Optional[str] = None,
        step: Optional[str] = None,
        step_status: Optional[str] = None,
        summary: Optional[str] = None,
        error: Optional[str] = None,
        started: bool = False,
        completed: bool = False,
        healthy: bool = False,
        execution_status: Optional[str] = None,
    ) -> None:
        state = await self._hydrate_runtime_state()
        now = datetime.now(timezone.utc).isoformat()
        state.heartbeat_at = now
        state.runtime_mode = self._resolve_runtime_mode()
        state.exchange_env = self._resolve_exchange_env()
        if run_id is not None:
            state.run_id = run_id
        if loop_status is not None:
            state.loop_status = loop_status
        if started:
            state.last_started_at = now
        if step is not None:
            state.last_step = step
            state.last_step_at = now
        if step_status is not None:
            state.last_step_status = step_status
        if summary is not None:
            state.last_summary = summary
        state.error = error
        if healthy:
            state.last_healthy_at = now
            state.last_healthy_step = step or state.last_step
        if execution_status is not None:
            state.latest_execution_at = now
            state.latest_execution_status = execution_status
        if completed:
            state.last_completed_at = now
        await self.db_manager.upsert_live_trade_runtime_state(state)
        # W9: push the dashboard hub a refresh nudge after every persisted
        # runtime-state change. The Node side decides whether anything
        # actually changed via its cursor, so over-notifying is cheap; the
        # cursor poll fallback covers any dropped pushes.
        await self._notify_refresh("runtime-state")

    async def run_once(self) -> LiveTradeLoopSummary:
        run_id = uuid.uuid4().hex[:12]
        summary = LiveTradeLoopSummary(run_id=run_id)
        await self._persist_runtime_state(
            run_id=run_id,
            loop_status="running",
            step="startup",
            step_status="started",
            summary="Live-trade loop cycle started.",
            started=True,
            healthy=True,
        )

        try:
            await self._manage_quick_flip_positions_for_cycle()

            daily_ai_cost = await self.db_manager.get_daily_ai_cost()
            if daily_ai_cost >= float(getattr(settings.trading, "daily_ai_budget", 0.0) or 0.0):
                summary.skipped_reason = "daily AI budget exhausted"
                self.logger.info("Skipping live-trade loop because daily AI budget is exhausted")
                await self._persist_runtime_state(
                    run_id=run_id,
                    loop_status="completed",
                    step="budget_check",
                    step_status="skipped",
                    summary=summary.skipped_reason,
                    completed=True,
                    healthy=True,
                )
                return summary

            categories = ["Sports", "Financials", "Crypto", "Economics"]
            events = await self.research_service.get_live_trade_events(
                limit=18,
                category_filters=categories,
                max_hours_to_expiry=int(getattr(settings.trading, "live_wagering_max_hours_to_expiry", 12) or 12),
            )
            summary.events_scanned = len(events)
            if not events:
                summary.skipped_reason = "no eligible live-trade events"
                await self._persist_runtime_state(
                    run_id=run_id,
                    loop_status="completed",
                    step="fetch_events",
                    step_status="skipped",
                    summary=summary.skipped_reason,
                    completed=True,
                    healthy=True,
                )
                return summary

            await self._persist_runtime_state(
                run_id=run_id,
                loop_status="running",
                step="fetch_events",
                step_status="completed",
                summary=f"Loaded {len(events)} eligible live-trade events.",
                healthy=True,
            )

            shortlisted = await self._run_scout(run_id=run_id, events=events)
            summary.shortlisted_events = len(shortlisted)
            if not shortlisted:
                summary.skipped_reason = "scout found no candidates"
                await self._persist_runtime_state(
                    run_id=run_id,
                    loop_status="completed",
                    step="scout",
                    step_status="skipped",
                    summary=summary.skipped_reason,
                    completed=True,
                    healthy=True,
                )
                return summary

            specialist_candidates: List[Dict[str, Any]] = []
            for event in shortlisted:
                specialist = await self._run_specialist(run_id=run_id, event=event)
                if specialist.get("action") == "TRADE":
                    candidate = dict(specialist)
                    selected_market = next(
                        (
                            market
                            for market in (event.get("markets") or [])
                            if str(market.get("ticker")) == str(candidate.get("market_ticker"))
                        ),
                        (event.get("markets") or [{}])[0],
                    )
                    candidate["event_ticker"] = event.get("event_ticker", "")
                    candidate["title"] = event.get("title", "")
                    candidate["category"] = event.get("category", "")
                    candidate["focus_type"] = event.get("focus_type", "")
                    candidate["hours_to_expiry"] = event.get("hours_to_expiry")
                    candidate["market_title"] = selected_market.get("title") or candidate.get("market_ticker")
                    candidate["yes_price"] = selected_market.get("yes_midpoint")
                    candidate["no_price"] = _market_side_entry_price(selected_market or {}, "NO")
                    candidate["volume"] = selected_market.get("volume") or selected_market.get("volume_24h")
                    specialist_candidates.append(candidate)

            summary.specialist_candidates = len(specialist_candidates)
            final_intent = await self._run_final_synth(
                run_id=run_id,
                candidates=specialist_candidates,
            )
            if final_intent.get("action") != "BUY":
                summary.skipped_reason = final_intent.get("summary") or "final synth skipped"
                await self._persist_runtime_state(
                    run_id=run_id,
                    loop_status="completed",
                    step="final",
                    step_status="skipped",
                    summary=summary.skipped_reason,
                    completed=True,
                    healthy=True,
                )
                return summary

            event_map = {str(event.get("event_ticker")): event for event in shortlisted}
            executed = await self._execute_final_intent(
                run_id=run_id,
                final_intent=final_intent,
                event_map=event_map,
            )
            summary.executed_positions = 1 if executed else 0
            if not executed and summary.skipped_reason is None:
                summary.skipped_reason = (
                    "live execution did not fill"
                    if self._is_live_execution_mode()
                    else "paper execution did not fill"
                )
            return summary
        except Exception as exc:
            await self._persist_runtime_state(
                run_id=run_id,
                loop_status="error",
                step=self._runtime_state.last_step if self._runtime_state is not None else "runtime",
                step_status="error",
                summary="Live-trade loop cycle failed.",
                error=str(exc),
                completed=True,
            )
            raise

    async def _run_scout(
        self,
        *,
        run_id: str,
        events: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        candidates = _heuristic_shortlist(events, shortlist_size=min(len(events), 8))
        candidate_payload = [
            {
                "event_ticker": event.get("event_ticker"),
                "title": event.get("title"),
                "category": event.get("category"),
                "focus_type": event.get("focus_type"),
                "hours_to_expiry": event.get("hours_to_expiry"),
                "live_score": event.get("live_score"),
                "volume_24h": event.get("volume_24h"),
                "avg_yes_spread": event.get("avg_yes_spread"),
            }
            for event in candidates
        ]
        prompt = (
            f"You are the scout in a {_resolve_quick_flip_runtime_label()} live prediction-market trading loop.\n"
            "Rank the best short-dated events to send to specialists.\n"
            "Prefer in-play catalysts, tight spreads, real volume, and actionable time windows.\n"
            "Return only JSON.\n\n"
            f"Candidates:\n{json.dumps(candidate_payload, default=str)}"
        )

        raw = await self.model_router.get_completion(
            prompt=prompt,
            capability="cheap",
            strategy="live_trade",
            query_type="live_trade_scout",
            response_format=_response_format("live_trade_scout", SCOUT_SCHEMA),
        )
        parsed = _parse_json_payload(raw)
        selected_ids = [
            item.get("event_ticker")
            for item in (parsed or {}).get("selected_events", [])
            if item.get("event_ticker")
        ]
        if not selected_ids:
            selected_ids = [event.get("event_ticker") for event in candidates[: self.shortlist_size]]
            metadata = {"provider": "heuristic", "model": None}
            parsed = {
                "summary": "Scout fell back to heuristic event ranking.",
                "selected_events": [
                    {
                        "event_ticker": event.get("event_ticker"),
                        "priority": index + 1,
                        "reason": "High live score, volume, and manageable spread.",
                    }
                    for index, event in enumerate(candidates[: self.shortlist_size])
                ],
            }
        else:
            metadata = _extract_router_metadata(self.model_router)

        chosen = [
            event for event in events if str(event.get("event_ticker")) in set(selected_ids[: self.shortlist_size])
        ]
        await self.db_manager.add_live_trade_decision(
            self._new_decision_record(
                run_id=run_id,
                step="scout",
                title="Live-trade scout shortlist",
                provider=metadata.get("provider"),
                model=metadata.get("model"),
                status="completed",
                summary=str(parsed.get("summary", "") if parsed else ""),
                payload_json=json.dumps(parsed or {}, default=str),
            )
        )
        await self._persist_runtime_state(
            run_id=run_id,
            loop_status="running",
            step="scout",
            step_status="completed",
            summary=str(parsed.get("summary", "") if parsed else ""),
            healthy=True,
        )
        return chosen[: self.shortlist_size]

    async def _run_specialist(
        self,
        *,
        run_id: str,
        event: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any]
        try:
            payload = await self.research_service.build_event_research_payload(event)
        except Exception as exc:
            payload = {"event": event}
            self.logger.warning(
                "Falling back to event-only payload for specialist analysis",
                event_ticker=event.get("event_ticker"),
                error=str(exc),
            )

        # Best-effort source-health emission so the safety dashboard can
        # surface sports/crypto/macro/news adapter health alongside the
        # Kalshi/weather snapshots that the execution-safety guard already
        # records. Failures here must not break the specialist step, so the
        # helper swallows DB errors internally.
        try:
            await record_research_payload_snapshots(self.db_manager, payload)
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.debug(
                "Source-health snapshot recording skipped",
                event_ticker=event.get("event_ticker"),
                error=str(exc),
            )

        self._harvest_weather_model_probabilities(payload)
        self._harvest_sports_model_probabilities(payload)
        self._harvest_cross_market_probabilities(payload)

        focus_type = str(event.get("focus_type") or "general").lower()
        specialist_label = {
            "sports": "sports specialist",
            "bitcoin": "crypto specialist",
            "crypto": "crypto specialist",
            "weather": "weather specialist",
        }.get(focus_type, "macro specialist")
        weather_prompt_line = ""
        if focus_type == "weather":
            weather_prompt_line = (
                "The weather_context contains deterministic ensemble-forecast bucket "
                "probabilities (model_yes_probability per ticker) computed from "
                "GFS/ECMWF ensemble members recentered toward the official NWS point "
                "forecast. Anchor your fair_yes_probability on them â€” deviate only "
                "with concrete evidence the model missed (e.g. a frontal timing shift "
                "in the latest discussion), and say why.\n"
            )
        if focus_type == "sports":
            weather_prompt_line += (
                "The sports_context.signals.odds block carries DE-VIGGED sportsbook "
                "implied win probabilities (home_implied_win_probability / "
                "away_implied_win_probability) â€” the sharpest public consensus for "
                "game-winner markets. Treat them as the prior and anchor "
                "fair_yes_probability to the matching team's number; stray only with "
                "concrete in-game evidence the book may lag, and say why.\n"
            )
        prompt = (
            f"You are the {specialist_label} for a short-dated prediction-market bot.\n"
            f"Review the event packet and decide whether there is an actionable {_resolve_quick_flip_runtime_label()} trade right now.\n"
            "Estimate fair_yes_probability â€” your TRUE probability that the market resolves YES â€” "
            "independent of the current price, anchored in the evidence in the packet.\n"
            f"{weather_prompt_line}"
            "Estimate fair value independently from the current market price. Do not automatically "
            "downgrade a candidate to WATCH or SKIP merely because the estimated edge is modest. "
            "If the evidence supports a positive actionable edge, return TRADE and let the downstream "
            "deterministic EV/risk gates decide whether that edge is sufficient after fees.\n"
            "Kalshi taker fees are about 0.07 x P x (1-P) per contract (~1.75c at mid prices); "
            "include realistic entry economics in your estimate, but do not duplicate the final EV gate.\n"
            "Require a real thesis, usable liquidity, and an actionable time window. Use QUICK_FLIP only for sub-30-minute holds.\n"
            "Return only JSON.\n\n"
            f"Event packet:\n{json.dumps(_trim_research_payload(payload), default=str)}"
        )
        raw = await self.model_router.get_completion(
            prompt=prompt,
            capability="fast",
            strategy="live_trade",
            role=focus_type,
            query_type=f"live_trade_{focus_type}_specialist",
            market_id=event.get("event_ticker"),
            response_format=_response_format("live_trade_specialist", SPECIALIST_SCHEMA),
        )
        parsed = _parse_json_payload(raw)
        normalized = _normalize_specialist_payload(parsed, event=event)
        metadata = _extract_router_metadata(self.model_router) if parsed else {"provider": "heuristic", "model": None}
        await self.db_manager.add_live_trade_decision(
            self._new_decision_record(
                run_id=run_id,
                step="specialist",
                event_ticker=event.get("event_ticker"),
                market_ticker=normalized.get("market_ticker"),
                title=event.get("title"),
                focus_type=event.get("focus_type"),
                provider=metadata.get("provider"),
                model=metadata.get("model"),
                status="completed",
                action=normalized.get("action"),
                side=normalized.get("side"),
                confidence=normalized.get("confidence"),
                edge_pct=normalized.get("edge_pct"),
                limit_price=normalized.get("limit_price"),
                hold_minutes=normalized.get("hold_minutes"),
                summary=normalized.get("summary"),
                rationale=normalized.get("reasoning"),
                payload_json=json.dumps(normalized, default=str),
            )
        )
        await self._persist_runtime_state(
            run_id=run_id,
            loop_status="running",
            step="specialist",
            step_status="completed",
            summary=normalized.get("summary"),
            healthy=True,
        )
        return normalized

    async def _run_final_synth(
        self,
        *,
        run_id: str,
        candidates: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not candidates:
            final = _normalize_final_payload(None, candidates=[])
            await self.db_manager.add_live_trade_decision(
                self._new_decision_record(
                    run_id=run_id,
                    step="final",
                    status="skipped",
                    summary=final.get("summary"),
                    rationale=final.get("reasoning"),
                    payload_json=json.dumps(final, default=str),
                )
            )
            await self._persist_runtime_state(
                run_id=run_id,
                loop_status="running",
                step="final",
                step_status="skipped",
                summary=final.get("summary"),
                healthy=True,
            )
            return final

        from src.agents.debate import DebateRunner

        selected_candidate = _selected_candidate_for_debate(candidates)
        if selected_candidate is None:
            final = _normalize_final_payload(None, candidates=candidates)
            await self.db_manager.add_live_trade_decision(
                self._new_decision_record(
                    run_id=run_id,
                    step="final",
                    status="skipped",
                    summary=final.get("summary"),
                    rationale=final.get("reasoning"),
                    payload_json=json.dumps(final, default=str),
                )
            )
            await self._persist_runtime_state(
                run_id=run_id,
                loop_status="running",
                step="final",
                step_status="skipped",
                summary=final.get("summary"),
                healthy=True,
            )
            return final

        available_balance, portfolio_value = await self._load_portfolio_snapshot()
        open_positions = await self.db_manager.get_open_positions()
        portfolio_context = {
            "cash": available_balance,
            "max_trade_value": available_balance * (float(getattr(settings.trading, "max_position_size_pct", 3.0) or 3.0) / 100.0),
            "max_position_pct": float(getattr(settings.trading, "max_position_size_pct", 3.0) or 3.0),
            "existing_positions": len(open_positions or []),
            "portfolio_value": portfolio_value,
            "candidate_count": len(candidates),
        }
        market_data = _candidate_market_data(selected_candidate)
        role_models = settings.ensemble.get_role_model_map()

        async def _make_completion(role: str) -> Callable[..., Awaitable[Optional[str]]]:
            model_name = role_models.get(role)

            async def _fn(prompt: str, **request_options: Any) -> Optional[str]:
                return await self.model_router.get_completion(
                    prompt=prompt,
                    model=model_name,
                    capability="reasoning" if role in {"risk_manager", "trader"} else "fast",
                    strategy="live_trade",
                    role=role,
                    query_type=f"live_trade_final_{role}",
                    market_id=selected_candidate.get("market_ticker") or selected_candidate.get("event_ticker"),
                    **request_options,
                )

            return _fn

        completions = {
            role: await _make_completion(role)
            for role in ("bull_researcher", "bear_researcher", "risk_manager", "trader")
        }
        debate_runner = DebateRunner()
        debate_result = await debate_runner.run_debate(
            market_data=market_data,
            get_completions=completions,
            context={
                "portfolio": portfolio_context,
                "selected_candidate": selected_candidate,
                "specialist_candidates": list(candidates),
            },
        )
        debate_payload = _debate_final_payload(
            debate_result,
            candidate=selected_candidate,
            skill_weights=await self._get_model_skill_weights(
                selected_candidate.get("focus_type")
            ),
        )
        final = _normalize_final_payload(debate_payload, candidates=candidates)
        metadata = _extract_router_metadata(self.model_router) if debate_result else {"provider": "heuristic", "model": None}
        await self.db_manager.add_live_trade_decision(
            self._new_decision_record(
                run_id=run_id,
                step="final",
                event_ticker=final.get("event_ticker"),
                market_ticker=final.get("market_ticker"),
                provider=metadata.get("provider"),
                model=metadata.get("model"),
                status="completed" if final.get("action") == "BUY" else "skipped",
                action=final.get("action"),
                side=final.get("side"),
                confidence=final.get("confidence"),
                edge_pct=final.get("edge_pct"),
                limit_price=final.get("limit_price"),
                hold_minutes=final.get("hold_minutes"),
                summary=final.get("summary"),
                rationale=final.get("reasoning"),
                payload_json=json.dumps(final, default=str),
            )
        )
        await self._persist_runtime_state(
            run_id=run_id,
            loop_status="running",
            step="final",
            step_status="completed" if final.get("action") == "BUY" else "skipped",
            summary=final.get("summary"),
            healthy=True,
        )
        return final

    async def _load_portfolio_snapshot(self) -> tuple[float, float]:
        try:
            balance_response = await self.kalshi_client.get_balance()
            available_balance = max(get_balance_dollars(balance_response), 0.0)
            portfolio_value = max(
                available_balance + get_portfolio_value_dollars(balance_response),
                available_balance,
            )
            return available_balance, portfolio_value
        except Exception:
            open_positions = await self.db_manager.get_open_positions()
            open_exposure = 0.0
            for position in open_positions:
                contracts_cost = _safe_float(getattr(position, "contracts_cost", 0.0), 0.0)
                if contracts_cost <= 0:
                    contracts_cost = _safe_float(getattr(position, "entry_price", 0.0), 0.0) * _safe_float(
                        getattr(position, "quantity", 0.0), 0.0
                    )
                open_exposure += max(contracts_cost, 0.0)
            floor_balance = max(float(getattr(settings.trading, "min_balance", 100.0) or 100.0), 100.0)
            return floor_balance, floor_balance + open_exposure

    async def _passes_guardrails(
        self,
        *,
        market: Market,
        side: str,
        trade_value: float,
        portfolio_value: float,
        strategy: str = "live_trade",
    ) -> tuple[bool, Optional[str]]:
        if self.guardrail_fn is not None:
            return await self.guardrail_fn(
                market=market,
                side=side,
                trade_value=trade_value,
                portfolio_value=portfolio_value,
                db_manager=self.db_manager,
                strategy=strategy,
            )

        from src.jobs.decide import _passes_live_trade_guardrails
        from src.strategies.portfolio_enforcer import MODE_LIVE, MODE_PAPER

        guardrail_mode = (
            MODE_LIVE
            if (
                self._is_live_execution_mode()
                or bool(getattr(settings.trading, "shadow_mode_enabled", False))
            )
            else MODE_PAPER
        )
        return await _passes_live_trade_guardrails(
            market=market,
            side=side,
            trade_value=trade_value,
            portfolio_value=portfolio_value,
            db_manager=self.db_manager,
            enforcement_mode=guardrail_mode,
            strategy=strategy,
        )

    async def _execute_final_intent(
        self,
        *,
        run_id: str,
        final_intent: Dict[str, Any],
        event_map: Dict[str, Dict[str, Any]],
    ) -> bool:
        event_ticker = str(final_intent.get("event_ticker") or "")
        market_ticker = str(final_intent.get("market_ticker") or "")
        selected_event = event_map.get(event_ticker)
        if not selected_event:
            await self._record_execution_status(
                run_id=run_id,
                final_intent=final_intent,
                status="error",
                summary="Final intent references an event that is no longer present.",
                error="missing_event",
            )
            return False

        selected_market = next(
            (market for market in (selected_event.get("markets") or []) if str(market.get("ticker")) == market_ticker),
            None,
        )
        if not selected_market:
            await self._record_execution_status(
                run_id=run_id,
                final_intent=final_intent,
                status="error",
                summary="Final intent references a market that is no longer present.",
                error="missing_market",
            )
            return False

        execution_style = str(final_intent.get("execution_style") or "NONE").upper()
        if execution_style == "NONE":
            await self._record_execution_status(
                run_id=run_id,
                final_intent=final_intent,
                status="skipped",
                summary="Skipped because the final intent requested no execution.",
                error="execution_style_none",
            )
            return False

        existing = await self.db_manager.get_position_by_market_id(market_ticker)
        if existing is not None:
            await self._record_execution_status(
                run_id=run_id,
                final_intent=final_intent,
                status="skipped",
                summary="Skipped because an open position already exists for this market.",
                error="existing_position",
            )
            return False

        available_balance, portfolio_value = await self._load_portfolio_snapshot()
        limit_price = _clamp(_safe_float(final_intent.get("limit_price"), 0.5), lo=0.01, hi=0.99)
        position_size_pct = min(
            _safe_float(final_intent.get("position_size_pct"), 0.0),
            float(getattr(settings.trading, "max_position_size_pct", 3.0) or 3.0),
        )
        if position_size_pct <= 0:
            position_size_pct = min(1.0, float(getattr(settings.trading, "max_position_size_pct", 3.0) or 3.0))
        trade_budget = max(available_balance * (position_size_pct / 100.0), 0.0)
        quantity = int(math.floor(trade_budget / max(limit_price, 0.01)))
        if quantity <= 0:
            await self._record_execution_status(
                run_id=run_id,
                final_intent=final_intent,
                status="skipped",
                summary="Skipped because the calculated position size was below one contract.",
                error="zero_quantity",
            )
            return False

        market_record = Market(
            market_id=market_ticker,
            title=_market_title(selected_market),
            yes_price=_clamp(_safe_float(selected_market.get("yes_midpoint"), 0.5), lo=0.01, hi=0.99),
            no_price=_clamp(_market_side_entry_price(selected_market, "NO"), lo=0.01, hi=0.99),
            volume=max(_safe_int(selected_market.get("volume"), 0), 0),
            expiration_ts=max(_safe_int(selected_market.get("expiration_ts"), int(datetime.now(timezone.utc).timestamp() + 3600)), 0),
            category=str(selected_event.get("category") or "General"),
            status="active",
            last_updated=datetime.now(timezone.utc),
            has_position=False,
        )
        execution_style = str(final_intent.get("execution_style") or "NONE").upper()
        hold_minutes = _safe_int(final_intent.get("hold_minutes"), 0)
        is_quick_flip_intent = execution_style == "QUICK_FLIP"
        is_short_quick_flip = is_quick_flip_intent and hold_minutes <= 30
        live_mode = self._is_live_execution_mode()
        runtime_mode = self._resolve_runtime_mode()
        execution_mode_label = "live" if live_mode else runtime_mode
        if is_quick_flip_intent and not is_short_quick_flip:
            await self._record_execution_status(
                run_id=run_id,
                final_intent=final_intent,
                status="blocked",
                summary="Quick-flip intents must use a hold window of 30 minutes or less.",
                error="quick_flip_hold_exceeds_scalp_window",
                quantity=quantity,
            )
            return False
        if is_short_quick_flip and live_mode and not bool(getattr(settings.trading, "enable_live_quick_flip", False)):
            await self._record_execution_status(
                run_id=run_id,
                final_intent=final_intent,
                status="blocked",
                summary="Quick-flip live execution requires ENABLE_LIVE_QUICK_FLIP opt-in.",
                error="quick_flip_live_opt_in_required",
                quantity=quantity,
            )
            return False
        # ------------------------------------------------------------------
        # Deterministic fee-aware EV gate. The LLM's BUY is a proposal, not
        # an order: its fair probability is calibration-shrunk, blended with
        # the live market price, and the trade must clear a minimum net edge
        # per contract after estimated Kalshi fees.
        # ------------------------------------------------------------------
        from src.utils.probability_engine import evaluate_trade_intent

        intent_side = str(final_intent.get("side") or "YES").upper()
        intent_confidence = _safe_float(final_intent.get("confidence"), 0.0)
        category_label = str(selected_event.get("category") or "default").lower()
        category_multipliers = dict(
            getattr(settings.trading, "category_confidence_adjustments", {}) or {}
        )
        confidence_multiplier = float(
            category_multipliers.get(category_label, category_multipliers.get("default", 1.0))
        )
        min_confidence = (
            float(getattr(settings.trading, "live_trade_min_confidence", 0.55) or 0.55)
            * confidence_multiplier
        )
        if intent_confidence < min_confidence:
            await self._record_execution_status(
                run_id=run_id,
                final_intent=final_intent,
                status="blocked",
                summary=(
                    f"Confidence {intent_confidence:.2f} below the {category_label} "
                    f"minimum {min_confidence:.2f}."
                ),
                error="confidence_below_minimum",
                quantity=quantity,
            )
            return False

        market_yes_mid = _clamp(
            _safe_float(selected_market.get("yes_midpoint"), 0.5), lo=0.01, hi=0.99
        )
        fair_yes_for_gate = _safe_float(
            final_intent.get("fair_yes_probability"), market_yes_mid
        )

        # ------------------------------------------------------------------
        # Weather override: when a fresh deterministic ensemble-forecast
        # probability exists for this market, it dominates the LLM estimate
        # (log-odds pooling, weight scaled by estimate quality), and entries
        # beyond the configured forecast horizon are refused outright.
        # ------------------------------------------------------------------
        weather_entry = self._weather_model_entry(market_ticker)
        weather_pooled_from: Optional[Dict[str, Any]] = None
        if weather_entry is not None and bool(getattr(settings.weather, "enabled", True)):
            diagnostics = weather_entry.get("diagnostics") or {}
            lead_days = _safe_float(diagnostics.get("lead_days"), 0.0)
            max_lead = float(getattr(settings.weather, "max_lead_days", 6) or 6)
            if lead_days > max_lead:
                await self._record_execution_status(
                    run_id=run_id,
                    final_intent=final_intent,
                    status="blocked",
                    summary=(
                        f"Weather contract resolves {lead_days:.0f} days out â€” beyond "
                        f"the {max_lead:.0f}-day forecast-skill horizon."
                    ),
                    error="weather_lead_too_far",
                    quantity=quantity,
                )
                return False

            model_prob = _clamp(
                _safe_float(weather_entry.get("model_yes_probability"), 0.5),
                lo=0.01,
                hi=0.99,
            )
            quality = _clamp(_safe_float(weather_entry.get("quality"), 0.0), lo=0.0, hi=1.0)
            min_quality = float(
                getattr(settings.weather, "min_quality_to_pool", 0.35) or 0.35
            )
            if quality >= min_quality:
                from src.utils.probability_engine import pool_probabilities

                pool_weight = _clamp(
                    float(getattr(settings.weather, "model_pool_weight", 0.75) or 0.75)
                    * quality,
                    lo=0.0,
                    hi=0.95,
                )
                pooled = pool_probabilities(
                    [
                        (fair_yes_for_gate, 1.0 - pool_weight),
                        (model_prob, pool_weight),
                    ],
                    extremize=1.0,
                )
                if pooled is not None:
                    weather_pooled_from = {
                        "llm_fair_yes_probability": fair_yes_for_gate,
                        "weather_model_yes_probability": model_prob,
                        "weather_model_quality": quality,
                        "weather_model_method": weather_entry.get("method"),
                        "pool_weight": pool_weight,
                    }
                    fair_yes_for_gate = pooled.probability
                    self.logger.info(
                        "Pooled deterministic weather probability into EV gate",
                        market_ticker=market_ticker,
                        llm_fair=weather_pooled_from["llm_fair_yes_probability"],
                        weather_model=model_prob,
                        pooled=fair_yes_for_gate,
                        quality=quality,
                    )

        # ------------------------------------------------------------------
        # Sportsbook override: de-vigged moneyline implied win probabilities
        # are the sharpest public prior for game-winner markets â€” and sports
        # (NCAAB) is the bot's only proven-profitable niche. When a fresh,
        # unambiguous per-team odds estimate exists for this ticker it is
        # pooled into the LLM estimate (log-odds, weight scaled by quality),
        # exactly like the weather model. Mutually exclusive with weather (an
        # event is one focus type); fails closed to the LLM estimate on any
        # ambiguity, missing leg, in-game staleness, or low quality.
        # ------------------------------------------------------------------
        sports_pooled_from: Optional[Dict[str, Any]] = None
        if weather_pooled_from is None and bool(getattr(settings.sports, "enabled", True)):
            sports_entry = self._sports_model_entry(market_ticker)
            if sports_entry is not None:
                sports_model_prob = _clamp(
                    _safe_float(sports_entry.get("model_yes_probability"), 0.5),
                    lo=0.01,
                    hi=0.99,
                )
                sports_quality = _clamp(
                    _safe_float(sports_entry.get("quality"), 0.0), lo=0.0, hi=1.0
                )
                sports_min_quality = float(
                    getattr(settings.sports, "min_quality_to_pool", 0.5) or 0.5
                )
                if sports_quality >= sports_min_quality:
                    from src.utils.probability_engine import pool_probabilities

                    sports_pool_weight = _clamp(
                        float(getattr(settings.sports, "model_pool_weight", 0.55) or 0.55)
                        * sports_quality,
                        lo=0.0,
                        hi=0.95,
                    )
                    sports_pooled = pool_probabilities(
                        [
                            (fair_yes_for_gate, 1.0 - sports_pool_weight),
                            (sports_model_prob, sports_pool_weight),
                        ],
                        extremize=1.0,
                    )
                    if sports_pooled is not None:
                        sports_pooled_from = {
                            "llm_fair_yes_probability": fair_yes_for_gate,
                            "sportsbook_yes_probability": sports_model_prob,
                            "quality": sports_quality,
                            "source": sports_entry.get("source"),
                            "matched_side": sports_entry.get("matched_side"),
                            "pool_weight": sports_pool_weight,
                        }
                        fair_yes_for_gate = sports_pooled.probability
                        self.logger.info(
                            "Pooled sportsbook probability into EV gate",
                            market_ticker=market_ticker,
                            llm_fair=sports_pooled_from["llm_fair_yes_probability"],
                            sportsbook=sports_model_prob,
                            pooled=fair_yes_for_gate,
                            quality=sports_quality,
                        )

        # ------------------------------------------------------------------
        # Cross-market (Polymarket) anchor: an independent market pricing the
        # same question is a second external prior. OPT-IN and low-weight; only
        # pools when neither a weather nor a sportsbook model already anchored
        # this trade (a market is one focus type, but stay deterministic).
        # Agreement shrinks claimed edge; divergence pulls fair toward the
        # cross-market consensus. Fails closed on any miss.
        # ------------------------------------------------------------------
        cross_market_pooled_from: Optional[Dict[str, Any]] = None
        if (
            weather_pooled_from is None
            and sports_pooled_from is None
            and bool(getattr(settings.trading, "cross_market_pool_enabled", False))
        ):
            cross_entry = self._cross_market_entry(market_ticker)
            if cross_entry is not None:
                poly_prob = _clamp(
                    _safe_float(cross_entry.get("polymarket_yes_probability"), 0.5),
                    lo=0.01,
                    hi=0.99,
                )
                mapping_confidence = _clamp(
                    _safe_float(cross_entry.get("mapping_confidence"), 0.0), lo=0.0, hi=1.0
                )
                from src.utils.probability_engine import pool_probabilities

                weight_base = float(
                    getattr(settings.trading, "cross_market_pool_weight_base", 0.20) or 0.20
                )
                weight_cap = float(
                    getattr(settings.trading, "cross_market_pool_weight_cap", 0.25) or 0.25
                )
                cross_pool_weight = _clamp(
                    weight_base * mapping_confidence, lo=0.0, hi=max(0.0, weight_cap)
                )
                if cross_pool_weight > 0:
                    cross_pooled = pool_probabilities(
                        [
                            (fair_yes_for_gate, 1.0 - cross_pool_weight),
                            (poly_prob, cross_pool_weight),
                        ],
                        extremize=1.0,
                    )
                    if cross_pooled is not None:
                        cross_market_pooled_from = {
                            "llm_fair_yes_probability": fair_yes_for_gate,
                            "polymarket_yes_probability": poly_prob,
                            "mapping_confidence": mapping_confidence,
                            "volume_usd": cross_entry.get("volume_usd"),
                            "pool_weight": cross_pool_weight,
                        }
                        fair_yes_for_gate = cross_pooled.probability
                        self.logger.info(
                            "Pooled cross-market (Polymarket) probability into EV gate",
                            market_ticker=market_ticker,
                            llm_fair=cross_market_pooled_from["llm_fair_yes_probability"],
                            polymarket=poly_prob,
                            pooled=fair_yes_for_gate,
                            mapping_confidence=mapping_confidence,
                        )

        # ------------------------------------------------------------------
        # Microstructure guard: refuse entries into wide or empty books.
        # A wide spread makes the midpoint blend unreliable (the "market
        # probability" is a guess between distant quotes), and a thin top
        # of book means the eventual exit pays the spread all over again.
        # ------------------------------------------------------------------
        yes_bid = _safe_float(selected_market.get("yes_bid"), 0.0)
        yes_ask = _safe_float(selected_market.get("yes_ask"), 0.0)
        spread_cents: Optional[float] = None
        raw_spread = _safe_float(selected_market.get("yes_spread"), -1.0)
        if raw_spread >= 0:
            spread_cents = raw_spread * 100.0 if raw_spread <= 1.0 else float(raw_spread)
        elif 0 < yes_bid <= yes_ask:
            spread_cents = (yes_ask - yes_bid) * 100.0
        max_spread_cents = float(
            getattr(settings.trading, "live_trade_max_spread_cents", 6.0) or 0.0
        )
        if max_spread_cents > 0 and spread_cents is not None and spread_cents > max_spread_cents:
            await self._record_execution_status(
                run_id=run_id,
                final_intent=final_intent,
                status="blocked",
                summary=(
                    f"Bid-ask spread {spread_cents:.0f}c exceeds the "
                    f"{max_spread_cents:.0f}c microstructure cap."
                ),
                error="spread_too_wide",
                quantity=quantity,
                payload={"spread_cents": spread_cents, "max_spread_cents": max_spread_cents},
            )
            return False

        min_top_depth = int(
            getattr(settings.trading, "live_trade_min_top_depth_contracts", 10) or 0
        )
        top_depth_contracts: Optional[float] = None
        if min_top_depth > 0:
            # Depth is enforced identically in paper and live (shadow parity);
            # a failed orderbook *fetch* skips the check in both modes â€” the
            # spread guard and EV gate still bound the damage, and blocking on
            # telemetry hiccups would make paper/live decisions diverge.
            try:
                orderbook_response = await self.kalshi_client.get_orderbook(
                    market_ticker, depth=5
                )
                orderbook = (
                    orderbook_response.get(
                        "orderbook_fp", orderbook_response.get("orderbook", {})
                    )
                    or {}
                )
                top_depth_contracts = _top_of_book_bid_size(orderbook, intent_side)
            except Exception as exc:
                self.logger.warning(
                    "Orderbook depth check unavailable; spread guard and EV gate still apply",
                    market_ticker=market_ticker,
                    error=str(exc),
                )
            if top_depth_contracts is not None and top_depth_contracts < min_top_depth:
                await self._record_execution_status(
                    run_id=run_id,
                    final_intent=final_intent,
                    status="blocked",
                    summary=(
                        f"Top-of-book size {top_depth_contracts:.0f} contracts is below "
                        f"the {min_top_depth} minimum for {intent_side} entries."
                    ),
                    error="orderbook_too_thin",
                    quantity=quantity,
                    payload={
                        "top_depth_contracts": top_depth_contracts,
                        "min_top_depth_contracts": min_top_depth,
                    },
                )
                return False

        # A limit priced inside the spread rests on the book and earns maker
        # fee treatment (1.75% vs 7%); the EV gate should not tax a resting
        # order with taker fees it will never pay.
        side_ask = (
            yes_ask
            if intent_side == "YES"
            else _safe_float(selected_market.get("no_ask"), 0.0)
        )
        expects_maker_entry = bool(side_ask > 0 and limit_price < side_ask - 1e-9)

        raw_disagreement = final_intent.get("fair_yes_disagreement")
        try:
            intent_disagreement = (
                float(raw_disagreement) if raw_disagreement is not None else None
            )
        except (TypeError, ValueError):
            intent_disagreement = None

        # Market-prior calibration: when validated models exist (fit from
        # settled snapshot history), the gate's market anchor uses the
        # calibrated settlement probability instead of the raw mid â€”
        # correcting systematic price biases such as favorite-longshot.
        # Fails closed to the raw mid (identity) when no model is active.
        market_yes_prior = market_yes_mid
        market_prior_segment: Optional[str] = None
        if bool(getattr(settings.trading, "market_prior_calibration_enabled", True)):
            try:
                from src.utils.market_prior import adjusted_market_yes_probability

                market_yes_prior, market_prior_segment = (
                    await adjusted_market_yes_probability(
                        self.db_manager,
                        market_yes_mid,
                        _safe_float(selected_event.get("hours_to_expiry"), 6.0),
                    )
                )
            except Exception as exc:
                self.logger.debug(
                    "Market-prior adjustment unavailable", error=str(exc)
                )
                market_yes_prior = market_yes_mid
                market_prior_segment = None

        # Outcome meta-model: the settlement-trained corrector (the same one
        # decide.py already profits from) refines the pooled fair probability
        # in held-side space before the gate. No-op until it has enough settled
        # samples AND beats the raw LLM Brier in CV (fail-closed). Mode-blind,
        # so paper/live parity holds whether or not it is active. The PRE-meta
        # value stays the recorded calibration label so the model never trains
        # on its own corrected output.
        pre_meta_fair_yes_for_gate = fair_yes_for_gate
        gate_fair_yes = fair_yes_for_gate
        if bool(getattr(settings.trading, "ml_meta_model_enabled", True)):
            try:
                from src.jobs.decide import _apply_ml_meta_model
                from src.utils.probability_engine import side_win_probability

                side_prob = side_win_probability(fair_yes_for_gate, intent_side)
                adjusted_side = await _apply_ml_meta_model(
                    self.db_manager,
                    side_win_probability=side_prob,
                    entry_price=limit_price,
                    side=intent_side,
                    confidence=intent_confidence,
                )
                gate_fair_yes = _clamp(
                    adjusted_side if intent_side == "YES" else 1.0 - adjusted_side,
                    lo=0.01,
                    hi=0.99,
                )
            except Exception as exc:
                self.logger.debug(
                    "Outcome meta-model adjustment unavailable", error=str(exc)
                )
                gate_fair_yes = fair_yes_for_gate

        # Category-tiered net-edge floor: proven-weak categories must clear a
        # higher edge bar so capital concentrates in proven ones. Reuses the
        # category_label already computed for the confidence multiplier.
        base_min_net_edge = float(
            getattr(settings.trading, "live_trade_min_net_edge", 0.02) or 0.0
        )
        edge_multipliers = dict(
            getattr(settings.trading, "category_min_net_edge_multipliers", {}) or {}
        )
        min_net_edge_multiplier = float(
            edge_multipliers.get(category_label, edge_multipliers.get("default", 1.0))
        )
        effective_min_net_edge = max(0.0, base_min_net_edge * min_net_edge_multiplier)

        gate = evaluate_trade_intent(
            fair_yes_probability=gate_fair_yes,
            side=intent_side,
            entry_price=limit_price,
            market_yes_probability=market_yes_prior,
            model_blend_weight=float(
                getattr(settings.ensemble, "market_blend_model_weight", 0.65) or 0.65
            ),
            calibration_slope=await self._get_calibration_slope(
                market_type=str(selected_event.get("category") or "") or None
            ),
            maker=expects_maker_entry,
            include_exit_fee=is_quick_flip_intent,
            min_net_edge=effective_min_net_edge,
            disagreement=intent_disagreement,
        )
        gate_snapshot = {
            # Recorded as the calibration training label â€” deliberately the
            # PRE-meta-model pooled value (see above).
            "fair_yes_probability": pre_meta_fair_yes_for_gate,
            "meta_adjusted_fair_yes_probability": gate_fair_yes,
            "market_yes_midpoint": market_yes_mid,
            "market_yes_prior": market_yes_prior,
            "market_prior_segment": market_prior_segment,
            "shrunk_yes_probability": gate["shrunk_yes_probability"],
            "blended_yes_probability": gate["blended_yes_probability"],
            "win_probability": gate["win_probability"],
            "net_edge": gate["ev"].net_edge,
            "gross_edge": gate["ev"].gross_edge,
            "fees_per_contract": gate["ev"].entry_fee_per_contract
            + gate["ev"].exit_fee_per_contract,
            "expects_maker_entry": expects_maker_entry,
            "spread_cents": spread_cents,
            "top_depth_contracts": top_depth_contracts,
            "disagreement": intent_disagreement,
            "disagreement_edge_padding": gate["disagreement_edge_padding"],
            "min_net_edge_multiplier": min_net_edge_multiplier,
            "effective_min_net_edge": effective_min_net_edge,
            "weather_model": weather_pooled_from,
            "sports_model": sports_pooled_from,
            "cross_market_model": cross_market_pooled_from,
            "member_probabilities": final_intent.get("member_probabilities") or [],
        }
        if not gate["approved"]:
            await self._record_execution_status(
                run_id=run_id,
                final_intent=final_intent,
                status="blocked",
                summary=f"EV gate blocked the trade: {gate['reason']}.",
                error="ev_gate_blocked",
                quantity=quantity,
                payload={
                    "fair_yes_probability": gate["fair_yes_probability"],
                    "entry_price": limit_price,
                    **gate_snapshot,
                },
            )
            return False
        self.logger.info(
            "EV gate approved live-trade intent",
            market_ticker=market_ticker,
            side=intent_side,
            entry_price=limit_price,
            blended_yes_probability=gate["blended_yes_probability"],
            net_edge=gate["ev"].net_edge,
        )

        # Kelly sizing cap. The LLM's position_size_pct is a request, not an
        # entitlement: the funded size can never exceed the fractional-Kelly
        # bankroll fraction implied by the gate's blended win probability at
        # this entry price. Mode-blind (paper/live parity).
        if bool(getattr(settings.trading, "live_trade_kelly_sizing_enabled", True)):
            from src.utils.probability_engine import kelly_fraction

            max_pct_cap = float(
                getattr(settings.trading, "max_position_size_pct", 3.0) or 3.0
            )
            kelly_pct = kelly_fraction(
                win_probability=float(gate["win_probability"]),
                entry_price=limit_price,
                multiplier=float(
                    getattr(settings.trading, "live_trade_kelly_multiplier", 0.25)
                    or 0.25
                ),
                cap=max_pct_cap / 100.0,
            )
            kelly_budget = max(portfolio_value, 0.0) * kelly_pct
            kelly_quantity = int(math.floor(kelly_budget / max(limit_price, 0.01)))
            gate_snapshot["kelly_fraction"] = kelly_pct
            gate_snapshot["kelly_max_quantity"] = kelly_quantity
            if kelly_quantity < quantity:
                self.logger.info(
                    "Kelly cap reduced live-trade position size",
                    market_ticker=market_ticker,
                    requested_quantity=quantity,
                    kelly_quantity=kelly_quantity,
                    kelly_fraction=kelly_pct,
                )
                gate_snapshot["kelly_capped"] = True
                quantity = kelly_quantity
            if quantity <= 0:
                await self._record_execution_status(
                    run_id=run_id,
                    final_intent=final_intent,
                    status="blocked",
                    summary=(
                        "Kelly sizing allows less than one contract at the "
                        "gate's blended win probability â€” the measured edge "
                        "does not support a position."
                    ),
                    error="kelly_zero_size",
                    quantity=quantity,
                    payload={"gate_snapshot": gate_snapshot},
                )
                return False

        guardrail_strategy = "quick_flip" if is_short_quick_flip else "live_trade"
        allowed, reason = await self._passes_guardrails(
            market=market_record,
            side=str(final_intent.get("side") or "YES"),
            trade_value=quantity * limit_price,
            portfolio_value=portfolio_value,
            strategy=guardrail_strategy,
        )
        if not allowed:
            await self._record_execution_status(
                run_id=run_id,
                final_intent=final_intent,
                status="blocked",
                summary=reason or "Portfolio guardrail blocked the live-trade intent.",
                error="guardrail_blocked",
                quantity=quantity,
            )
            return False

        if is_short_quick_flip:
            quick_flip_result = await self.quick_flip_executor_fn(
                db_manager=self.db_manager,
                kalshi_client=self.kalshi_client,
                selected_event=selected_event,
                selected_market=selected_market,
                final_intent=final_intent,
                quantity=quantity,
            )
            await self._record_execution_status(
                run_id=run_id,
                final_intent=final_intent,
                status=str(quick_flip_result.get("status") or "executed"),
                summary=str(
                    quick_flip_result.get("summary")
                    or "Quick-flip execution path handled the live-trade intent."
                ),
                error=quick_flip_result.get("error"),
                quantity=quick_flip_result.get("quantity", quantity),
                payload={
                    **(quick_flip_result.get("payload") or {}),
                    "gate_snapshot": gate_snapshot,
                },
            )
            return bool(quick_flip_result.get("executed"))

        from src.utils.stop_loss_calculator import StopLossCalculator

        exit_plan = StopLossCalculator.calculate_stop_loss_levels(
            entry_price=limit_price,
            side=str(final_intent.get("side") or "YES"),
            confidence=_safe_float(final_intent.get("confidence"), 0.0),
            market_volatility=max(_safe_float(selected_market.get("yes_spread"), 0.05), 0.05),
            time_to_expiry_days=max(_safe_float(selected_event.get("hours_to_expiry"), 6.0) / 24.0, 0.25),
        )
        position = Position(
            market_id=market_ticker,
            side=str(final_intent.get("side") or "YES"),
            entry_price=limit_price,
            quantity=quantity,
            timestamp=datetime.now(),
            rationale=(
                f"W5 live-trade loop | {final_intent.get('summary') or 'live-trade entry'}"
            ),
            confidence=_safe_float(final_intent.get("confidence"), 0.0),
            live=live_mode,
            strategy="live_trade",
            stop_loss_price=exit_plan["stop_loss_price"],
            take_profit_price=exit_plan["take_profit_price"],
            max_hold_hours=max(1, math.ceil(max(_safe_int(final_intent.get("hold_minutes"), 0), 30) / 60)),
            target_confidence_change=exit_plan.get("target_confidence_change"),
        )
        await self.db_manager.upsert_markets([market_record])
        position_id = await self.db_manager.add_position(position)
        if position_id is None:
            await self._record_execution_status(
                run_id=run_id,
                final_intent=final_intent,
                status="skipped",
                summary="Skipped because the market/side position already exists.",
                error="duplicate_position",
                quantity=quantity,
            )
            return False

        position.id = position_id
        success = await self.execute_position_fn(
            position=position,
            live_mode=live_mode,
            db_manager=self.db_manager,
            kalshi_client=self.kalshi_client,
            shadow_mode=runtime_mode == "shadow",
            paper_market_info=selected_market if not live_mode else None,
        )
        if success:
            await self.db_manager.update_position_execution_details(
                position.id,
                entry_price=position.entry_price,
                quantity=position.quantity,
                live=live_mode,
                stop_loss_price=position.stop_loss_price,
                take_profit_price=position.take_profit_price,
                max_hold_hours=position.max_hold_hours,
                entry_fee=position.entry_fee,
                contracts_cost=position.contracts_cost,
                entry_order_id=position.entry_order_id,
            )
            await self._record_execution_status(
                run_id=run_id,
                final_intent=final_intent,
                status="executed",
                summary=(
                    "Live-trade position opened in live mode."
                    if live_mode
                    else (
                        "Shadow live-trade position opened with paper execution."
                        if runtime_mode == "shadow"
                        else "Paper live-trade position opened."
                    )
                ),
                quantity=position.quantity,
                payload={
                    "position_id": position.id,
                    "execution_mode": execution_mode_label,
                    "entry_fee": position.entry_fee,
                    "contracts_cost": position.contracts_cost,
                    "stop_loss_price": position.stop_loss_price,
                    "take_profit_price": position.take_profit_price,
                    "gate_snapshot": gate_snapshot,
                },
            )
            return True

        await self.db_manager.update_position_status(position_id, "voided")
        await self._record_execution_status(
            run_id=run_id,
            final_intent=final_intent,
            status="error",
            summary=(
                "Live execution did not fill the selected live-trade intent."
                if live_mode
                else (
                    "Shadow paper execution did not fill the selected live-trade intent."
                    if runtime_mode == "shadow"
                    else "Paper execution did not fill the selected live-trade intent."
                )
            ),
            error="execution_failed",
            quantity=quantity,
        )
        return False

    async def _record_execution_status(
        self,
        *,
        run_id: str,
        final_intent: Dict[str, Any],
        status: str,
        summary: str,
        error: Optional[str] = None,
        quantity: Optional[float] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self.db_manager.add_live_trade_decision(
            self._new_decision_record(
                run_id=run_id,
                step="execution",
                event_ticker=final_intent.get("event_ticker"),
                market_ticker=final_intent.get("market_ticker"),
                status=status,
                action=final_intent.get("action"),
                side=final_intent.get("side"),
                confidence=_safe_float(final_intent.get("confidence"), 0.0),
                edge_pct=_safe_float(final_intent.get("edge_pct"), 0.0),
                limit_price=_safe_float(final_intent.get("limit_price"), 0.0),
                quantity=quantity,
                hold_minutes=_safe_int(final_intent.get("hold_minutes"), 0),
                summary=summary,
                rationale=str(final_intent.get("reasoning", "") or ""),
                payload_json=json.dumps(payload or final_intent, default=str),
                error=error,
            )
        )
        await self._persist_runtime_state(
            run_id=run_id,
            loop_status="completed",
            step="execution",
            step_status=status,
            summary=summary,
            error=error,
            completed=True,
            healthy=status != "error",
            execution_status=status,
        )


async def run_live_trade_loop_cycle(
    *,
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
) -> LiveTradeLoopSummary:
    """Execute one live-trade cycle inside the existing runtime."""
    loop = LiveTradeDecisionLoop(
        db_manager=db_manager,
        kalshi_client=kalshi_client,
    )
    try:
        return await loop.run_once()
    finally:
        await loop.close()
