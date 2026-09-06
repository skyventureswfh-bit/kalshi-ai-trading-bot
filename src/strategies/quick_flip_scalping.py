"""
Quick Flip Scalping Strategy

Buy low-priced contracts, immediately rest an exit order, and cut stale
positions with docs-compatible limit orders only.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple
import uuid

from json_repair import repair_json

from src.clients.kalshi_client import KalshiClient
from src.config.settings import settings
from src.jobs.execute import (
    _record_live_exit_fee_divergence_if_filled,
    execute_position,
    place_sell_limit_order,
    reconcile_simulated_exit_orders,
    record_live_fee_divergence_if_needed,
    record_simulated_position_exit,
    submit_simulated_sell_limit_order,
)
from src.utils.database import DatabaseManager, Market, Position, TradeLog
from src.utils.kalshi_normalization import (
    get_balance_dollars,
    build_limit_order_price_fields,
    find_fill_price_for_order,
    get_fill_count,
    get_best_ask_price,
    get_best_bid_price,
    get_fill_price_dollars,
    get_last_price,
    get_market_result,
    get_market_expiration_ts,
    get_market_status,
    get_market_tick_size,
    get_market_volume,
    get_portfolio_value_dollars,
    get_order_average_fill_price,
    get_order_fill_count,
    get_position_exposure_dollars,
    get_position_size,
    get_position_ticker,
    is_active_market_status,
)
from src.utils.logging_setup import get_trading_logger
from src.utils.trade_pricing import estimate_kalshi_fee
from src.utils.trade_pricing import (
    FeeMetadata,
    calculate_entry_cost,
    extract_fee_metadata,
    sum_fill_fees,
)

@dataclass
class QuickFlipOpportunity:
    """Represents a quick flip scalping opportunity."""

    market_id: str
    market_title: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    expected_profit: float
    confidence_score: float
    movement_indicator: str
    max_hold_time: int
    tick_size: float = 0.01


@dataclass
class QuickFlipConfig:
    """Configuration for quick flip strategy."""

    min_entry_price: float = 0.01
    max_entry_price: float = 0.20
    min_profit_margin: float = 0.10
    max_position_size: int = 100
    max_concurrent_positions: int = 50
    capital_per_trade: float = 50.0
    confidence_threshold: float = 0.6
    max_hold_minutes: int = 30
    max_market_checks: int = 100
    target_opportunity_buffer: int = 6
    min_market_volume: int = 1000
    max_hours_to_expiry: int = 72
    max_bid_ask_spread: float = 0.03
    min_orderbook_depth_contracts: int = 10
    min_net_profit_per_trade: float = 0.10
    min_net_roi: float = 0.03
    recent_trade_window_seconds: int = 3600
    min_recent_trade_count: int = 5
    max_target_vs_recent_trade_gap: float = 0.02
    min_recent_range_ticks: int = 2
    min_recent_price_position: float = 0.4
    max_entry_vs_recent_last_gap: float = 0.02
    maker_entry_timeout_seconds: int = 180
    maker_entry_poll_seconds: int = 5
    maker_entry_reprice_seconds: int = 30
    dynamic_exit_reprice_seconds: int = 60
    stop_loss_pct: float = 0.08
    ev_gate_enabled: bool = True
    ev_confidence_margin: float = 0.0
    max_last_trade_age_seconds: int = 900
    min_bid_ask_size_ratio: float = 0.5


class QuickFlipScalpingStrategy:
    """Implements a fast-turnover scalp strategy."""

    _NEGATIVE_REASON_PHRASES: Tuple[str, ...] = (
        "not a scalping opportunity",
        "not a scalp",
        "do not scalp",
        "recommendation: do not scalp",
        "no immediate catalyst",
        "no immediate catalysts",
        "no credible near-term catalyst",
        "low probability of 30-minute movement",
        "extremely low probability of 30-minute movement",
        "zero expected volatility",
        "extremely low volatility",
        "price likely remains flat",
        "hold and wait for news",
        "long-term market",
        "not news-reactive",
    )

    def __init__(
        self,
        db_manager: DatabaseManager,
        kalshi_client: KalshiClient,
        xai_client: Any,
        config: Optional[QuickFlipConfig] = None,
        *,
        disable_ai: Optional[bool] = None,
    ) -> None:
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.xai_client = xai_client
        self.config = config or QuickFlipConfig()
        self.logger = get_trading_logger("quick_flip_scalping")
        self.active_positions: Dict[str, Position] = {}
        self.pending_sells: Dict[str, dict] = {}
        self.disable_ai = (
            bool(disable_ai)
            if disable_ai is not None
            else bool(getattr(settings.trading, "quick_flip_disable_ai", False))
        )
        self._portfolio_enforcer: Optional[Any] = None

    def _snapshot_prices_in_entry_band(self, market: Market) -> Tuple[float, ...]:
        buffer = max(0.01, self.config.max_bid_ask_spread)
        lower_bound = max(0.0, self.config.min_entry_price - buffer)
        upper_bound = min(1.0, self.config.max_entry_price + buffer)
        return tuple(
            price
            for price in (market.yes_price, market.no_price)
            if price > 0 and lower_bound <= price <= upper_bound
        )

    def _snapshot_candidate_matches(self, market: Market) -> bool:
        return bool(self._snapshot_prices_in_entry_band(market))

    async def _resolve_portfolio_value(self, *, fallback_trade_value: float) -> float:
        if not hasattr(self.kalshi_client, "get_balance"):
            return max(float(fallback_trade_value), 0.0)
        try:
            balance_response = await self.kalshi_client.get_balance()
        except Exception as exc:
            self.logger.warning(
                f"Could not fetch quick flip balance for portfolio guardrails: {exc}"
            )
            return max(float(fallback_trade_value), 0.0)
        available_cash = get_balance_dollars(balance_response)
        marked_position_value = get_portfolio_value_dollars(balance_response)
        total_portfolio_value = available_cash + marked_position_value
        if total_portfolio_value > 0:
            return total_portfolio_value
        return max(available_cash, float(fallback_trade_value), 0.0)

    async def _get_portfolio_enforcer(self, *, portfolio_value: float) -> Optional[Any]:
        db_path = getattr(self.db_manager, "db_path", None)
        if not db_path:
            return None
        if self._portfolio_enforcer is None:
            from src.strategies.portfolio_enforcer import PortfolioEnforcer
            self._portfolio_enforcer = PortfolioEnforcer(
                db_path=db_path,
                portfolio_value=portfolio_value,
            )
            await self._portfolio_enforcer.initialize()
        self._portfolio_enforcer.portfolio_value = portfolio_value
        return self._portfolio_enforcer

    async def _passes_portfolio_enforcer(
        self,
        opportunity: QuickFlipOpportunity,
        *,
        live_mode: Optional[bool] = None,
        shadow_mode: Optional[bool] = None,
        current_positions: Optional[Dict[str, float]] = None,
    ) -> bool:
        from src.strategies.portfolio_enforcer import (
            MODE_LIVE,
            MODE_PAPER,
            MODE_SHADOW,
            STRATEGY_QUICK_FLIP,
        )
        trade_value = float(opportunity.quantity) * float(opportunity.entry_price)
        portfolio_value = await self._resolve_portfolio_value(fallback_trade_value=trade_value)
        enforcer = await self._get_portfolio_enforcer(portfolio_value=portfolio_value)
        if enforcer is None:
            return True
        if live_mode is None:
            live_mode = getattr(settings.trading, "live_trading_enabled", False)
        if shadow_mode is None:
            shadow_mode = getattr(settings.trading, "shadow_mode_enabled", False)
        if current_positions is None:
            current_positions = await self._get_current_position_exposures()
        mode = MODE_LIVE if live_mode else MODE_SHADOW if shadow_mode else MODE_PAPER
        allowed, reason = await enforcer.check_trade(
            ticker=opportunity.market_id,
            side=opportunity.side.lower(),
            amount=trade_value,
            title=opportunity.market_title,
            strategy=STRATEGY_QUICK_FLIP,
            mode=mode,
            current_positions=current_positions,
        )
        if not allowed:
            self.logger.warning(
                f"Portfolio enforcer blocked quick flip entry for {opportunity.market_id}: {reason}"
            )
        return allowed

    @staticmethod
    def _estimate_position_cost_basis(position: Any) -> float:
        def _field(name: str, default: float = 0.0):
            if isinstance(position, dict):
                return position.get(name, default)
            return getattr(position, name, default)
        contracts_cost = float(_field("contracts_cost", 0.0) or 0.0)
        if contracts_cost > 0:
            return contracts_cost
        quantity = float(_field("quantity", 0.0) or 0.0)
        entry_price = float(_field("entry_price", 0.0) or 0.0)
        entry_fee = max(float(_field("entry_fee", 0.0) or 0.0), 0.0)
        return max((quantity * entry_price) + entry_fee, 0.0)

    async def _get_current_position_exposures(self) -> Dict[str, float]:
        getter = getattr(self.db_manager, "get_open_positions", None)
        if not callable(getter):
            return {}
        try:
            positions_result = getter()
            if asyncio.iscoroutine(positions_result):
                positions_result = await positions_result
            exposures: Dict[str, float] = {}
            for position in positions_result or []:
                market_id = str(position.get("market_id", "") if isinstance(position, dict) else getattr(position, "market_id", "") or "")
                if market_id:
                    exposures[market_id] = exposures.get(market_id, 0.0) + self._estimate_position_cost_basis(position)
            return exposures
        except Exception as exc:
            self.logger.warning(f"Could not calculate current quick-flip position exposures: {exc}")
            return {}

    @staticmethod
    def _estimate_kalshi_fee(price: float, quantity: float, *, maker: bool) -> float:
        return estimate_kalshi_fee(price, quantity, maker=maker)

    async def _resolve_fee_metadata(self, market_info: Dict[str, Any]) -> FeeMetadata:
        metadata = extract_fee_metadata(market_info)
        if metadata.fee_type:
            return metadata
        series_ticker = str(market_info.get("series_ticker") or "").strip()
        if not series_ticker:
            return metadata
        try:
            series_response = await self.kalshi_client.get_series(series_ticker)
            series_info = series_response.get("series", {}) if isinstance(series_response, dict) else {}
            return extract_fee_metadata(market_info, series_info)
        except Exception as exc:
            self.logger.debug(f"Could not resolve fee metadata for quick flip series {series_ticker}: {exc}")
            return metadata

    @staticmethod
    def _normalize_orderbook_levels(orderbook: Dict[str, Any], side: str) -> List[Tuple[float, float]]:
        raw_levels = orderbook.get(f"{side.lower()}_dollars", orderbook.get(side.lower(), []))
        levels: List[Tuple[float, float]] = []
        for level in raw_levels:
            if not isinstance(level, (list, tuple)) or len(level) < 2:
                continue
            try:
                price = float(level[0]); size = float(level[1])
            except (TypeError, ValueError):
                continue
            if price > 1.0:
                price /= 100.0
            if price > 0 and size > 0:
                levels.append((price, size))
        return sorted(levels, key=lambda item: item[0], reverse=True)

    @classmethod
    def _get_best_bid_size(cls, orderbook: Dict[str, Any], side: str) -> float:
        levels = cls._normalize_orderbook_levels(orderbook, side)
        if not levels:
            return 0.0
        best_price = levels[0][0]
        return sum(size for price, size in levels if abs(price - best_price) < 1e-9)

    @staticmethod
    def _round_up_to_tick(price: float, tick_size: float) -> float:
        if tick_size <= 0:
            tick_size = 0.01
        rounded = math.ceil(max(price, tick_size) / tick_size) * tick_size
        return round(min(0.95, rounded), 4)

    def _estimate_trade_profit(self, *, entry_price: float, exit_price: float, quantity: float) -> Dict[str, float]:
        entry_fee = self._estimate_kalshi_fee(entry_price, quantity, maker=True)
        exit_fee = self._estimate_kalshi_fee(exit_price, quantity, maker=True)
        gross_profit = (exit_price - entry_price) * quantity
        fees_paid = entry_fee + exit_fee
        net_profit = gross_profit - fees_paid
        deployed_capital = (entry_price * quantity) + entry_fee
        net_roi = net_profit / deployed_capital if deployed_capital > 0 else 0.0
        return {"gross_profit": gross_profit, "fees_paid": fees_paid, "net_profit": net_profit, "net_roi": net_roi, "entry_fee": entry_fee, "exit_fee": exit_fee}

    def _required_win_probability(self, *, entry_price: float, quantity: float, net_profit_at_target: float, tick_size: float = 0.01) -> Optional[float]:
        stop_price = self._calculate_stop_loss_price(entry_price=entry_price, tick_size=tick_size)
        entry_fee = self._estimate_kalshi_fee(entry_price, quantity, maker=True)
        stop_exit_fee = self._estimate_kalshi_fee(stop_price, quantity, maker=False)
        risk = (entry_price - stop_price) * quantity + entry_fee + stop_exit_fee
        reward = float(net_profit_at_target)
        if reward <= 0:
            return None
        if risk <= 0:
            return 0.0
        return risk / (risk + reward)

    @staticmethod
    def _normalize_fill_price(price: float) -> float:
        return price / 100.0 if price > 1.0 else price

    def _round_up_to_valid_tick(self, *, price: float, tick_size: float, market_info: Optional[Dict[str, Any]] = None) -> float:
        effective_tick = get_market_tick_size(market_info, price) if market_info else tick_size
        return self._round_up_to_tick(price, effective_tick)

    def _minimum_profitable_exit_price(self, *, entry_price: float, quantity: float, tick_size: float, market_info: Optional[Dict[str, Any]] = None) -> float:
        candidate_price = self._round_up_to_valid_tick(price=entry_price + tick_size, tick_size=tick_size, market_info=market_info)
        while candidate_price <= 0.95:
            profit_estimate = self._estimate_trade_profit(entry_price=entry_price, exit_price=candidate_price, quantity=quantity)
            if profit_estimate["net_profit"] >= self.config.min_net_profit_per_trade and profit_estimate["net_roi"] >= self.config.min_net_roi:
                return candidate_price
            next_tick = get_market_tick_size(market_info, candidate_price) if market_info else tick_size
            candidate_price = self._round_up_to_valid_tick(price=candidate_price + next_tick, tick_size=next_tick, market_info=market_info)
        return 1.0

    def _calculate_maker_entry_price(self, *, best_bid: float, best_ask: float, tick_size: float) -> Optional[float]:
        if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
            return None
        maker_price = round(best_ask - tick_size, 4)
        if maker_price <= best_bid or maker_price >= best_ask:
            return None
        return round(max(tick_size, maker_price), 4)

    @classmethod
    def _reason_blocks_trade(cls, reason: str) -> bool:
        normalized_reason = str(reason or "").strip().lower()
        return bool(normalized_reason) and any(phrase in normalized_reason for phrase in cls._NEGATIVE_REASON_PHRASES)

    def _calculate_stop_loss_price(self, *, entry_price: float, tick_size: float) -> float:
        stop_price = entry_price * (1.0 - self.config.stop_loss_pct)
        stop_price = math.floor(stop_price / tick_size) * tick_size
        return round(max(tick_size, stop_price), 4)

    @staticmethod
    def _get_fill_timestamp(fill: Dict[str, Any]) -> Optional[float]:
        ts_value = fill.get("ts")
        if ts_value not in (None, ""):
            try:
                numeric = float(ts_value)
                if numeric > 1e12:
                    numeric /= 1000.0
                return numeric
            except (TypeError, ValueError):
                pass
        created_time = fill.get("created_time")
        if isinstance(created_time, str):
            try:
                return datetime.fromisoformat(created_time.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return None
        return None

    @staticmethod
    def _fill_identity(fill: Dict[str, Any]) -> Tuple[Any, ...]:
        """Stable best-effort identity used to dedupe live/history fill overlap."""
        explicit_id = fill.get("fill_id") or fill.get("trade_id")
        if explicit_id not in (None, ""):
            return ("id", str(explicit_id))
        return (
            "fallback",
            str(fill.get("order_id") or ""),
            str(fill.get("client_order_id") or ""),
            str(fill.get("ticker") or ""),
            str(fill.get("side") or "").lower(),
            str(fill.get("action") or "").lower(),
            str(fill.get("count") or fill.get("quantity") or ""),
            str(fill.get("yes_price_dollars") or fill.get("yes_price") or ""),
            str(fill.get("no_price_dollars") or fill.get("no_price") or ""),
            str(fill.get("ts") or fill.get("created_time") or ""),
        )

    def _dedupe_position_fills(self, fills: List[Dict[str, Any]], position: Position) -> List[Dict[str, Any]]:
        """Keep only unique fills that can belong to this ticker and contract side."""
        target_side = str(position.side or "").lower()
        target_ticker = str(position.market_id or "")
        seen = set()
        unique: List[Dict[str, Any]] = []
        for fill in fills:
            ticker = str(fill.get("ticker") or "")
            if ticker and ticker != target_ticker:
                continue
            side = str(fill.get("side") or "").lower()
            if side and target_side and side != target_side:
                continue
            key = self._fill_identity(fill)
            if key in seen:
                continue
            seen.add(key)
            unique.append(fill)
        return unique

    async def _find_order_snapshot(self, *, ticker: str, client_order_id: Optional[str] = None, order_id: Optional[str] = None) -> Dict[str, Any]:
        orders_response = await self.kalshi_client.get_orders(ticker=ticker, limit=100)
        for order in orders_response.get("orders", []):
            if order_id and order.get("order_id") == order_id:
                return order
            if client_order_id and order.get("client_order_id") == client_order_id:
                return order
        return {}

    async def _wait_for_entry_fill(self, *, ticker: str, side: str, client_order_id: str, order_id: Optional[str], submitted_price: float, timeout_seconds: int) -> Dict[str, float]:
        deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)
        while datetime.now(timezone.utc) < deadline:
            order_info = await self._find_order_snapshot(ticker=ticker, client_order_id=client_order_id, order_id=order_id)
            filled_quantity = get_order_fill_count(order_info)
            order_status = str(order_info.get("status", "")).lower()
            if filled_quantity > 0:
                fill_price = get_order_average_fill_price(order_info, side=side)
                if not fill_price or fill_price <= 0:
                    fills_response = await self.kalshi_client.get_fills(ticker=ticker, limit=30)
                    fill_price = find_fill_price_for_order(fills_response.get("fills", []), side=side, order_id=order_info.get("order_id"), client_order_id=client_order_id, ticker=ticker)
                return {"filled_quantity": float(filled_quantity), "fill_price": float(fill_price or submitted_price), "status": order_status}
            if order_status in {"canceled", "cancelled", "expired", "rejected"}:
                return {"filled_quantity": 0.0, "fill_price": submitted_price, "status": order_status}
            await asyncio.sleep(self.config.maker_entry_poll_seconds)
        return {"filled_quantity": 0.0, "fill_price": submitted_price, "status": "timeout"}

    @staticmethod
    def _trade_timestamp_epoch(trade: Dict[str, Any]) -> Optional[float]:
        raw_ts = trade.get("ts")
        if raw_ts is not None:
            try:
                numeric = float(raw_ts)
                if numeric > 1e12:
                    numeric /= 1000.0
                if numeric > 0:
                    return numeric
            except (TypeError, ValueError):
                pass
        created = trade.get("created_time") or trade.get("created_ts")
        if isinstance(created, str) and created.strip():
            try:
                return datetime.fromisoformat(created.strip().replace("Z", "+00:00")).timestamp()
            except ValueError:
                return None
        return None

    async def _get_recent_trade_stats(self, market_id: str, side: str) -> Dict[str, float]:
        now_ts = int(datetime.now(timezone.utc).timestamp())
        min_ts = now_ts - self.config.recent_trade_window_seconds
        trades_response = await self.kalshi_client.get_market_trades(market_id, limit=100, min_ts=min_ts)
        trades = trades_response.get("trades", [])
        prices: List[float] = []
        for trade in trades:
            price_key = "yes_price_dollars" if side.upper() == "YES" else "no_price_dollars"
            try:
                price = float(trade.get(price_key, 0) or 0)
            except (TypeError, ValueError):
                continue
            if price > 1.0:
                price /= 100.0
            if price > 0:
                prices.append(price)
        if not prices:
            return {"trade_count": 0.0, "recent_max_price": 0.0, "recent_min_price": 0.0, "recent_last_price": 0.0}
        stats: Dict[str, float] = {
            "trade_count": float(len(prices)),
            "recent_max_price": max(prices),
            "recent_min_price": min(prices),
            "recent_last_price": prices[0],
            "recent_range": max(prices) - min(prices),
        }
        if trades:
            newest_ts = self._trade_timestamp_epoch(trades[0])
            if newest_ts is not None:
                stats["last_trade_age_seconds"] = max(0.0, now_ts - newest_ts)
        return stats

    async def _calculate_dynamic_exit_price(self, position: Position, market_info: Dict[str, Any]) -> Optional[float]:
        tick_size = get_market_tick_size(market_info, position.entry_price)
        best_bid = get_best_bid_price(market_info, position.side)
        best_ask = get_best_ask_price(market_info, position.side)
        if best_bid <= 0 or best_ask <= 0:
            return None
        profit_floor = self._minimum_profitable_exit_price(entry_price=position.entry_price, quantity=position.quantity, tick_size=tick_size, market_info=market_info)
        if profit_floor > 0.95:
            return None
        recent_trade_stats = await self._get_recent_trade_stats(position.market_id, position.side)
        recent_last_price = recent_trade_stats["recent_last_price"]
        recent_max_price = recent_trade_stats["recent_max_price"]
        held_seconds = max(0.0, (datetime.now() - position.timestamp).total_seconds())
        max_hold_seconds = max(60.0, self.config.max_hold_minutes * 60.0)
        hold_progress = min(1.0, held_seconds / max_hold_seconds)
        competitive_exit = max(profit_floor, self._round_up_to_valid_tick(price=best_bid + tick_size, tick_size=tick_size, market_info=market_info))
        recent_trade_target = max(competitive_exit, self._round_up_to_valid_tick(price=max(recent_last_price, recent_max_price) + tick_size, tick_size=tick_size, market_info=market_info))
        target = min(best_ask, recent_trade_target) if hold_progress < 0.5 else min(best_ask, competitive_exit)
        target = self._round_up_to_valid_tick(price=max(profit_floor, target), tick_size=tick_size, market_info=market_info)
        if target <= best_bid:
            target = self._round_up_to_valid_tick(price=best_bid + tick_size, tick_size=tick_size, market_info=market_info)
        return round(min(0.95, target), 4)

    async def _place_live_dynamic_exit_order(self, position: Position) -> Dict[str, float | bool]:
        market_response = await self.kalshi_client.get_market(position.market_id)
        market_info = market_response.get("market", {})
        if not market_info:
            return {"success": False}
        sell_price = await self._calculate_dynamic_exit_price(position, market_info)
        if sell_price is None:
            return {"success": False}
        success = await place_sell_limit_order(position=position, limit_price=sell_price, db_manager=self.db_manager, kalshi_client=self.kalshi_client, live_mode=True)
        if not success:
            return {"success": False}
        self.pending_sells[position.market_id] = {"position": position, "target_price": sell_price, "placed_at": datetime.now(), "max_hold_until": datetime.now() + timedelta(minutes=self.config.max_hold_minutes)}
        return {"success": True, "filled": False, "gross_pnl": 0.0, "net_pnl": 0.0, "fees_paid": 0.0, "is_win": False, "is_loss": False, "target_price": sell_price}

    async def _execute_live_maker_entry(self, position: Position) -> bool:
        remaining_timeout = self.config.maker_entry_timeout_seconds
        while remaining_timeout > 0:
            market_response = await self.kalshi_client.get_market(position.market_id)
            market_info = market_response.get("market", {})
            if not market_info:
                return False
            best_bid = get_best_bid_price(market_info, position.side)
            best_ask = get_best_ask_price(market_info, position.side)
            tick_size = get_market_tick_size(market_info, best_ask or best_bid)
            maker_price = self._calculate_maker_entry_price(best_bid=best_bid, best_ask=best_ask, tick_size=tick_size)
            if maker_price is None or maker_price > float(position.entry_price) + 1e-9:
                return False
            attempt_timeout = min(remaining_timeout, self.config.maker_entry_reprice_seconds)
            order_params = {
                "ticker": position.market_id,
                "client_order_id": str(uuid.uuid4()),
                "side": position.side.lower(),
                "action": "buy",
                "count": position.quantity,
                "type_": "limit",
                "time_in_force": "good_till_canceled",
                "post_only": True,
                "expiration_ts": int(datetime.now(timezone.utc).timestamp() + attempt_timeout),
                **build_limit_order_price_fields(position.side, maker_price),
            }
            order_response = await self.kalshi_client.place_order(**order_params)
            order_info = order_response.get("order", {}) if isinstance(order_response, dict) else {}
            order_id = order_info.get("order_id")
            fill_result = await self._wait_for_entry_fill(ticker=position.market_id, side=position.side, client_order_id=order_params["client_order_id"], order_id=order_id, submitted_price=maker_price, timeout_seconds=attempt_timeout)
            filled_quantity = float(fill_result["filled_quantity"])
            if filled_quantity > 0:
                requested_quantity = position.quantity
                if order_id and filled_quantity < requested_quantity:
                    try:
                        await self.kalshi_client.cancel_order(order_id)
                    except Exception:
                        pass
                actual_fill_price = self._normalize_fill_price(float(fill_result["fill_price"]))
                position.quantity = filled_quantity
                position.entry_price = actual_fill_price
                position.live = True
                fee_metadata = await self._resolve_fee_metadata(market_info)
                entry_cost = calculate_entry_cost(actual_fill_price, filled_quantity, maker=True, fee_type=fee_metadata.fee_type, fee_multiplier=fee_metadata.fee_multiplier, fee_waiver_expiration_time=fee_metadata.fee_waiver_expiration_time, trade_ts=datetime.now(timezone.utc))
                position.entry_fee = entry_cost["fee"]
                position.contracts_cost = entry_cost["contracts_cost"]
                position.entry_order_id = order_id or order_params["client_order_id"]
                position.stop_loss_price = self._calculate_stop_loss_price(entry_price=actual_fill_price, tick_size=tick_size)
                position.take_profit_price = await self._calculate_dynamic_exit_price(position, market_info)
                position.max_hold_hours = max(1, math.ceil(self.config.max_hold_minutes / 60))
                await self.db_manager.update_position_execution_details(position.id, entry_price=position.entry_price, quantity=position.quantity, live=True, stop_loss_price=position.stop_loss_price, take_profit_price=position.take_profit_price, max_hold_hours=position.max_hold_hours, entry_fee=position.entry_fee, contracts_cost=position.contracts_cost, entry_order_id=position.entry_order_id)
                return True
            if order_id:
                try:
                    await self.kalshi_client.cancel_order(order_id)
                except Exception:
                    pass
            remaining_timeout -= attempt_timeout
        return False

    async def _close_position_from_recent_fills(self, position: Position, *, market_info: Optional[Dict[str, Any]] = None) -> bool:
        """Close local quick-flip accounting only from unique, same-side fills."""
        fills: List[Dict[str, Any]] = []
        live_fills_response = await self.kalshi_client.get_fills(ticker=position.market_id, limit=50)
        fills.extend(live_fills_response.get("fills", []))
        try:
            historical_fills_response = await self.kalshi_client.get_historical_fills(ticker=position.market_id, limit=50)
            fills.extend(historical_fills_response.get("fills", []))
        except Exception as exc:
            self.logger.debug(f"Historical fills unavailable for {position.market_id}: {exc}")

        fills = self._dedupe_position_fills(fills, position)
        entry_ts = position.timestamp
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.replace(tzinfo=timezone.utc)
        entry_epoch = entry_ts.timestamp() - 1.0
        relevant_fills = [fill for fill in fills if (self._get_fill_timestamp(fill) or 0.0) >= entry_epoch]
        sell_fills = [fill for fill in relevant_fills if str(fill.get("action", "")).lower() == "sell"]
        buy_candidates = [fill for fill in relevant_fills if str(fill.get("action", "")).lower() == "buy"]

        if not sell_fills:
            return False
        sell_quantity = sum(get_fill_count(fill) for fill in sell_fills)
        if sell_quantity <= 0:
            return False
        closed_quantity = min(position.quantity, sell_quantity)
        if closed_quantity <= 0:
            return False

        entry_order_id = str(position.entry_order_id or "")
        if entry_order_id:
            buy_fills = [
                fill for fill in buy_candidates
                if entry_order_id in {
                    str(fill.get("order_id") or ""),
                    str(fill.get("client_order_id") or ""),
                }
            ]
        else:
            buy_fills = buy_candidates

        def _safe_float(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        buy_fill_quantity = 0.0
        buy_fill_value = 0.0
        for fill in buy_fills:
            count = get_fill_count(fill)
            if count > 0:
                buy_fill_quantity += count
                buy_fill_value += get_fill_price_dollars(fill, side=position.side) * count

        sell_fill_quantity = 0.0
        sell_fill_value = 0.0
        for fill in sell_fills:
            count = get_fill_count(fill)
            if count > 0:
                sell_fill_quantity += count
                sell_fill_value += get_fill_price_dollars(fill, side=position.side) * count

        if sell_fill_quantity <= 0:
            return False
        exit_price = sell_fill_value / sell_fill_quantity
        entry_price = buy_fill_value / buy_fill_quantity if buy_fill_quantity > 0 else position.entry_price

        total_entry_fee = position.entry_fee if position.entry_order_id or position.contracts_cost > 0 or position.entry_fee > 0 else self._estimate_kalshi_fee(entry_price, closed_quantity, maker=True)
        if buy_fills and buy_fill_quantity > 0:
            try:
                buy_fees = [_safe_float(fill.get("fee_cost", 0) or 0) for fill in buy_fills]
                if any(fee > 0 for fee in buy_fees):
                    total_entry_fee = sum(buy_fees) * (closed_quantity / buy_fill_quantity)
            except (TypeError, ValueError):
                pass
        try:
            actual_exit_fee = sum_fill_fees(sell_fills)
            if actual_exit_fee is not None and sell_fill_quantity > 0:
                actual_exit_fee *= closed_quantity / sell_fill_quantity
        except (TypeError, ValueError):
            actual_exit_fee = None

        effective_market_info = market_info or {}
        if not effective_market_info and actual_exit_fee is not None:
            try:
                market_response = await self.kalshi_client.get_market(position.market_id)
                effective_market_info = market_response.get("market", {}) if isinstance(market_response, dict) else {}
            except Exception:
                effective_market_info = {}
        fee_metadata = await self._resolve_fee_metadata(effective_market_info) if effective_market_info else FeeMetadata()
        estimated_exit_fee = estimate_kalshi_fee(exit_price, closed_quantity, maker=True, fee_type=fee_metadata.fee_type, fee_multiplier=fee_metadata.fee_multiplier, fee_waiver_expiration_time=fee_metadata.fee_waiver_expiration_time, trade_ts=datetime.now())
        exit_fee = actual_exit_fee if actual_exit_fee is not None else estimated_exit_fee
        net_pnl = ((exit_price - entry_price) * closed_quantity) - total_entry_fee - exit_fee
        trade_log = TradeLog(
            market_id=position.market_id,
            side=position.side,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=closed_quantity,
            pnl=net_pnl,
            entry_timestamp=position.timestamp,
            exit_timestamp=datetime.now(),
            rationale=f"{position.rationale} | LIVE QUICK FLIP EXIT FILLED",
            entry_fee=total_entry_fee,
            exit_fee=exit_fee,
            fees_paid=total_entry_fee + exit_fee,
            live=True,
            contracts_cost=position.contracts_cost if position.contracts_cost > 0 else entry_price * closed_quantity,
            strategy=position.strategy,
        )
        await self.db_manager.add_trade_log(trade_log)
        sell_order_ids = {str(fill.get("order_id")) for fill in sell_fills if fill.get("order_id") not in (None, "")}
        await record_live_fee_divergence_if_needed(db_manager=self.db_manager, market_id=position.market_id, side=position.side, leg="exit", estimated_fee=estimated_exit_fee, actual_fee=actual_exit_fee, position_id=position.id, order_id=next(iter(sell_order_ids)) if len(sell_order_ids) == 1 else None, quantity=closed_quantity, price=exit_price)
        await self.db_manager.update_position_status(position.id, "closed")
        self.active_positions.pop(position.market_id, None)
        self.pending_sells.pop(position.market_id, None)
        return True

    async def _cancel_resting_exit_orders(self, position: Position) -> int:
        cancelled = 0
        try:
            orders_response = await self.kalshi_client.get_orders(ticker=position.market_id, status="resting", limit=100)
            for order in orders_response.get("orders", []):
                if str(order.get("action", "")).lower() == "sell" and str(order.get("side", "")).upper() == position.side.upper():
                    order_id = order.get("order_id")
                    if order_id:
                        await self.kalshi_client.cancel_order(order_id)
                        cancelled += 1
        except Exception as exc:
            self.logger.error(f"Error cancelling stale exit orders for {position.market_id}: {exc}")
        return cancelled

    async def reconcile_persisted_live_positions(self) -> Dict[str, int]:
        results = {"positions_examined": 0, "positions_closed": 0, "positions_voided": 0, "positions_synced": 0, "orders_cancelled": 0, "side_mismatches": 0, "errors": 0}
        if not getattr(settings.trading, "live_trading_enabled", False):
            return results
        all_positions = await self.db_manager.get_open_live_positions()
        positions = [position for position in all_positions if position.strategy == "quick_flip_scalping"]
        for position in positions:
            results["positions_examined"] += 1
            try:
                positions_response = await self.kalshi_client.get_positions(ticker=position.market_id)
                market_positions = [item for item in positions_response.get("market_positions", []) if get_position_ticker(item) in ("", position.market_id)]
                signed_size = sum(get_position_size(item) for item in market_positions)
                exposure = sum(get_position_exposure_dollars(item) for item in market_positions if get_position_exposure_dollars(item) > 0)
                if exposure <= 0 and abs(signed_size) <= 1e-9:
                    results["orders_cancelled"] += await self._cancel_resting_exit_orders(position)
                    closed = await self._close_position_from_recent_fills(position)
                    if closed:
                        results["positions_closed"] += 1
                    else:
                        await self.db_manager.update_position_status(position.id, "voided", rationale_suffix="RECONCILIATION: no Kalshi exposure, no resting exit order, and no matching fill history found")
                        self.active_positions.pop(position.market_id, None)
                        self.pending_sells.pop(position.market_id, None)
                        results["positions_voided"] += 1
                    continue
                actual_qty = abs(signed_size)
                if actual_qty <= 1e-9:
                    continue
                actual_side = "YES" if signed_size > 0 else "NO"
                if actual_side != position.side:
                    results["side_mismatches"] += 1
                    continue
                if abs(actual_qty - position.quantity) > 1e-9:
                    await self.db_manager.update_position_execution_details(position.id, entry_price=position.entry_price, quantity=actual_qty, live=True, stop_loss_price=position.stop_loss_price, take_profit_price=position.take_profit_price, max_hold_hours=position.max_hold_hours, entry_fee=position.entry_fee, contracts_cost=position.contracts_cost, entry_order_id=position.entry_order_id)
                    position.quantity = actual_qty
                    results["positions_synced"] += 1
            except Exception as exc:
                results["errors"] += 1
                self.logger.error(f"Error reconciling quick-flip position {position.market_id}: {exc}")
        return results

    async def identify_quick_flip_opportunities(self, markets: List[Market], available_capital: float) -> List[QuickFlipOpportunity]:
        opportunities: List[QuickFlipOpportunity] = []
        max_positions = min(self.config.max_concurrent_positions, int(available_capital / max(self.config.capital_per_trade, 1)))
        if max_positions <= 0:
            return []
        target_opportunities = max_positions * max(1, self.config.target_opportunity_buffer)
        checked_markets = 0
        for market in markets:
            if checked_markets >= self.config.max_market_checks or len(opportunities) >= target_opportunities:
                break
            try:
                market_response = await self.kalshi_client.get_market(market.market_id)
                market_info = market_response.get("market", {})
                if not market_info or not is_active_market_status(get_market_status(market_info)):
                    continue
                market_volume = get_market_volume(market_info) or market.volume
                if market_volume < self.config.min_market_volume:
                    continue
                expiration_ts = get_market_expiration_ts(market_info) or market.expiration_ts
                if expiration_ts is None:
                    continue
                hours_to_expiry = max(0.0, (expiration_ts - datetime.now(timezone.utc).timestamp()) / 3600.0)
                if hours_to_expiry > self.config.max_hours_to_expiry:
                    continue
                orderbook_response = await self.kalshi_client.get_orderbook(market.market_id, depth=10)
                orderbook = orderbook_response.get("orderbook_fp", orderbook_response.get("orderbook", {}))
                if not orderbook:
                    continue
                checked_markets += 1
                for side in ("YES", "NO"):
                    opportunity = await self._evaluate_price_opportunity(market, market_info, orderbook, side, hours_to_expiry=hours_to_expiry, market_volume=market_volume)
                    if opportunity:
                        opportunities.append(opportunity)
            except Exception as exc:
                self.logger.error(f"Error analyzing market {market.market_id}: {exc}")
        opportunities.sort(key=lambda opp: opp.expected_profit * opp.confidence_score, reverse=True)
        return opportunities[:max_positions]

    async def _evaluate_price_opportunity(self, market: Market, market_info: Dict[str, Any], orderbook: Dict[str, Any], side: str, *, hours_to_expiry: float, market_volume: int) -> Optional[QuickFlipOpportunity]:
        current_ask = get_best_ask_price(market_info, side)
        current_bid = get_best_bid_price(market_info, side)
        if current_ask <= 0 or current_bid <= 0 or current_ask < self.config.min_entry_price or current_ask > self.config.max_entry_price:
            return None
        spread = current_ask - current_bid
        if spread <= 0 or spread > self.config.max_bid_ask_spread:
            return None
        best_bid_size = self._get_best_bid_size(orderbook, side)
        if best_bid_size < self.config.min_orderbook_depth_contracts:
            return None
        if self.config.min_bid_ask_size_ratio > 0:
            opposing_side = "no" if side.lower() == "yes" else "yes"
            opposing_size = self._get_best_bid_size(orderbook, opposing_side)
            if opposing_size > 0 and best_bid_size < opposing_size * self.config.min_bid_ask_size_ratio:
                return None
        quantity = min(self.config.max_position_size, int(self.config.capital_per_trade / current_ask), int(best_bid_size))
        if quantity < 1:
            return None
        tick_size = get_market_tick_size(market_info, current_ask)
        min_profit_margin_exit = self._round_up_to_valid_tick(price=current_ask * (1 + self.config.min_profit_margin), tick_size=tick_size, market_info=market_info)
        min_profitable_exit = max(min_profit_margin_exit, self._minimum_profitable_exit_price(entry_price=current_ask, quantity=quantity, tick_size=tick_size, market_info=market_info))
        if min_profitable_exit > 0.95:
            return None
        try:
            recent_trade_stats = await self._get_recent_trade_stats(market.market_id, side)
        except Exception:
            recent_trade_stats = {"trade_count": 0.0, "recent_max_price": 0.0, "recent_min_price": 0.0, "recent_last_price": 0.0, "recent_range": 0.0}
        max_trade_age = int(self.config.max_last_trade_age_seconds or 0)
        last_trade_age = recent_trade_stats.get("last_trade_age_seconds")
        if max_trade_age > 0 and last_trade_age is not None and float(last_trade_age) > max_trade_age:
            return None
        movement_analysis = await self._analyze_market_movement(market, side, current_ask, required_exit_price=min_profitable_exit, hours_to_expiry=hours_to_expiry, market_volume=market_volume, spread=spread, recent_trade_stats=recent_trade_stats)
        if movement_analysis["confidence"] < self.config.confidence_threshold or self._reason_blocks_trade(movement_analysis["reason"]):
            return None
        target_price = self._round_up_to_valid_tick(price=max(min_profitable_exit, movement_analysis["target_price"]), tick_size=tick_size, market_info=market_info)
        if target_price > 0.95:
            return None
        profit_estimate = self._estimate_trade_profit(entry_price=current_ask, exit_price=target_price, quantity=quantity)
        if profit_estimate["net_profit"] < self.config.min_net_profit_per_trade or profit_estimate["net_roi"] < self.config.min_net_roi:
            return None
        if self.config.ev_gate_enabled:
            required_win_probability = self._required_win_probability(entry_price=current_ask, quantity=quantity, net_profit_at_target=profit_estimate["net_profit"], tick_size=tick_size)
            if required_win_probability is not None:
                required_win_probability = min(0.99, required_win_probability + max(0.0, float(self.config.ev_confidence_margin)))
                if movement_analysis["confidence"] < required_win_probability:
                    return None
        if recent_trade_stats.get("trade_count", 0.0) < self.config.min_recent_trade_count:
            return None
        recent_last_price = float(recent_trade_stats.get("recent_last_price", 0.0))
        recent_max_price = float(recent_trade_stats.get("recent_max_price", 0.0))
        recent_min_price = float(recent_trade_stats.get("recent_min_price", 0.0))
        recent_range = recent_trade_stats.get("recent_range", recent_max_price - recent_min_price)
        required_move = target_price - current_ask
        min_recent_range = max(required_move, tick_size * self.config.min_recent_range_ticks)
        if recent_range + 1e-9 < min_recent_range:
            return None
        if recent_range > 0 and (recent_last_price - recent_min_price) / recent_range < self.config.min_recent_price_position:
            return None
        if current_ask - recent_last_price > self.config.max_entry_vs_recent_last_gap or target_price - recent_max_price > self.config.max_target_vs_recent_trade_gap:
            return None
        expected_profit = profit_estimate["net_profit"]
        reason = f"{movement_analysis['reason']} | bid=${current_bid:.4f} ask=${current_ask:.4f} spread=${spread:.4f} qty={quantity} vol={market_volume} hours_to_expiry={hours_to_expiry:.1f} recent_trades={int(recent_trade_stats.get('trade_count', 0.0))} recent_max=${recent_max_price:.4f} expected_net=${expected_profit:.2f}"
        return QuickFlipOpportunity(market_id=market.market_id, market_title=market.title, side=side, entry_price=current_ask, exit_price=target_price, quantity=quantity, expected_profit=expected_profit, confidence_score=movement_analysis["confidence"], movement_indicator=reason, max_hold_time=self.config.max_hold_minutes, tick_size=tick_size)

    async def _heuristic_movement_analysis(self, market: Market, side: str, current_price: float, *, required_exit_price: float, hours_to_expiry: float, spread: float, recent_trade_stats: Optional[Dict[str, float]] = None) -> dict:
        try:
            recent = recent_trade_stats if isinstance(recent_trade_stats, dict) else await self._get_recent_trade_stats(market.market_id, side)
        except Exception as exc:
            return {"target_price": current_price, "confidence": 0.0, "reason": f"Heuristic fallback failed to fetch trades: {exc}"}
        trade_count = float(recent.get("trade_count") or 0.0)
        recent_last = float(recent.get("recent_last_price") or 0.0)
        recent_max = float(recent.get("recent_max_price") or 0.0)
        recent_min = float(recent.get("recent_min_price") or 0.0)
        recent_range = float(recent.get("recent_range", max(0.0, recent_max - recent_min)) or 0.0)
        if trade_count < max(1, self.config.min_recent_trade_count) or recent_range <= 0:
            return {"target_price": current_price, "confidence": 0.0, "reason": "Heuristic: insufficient or flat recent tape"}
        price_position = (recent_last - recent_min) / recent_range
        if price_position < self.config.min_recent_price_position or current_price - recent_last > self.config.max_entry_vs_recent_last_gap:
            return {"target_price": current_price, "confidence": 0.0, "reason": "Heuristic: momentum not confirmed"}
        target_price = max(current_price, min(0.95, max(required_exit_price, recent_max)))
        tape_strength = min(1.0, trade_count / max(1.0, 3.0 * self.config.min_recent_trade_count))
        time_pressure = min(1.0, max(0.0, 1.0 - (hours_to_expiry / 24.0)))
        spread_health = 1.0 if spread <= self.config.max_bid_ask_spread else 0.0
        confidence = max(0.0, min(1.0, 0.45 * tape_strength + 0.35 * price_position + 0.15 * time_pressure + 0.05 * spread_health))
        if target_price + 1e-9 < required_exit_price:
            confidence = 0.0
        return {"target_price": target_price, "confidence": confidence, "reason": f"Heuristic momentum: trades={int(trade_count)} pos={price_position:.2f} recent=${recent_last:.4f} max=${recent_max:.4f} hours_to_expiry={hours_to_expiry:.1f}"}

    async def _analyze_market_movement(self, market: Market, side: str, current_price: float, *, required_exit_price: float, hours_to_expiry: float, market_volume: int, spread: float, recent_trade_stats: Optional[Dict[str, float]] = None) -> dict:
        if self.disable_ai or self.xai_client is None:
            return await self._heuristic_movement_analysis(market, side, current_price, required_exit_price=required_exit_price, hours_to_expiry=hours_to_expiry, spread=spread, recent_trade_stats=recent_trade_stats)
        prompt = f"""
QUICK SCALP ANALYSIS for {market.title}
Current {side} price: ${current_price:.2f}
Required profitable exit: ${required_exit_price:.2f}
Bid/ask spread: ${spread:.2f}
Market volume: {market_volume}
Hours to expiry: {hours_to_expiry:.1f}
Hold window: {self.config.max_hold_minutes} minutes
Respond with JSON only: {{"target_price": 0.23, "confidence": 0.72, "reason": "brief explanation"}}
"""
        try:
            response = await self.xai_client.get_completion(prompt=prompt, max_tokens=3000, strategy="quick_flip_scalping", query_type="movement_prediction", market_id=market.market_id)
            if response is None:
                raise ValueError("empty AI response")
            json_payload = response.strip()
            match = re.search(r"\{.*\}", json_payload, flags=re.DOTALL)
            if match:
                json_payload = match.group(0)
            try:
                parsed = json.loads(json_payload)
            except json.JSONDecodeError:
                parsed = json.loads(repair_json(json_payload))
            target_price = min(0.95, max(current_price, float(parsed.get("target_price", current_price))))
            confidence = float(parsed.get("confidence", 0.0))
            if not 0 <= confidence <= 1:
                confidence = 0.0
            return {"target_price": target_price, "confidence": confidence, "reason": str(parsed.get("reason", "Missing AI reason"))}
        except Exception:
            return await self._heuristic_movement_analysis(market, side, current_price, required_exit_price=required_exit_price, hours_to_expiry=hours_to_expiry, spread=spread, recent_trade_stats=recent_trade_stats)

    async def execute_quick_flip_opportunities(self, opportunities: List[QuickFlipOpportunity], *, shadow_mode: Optional[bool] = None) -> Dict:
        results = {"positions_created": 0, "sell_orders_placed": 0, "total_capital_used": 0.0, "expected_profit": 0.0, "failed_executions": 0, "positions_closed": 0, "gross_pnl": 0.0, "net_pnl": 0.0, "fees_paid": 0.0, "winning_trades": 0, "losing_trades": 0}
        for opportunity in opportunities:
            try:
                success = await self._execute_single_quick_flip(opportunity, shadow_mode=shadow_mode)
                if not success:
                    results["failed_executions"] += 1
                    continue
                results["positions_created"] += 1
                results["total_capital_used"] += opportunity.quantity * opportunity.entry_price
                results["expected_profit"] += opportunity.expected_profit
                sell_result = await self._place_immediate_sell_order(opportunity)
                if sell_result and sell_result.get("success"):
                    results["sell_orders_placed"] += 1
                    results["positions_closed"] += int(bool(sell_result.get("filled")))
                    results["gross_pnl"] += float(sell_result.get("gross_pnl", 0.0))
                    results["net_pnl"] += float(sell_result.get("net_pnl", 0.0))
                    results["fees_paid"] += float(sell_result.get("fees_paid", 0.0))
                    results["winning_trades"] += int(bool(sell_result.get("is_win")))
                    results["losing_trades"] += int(bool(sell_result.get("is_loss")))
            except Exception:
                results["failed_executions"] += 1
        return results

    async def _execute_single_quick_flip(self, opportunity: QuickFlipOpportunity, *, shadow_mode: Optional[bool] = None) -> bool:
        try:
            if self._reason_blocks_trade(opportunity.movement_indicator):
                return False
            live_mode = getattr(settings.trading, "live_trading_enabled", False)
            if shadow_mode is None:
                shadow_mode = getattr(settings.trading, "shadow_mode_enabled", False)
            try:
                existing_position = await self.db_manager.get_position_by_market_id(opportunity.market_id)
                if existing_position is not None:
                    return False
            except Exception:
                pass
            current_positions = await self._get_current_position_exposures()
            if not await self._passes_portfolio_enforcer(opportunity, live_mode=live_mode, shadow_mode=shadow_mode, current_positions=current_positions):
                return False
            position = Position(market_id=opportunity.market_id, side=opportunity.side, quantity=opportunity.quantity, entry_price=opportunity.entry_price, live=live_mode, timestamp=datetime.now(), rationale=f"QUICK FLIP: {opportunity.movement_indicator} | Target: ${opportunity.entry_price:.2f}->${opportunity.exit_price:.2f}", strategy="quick_flip_scalping")
            position_id = await self.db_manager.add_position(position)
            if position_id is None:
                return False
            position.id = position_id
            success = await self._execute_live_maker_entry(position) if live_mode else await execute_position(position=position, live_mode=False, db_manager=self.db_manager, kalshi_client=self.kalshi_client, shadow_mode=shadow_mode)
            if success:
                if not live_mode:
                    position.stop_loss_price = self._calculate_stop_loss_price(entry_price=position.entry_price, tick_size=opportunity.tick_size)
                    position.take_profit_price = opportunity.exit_price
                    position.max_hold_hours = max(1, math.ceil(self.config.max_hold_minutes / 60))
                    await self.db_manager.update_position_execution_details(position.id, entry_price=position.entry_price, quantity=position.quantity, live=False, stop_loss_price=position.stop_loss_price, take_profit_price=position.take_profit_price, max_hold_hours=position.max_hold_hours, entry_fee=position.entry_fee, contracts_cost=position.contracts_cost, entry_order_id=position.entry_order_id)
                self.active_positions[opportunity.market_id] = position
                return True
            await self.db_manager.update_position_status(position_id, "voided")
            return False
        except Exception:
            return False

    async def _record_paper_exit(self, *, position: Position, exit_price: float, rationale_suffix: str) -> Dict[str, float | bool]:
        exit_result = await record_simulated_position_exit(position=position, exit_price=exit_price, db_manager=self.db_manager, rationale_suffix=rationale_suffix, entry_maker=False, exit_maker=True, charge_entry_fee=True, charge_exit_fee=True)
        self.active_positions.pop(position.market_id, None)
        self.pending_sells.pop(position.market_id, None)
        return {"success": True, "filled": True, "gross_pnl": float(exit_result["gross_pnl"]), "net_pnl": float(exit_result["net_pnl"]), "fees_paid": float(exit_result["fees_paid"]), "is_win": bool(exit_result["is_win"]), "is_loss": bool(exit_result["is_loss"])}

    async def _get_paper_reachable_exit_price(self, *, position: Position, target_price: float) -> Optional[float]:
        market_response = await self.kalshi_client.get_market(position.market_id)
        market_info = market_response.get("market", {})
        if not market_info:
            return None
        tick_size = get_market_tick_size(market_info, position.entry_price)
        best_bid = get_best_bid_price(market_info, position.side)
        best_ask = get_best_ask_price(market_info, position.side)
        recent_trade_stats = await self._get_recent_trade_stats(position.market_id, position.side)
        if best_bid + 1e-9 >= target_price:
            return target_price
        if best_ask > 0 and abs(best_ask - target_price) <= (tick_size / 2.0) + 1e-9 and recent_trade_stats["recent_last_price"] + 1e-9 >= target_price and recent_trade_stats["recent_max_price"] + 1e-9 >= target_price:
            return target_price
        return None

    async def _place_immediate_sell_order(self, opportunity: QuickFlipOpportunity) -> Dict[str, float | bool]:
        try:
            position = self.active_positions.get(opportunity.market_id)
            if not position:
                return {"success": False}
            if getattr(settings.trading, "live_trading_enabled", False):
                return await self._place_live_dynamic_exit_order(position)
            sell_price = opportunity.exit_price
            success = await place_sell_limit_order(position=position, limit_price=sell_price, db_manager=self.db_manager, kalshi_client=self.kalshi_client, live_mode=False)
            if not success:
                return {"success": False}
            reconciliation = await reconcile_simulated_exit_orders(db_manager=self.db_manager, kalshi_client=self.kalshi_client, strategy="quick_flip_scalping", market_id=position.market_id)
            if int(reconciliation.get("positions_closed", 0)) > 0:
                self.active_positions.pop(position.market_id, None); self.pending_sells.pop(position.market_id, None)
                net_pnl = float(reconciliation.get("net_pnl", 0.0))
                return {"success": True, "filled": True, "gross_pnl": 0.0, "net_pnl": net_pnl, "fees_paid": float(reconciliation.get("fees_paid", 0.0)), "is_win": net_pnl > 0, "is_loss": net_pnl <= 0}
            self.pending_sells[opportunity.market_id] = {"position": position, "target_price": sell_price, "placed_at": datetime.now(), "max_hold_until": datetime.now() + timedelta(minutes=opportunity.max_hold_time)}
            return {"success": True, "filled": False, "gross_pnl": 0.0, "net_pnl": 0.0, "fees_paid": 0.0, "is_win": False, "is_loss": False}
        except Exception:
            return {"success": False}

    async def manage_active_positions(self) -> Dict:
        return await self.manage_live_positions(persisted_only=False)

    async def manage_live_positions(self, *, persisted_only: bool = True) -> Dict:
        results = {"positions_closed": 0, "positions_voided": 0, "positions_synced": 0, "orders_cancelled": 0, "side_mismatches": 0, "orders_adjusted": 0, "losses_cut": 0, "total_pnl": 0.0}
        live_mode = getattr(settings.trading, "live_trading_enabled", False)
        if live_mode:
            reconciliation = await self.reconcile_persisted_live_positions()
            for key in ("positions_closed", "positions_voided", "positions_synced", "orders_cancelled", "side_mismatches"):
                results[key] += int(reconciliation.get(key, 0))
            all_positions = await self.db_manager.get_open_live_positions()
            positions = [position for position in all_positions if position.strategy == "quick_flip_scalping"]
        else:
            reconciliation = await reconcile_simulated_exit_orders(db_manager=self.db_manager, kalshi_client=self.kalshi_client, strategy="quick_flip_scalping")
            results["positions_closed"] += int(reconciliation.get("positions_closed", 0))
            results["orders_cancelled"] += int(reconciliation.get("orders_cancelled", 0))
            results["total_pnl"] += float(reconciliation.get("net_pnl", 0.0))
            all_positions = await self.db_manager.get_open_non_live_positions()
            positions = [position for position in all_positions if position.strategy == "quick_flip_scalping"]
        current_time = datetime.now()
        for position in positions:
            try:
                market_response = await self.kalshi_client.get_market(position.market_id)
                market_info = market_response.get("market", {})
                if not market_info:
                    continue
                if live_mode:
                    positions_response = await self.kalshi_client.get_positions(ticker=position.market_id)
                    exposure = sum(get_position_exposure_dollars(item) for item in positions_response.get("market_positions", []) if get_position_exposure_dollars(item) > 0)
                    if exposure <= 0:
                        if await self._close_position_from_recent_fills(position, market_info=market_info):
                            results["positions_closed"] += 1
                        continue
                market_status = get_market_status(market_info)
                if not live_mode and market_status in {"closed", "settled", "finalized"}:
                    market_result = get_market_result(market_info)
                    exit_price = 1.0 if market_result and str(market_result).upper() == position.side.upper() else 0.0 if market_result else (get_last_price(market_info, position.side) or get_best_bid_price(market_info, position.side) or position.entry_price)
                    exit_result = await record_simulated_position_exit(position=position, exit_price=float(exit_price), db_manager=self.db_manager, rationale_suffix=f"PAPER QUICK FLIP MARKET RESOLUTION @ ${float(exit_price):.4f}", entry_maker=False, exit_maker=False, charge_entry_fee=True, charge_exit_fee=False)
                    self.active_positions.pop(position.market_id, None); self.pending_sells.pop(position.market_id, None)
                    results["positions_closed"] += 1; results["total_pnl"] += float(exit_result["net_pnl"])
                    continue
                current_price = get_best_bid_price(market_info, position.side)
                mark_price = max(current_price, get_last_price(market_info, position.side))
                if position.stop_loss_price and mark_price <= position.stop_loss_price:
                    if live_mode:
                        if await self._cut_losses_market_order(position):
                            results["losses_cut"] += 1
                    else:
                        if current_price <= 0 or current_price >= 1:
                            continue
                        exit_result = await record_simulated_position_exit(position=position, exit_price=current_price, db_manager=self.db_manager, rationale_suffix=f"PAPER QUICK FLIP STOP LOSS @ ${current_price:.4f}", entry_maker=False, exit_maker=False, charge_entry_fee=True, charge_exit_fee=True)
                        self.active_positions.pop(position.market_id, None); self.pending_sells.pop(position.market_id, None)
                        results["losses_cut"] += 1; results["positions_closed"] += 1; results["total_pnl"] += float(exit_result["net_pnl"])
                    continue
                desired_exit = await self._calculate_dynamic_exit_price(position, market_info)
                if desired_exit is None:
                    continue
                if live_mode:
                    existing_orders_response = await self.kalshi_client.get_orders(ticker=position.market_id, status="resting", limit=100)
                    existing_exit_orders = [order for order in existing_orders_response.get("orders", []) if str(order.get("action", "")).lower() == "sell" and str(order.get("side", "")).lower() == position.side.lower()]
                else:
                    existing_exit_orders = await self.db_manager.get_simulated_orders(strategy="quick_flip_scalping", market_id=position.market_id, side=position.side, action="sell", status="resting")
                tick_size = get_market_tick_size(market_info, position.entry_price)
                should_replace = not existing_exit_orders
                if existing_exit_orders:
                    current_order = existing_exit_orders[0]
                    if live_mode:
                        current_order_price = self._normalize_fill_price(float(current_order.get("yes_price_dollars" if position.side == "YES" else "no_price_dollars", 0) or 0))
                        placed_at = self.pending_sells.get(position.market_id, {}).get("placed_at")
                    else:
                        current_order_price = float(current_order.price or 0.0); placed_at = current_order.placed_at
                    age_seconds = (current_time - placed_at).total_seconds() if isinstance(placed_at, datetime) else self.config.dynamic_exit_reprice_seconds + 1
                    should_replace = abs(current_order_price - desired_exit) >= tick_size and age_seconds >= self.config.dynamic_exit_reprice_seconds
                    if should_replace:
                        if live_mode:
                            for order in existing_exit_orders:
                                if order.get("order_id"):
                                    await self.kalshi_client.cancel_order(order["order_id"])
                        else:
                            for order in existing_exit_orders:
                                await self.db_manager.update_simulated_order(int(order.id), status="cancelled")
                if should_replace:
                    if live_mode:
                        success = await place_sell_limit_order(position=position, limit_price=desired_exit, db_manager=self.db_manager, kalshi_client=self.kalshi_client, live_mode=True)
                        if success:
                            results["orders_adjusted"] += 1
                            self.pending_sells[position.market_id] = {"position": position, "target_price": desired_exit, "placed_at": current_time, "max_hold_until": current_time + timedelta(minutes=self.config.max_hold_minutes)}
                    else:
                        paper_result = await submit_simulated_sell_limit_order(position=position, limit_price=desired_exit, db_manager=self.db_manager, kalshi_client=self.kalshi_client)
                        if paper_result.get("success"):
                            results["orders_adjusted"] += int(paper_result.get("orders_placed", 0)); results["positions_closed"] += int(paper_result.get("positions_closed", 0)); results["total_pnl"] += float(paper_result.get("net_pnl", 0.0))
            except Exception as exc:
                self.logger.error(f"Error managing position {position.market_id}: {exc}")
        return results

    async def _cut_losses_market_order(self, position: Position) -> bool:
        try:
            market_response = await self.kalshi_client.get_market(position.market_id)
            market_info = market_response.get("market", {})
            exit_price = get_best_bid_price(market_info, position.side)
            if exit_price <= 0 or exit_price >= 1:
                return False
            order_params = {"ticker": position.market_id, "client_order_id": str(uuid.uuid4()), "side": position.side.lower(), "action": "sell", "count": position.quantity, "type_": "limit", "time_in_force": "immediate_or_cancel", "reduce_only": True, **build_limit_order_price_fields(position.side, exit_price)}
            if getattr(settings.trading, "live_trading_enabled", False):
                response = await self.kalshi_client.place_order(**order_params)
                if response and "order" in response:
                    fee_metadata = await self._resolve_fee_metadata(market_info)
                    await _record_live_exit_fee_divergence_if_filled(position=position, limit_price=exit_price, db_manager=self.db_manager, kalshi_client=self.kalshi_client, client_order_id=order_params["client_order_id"], order_response=response, fee_metadata=fee_metadata, allow_partial_fill=True)
                    return True
                return False
            return True
        except Exception as exc:
            self.logger.error(f"Error cutting losses: {exc}")
            return False


async def run_quick_flip_strategy(db_manager: DatabaseManager, kalshi_client: KalshiClient, xai_client: Any, available_capital: float, config: Optional[QuickFlipConfig] = None, *, disable_ai: Optional[bool] = None) -> Dict:
    try:
        config = config or QuickFlipConfig()
        strategy = QuickFlipScalpingStrategy(db_manager, kalshi_client, xai_client, config, disable_ai=disable_ai)
        markets = await db_manager.get_eligible_markets(volume_min=max(100, config.min_market_volume), max_days_to_expiry=max(1, math.ceil(config.max_hours_to_expiry / 24)))
        if not markets:
            return {"error": "No markets available"}
        snapshot_candidates = [market for market in markets if strategy._snapshot_candidate_matches(market)]
        if not snapshot_candidates:
            return {"opportunities_found": 0, "eligible_markets": len(markets), "snapshot_candidates": 0}
        snapshot_candidates.sort(key=lambda market: (market.expiration_ts, -market.volume, min(strategy._snapshot_prices_in_entry_band(market))))
        scan_limit = max(config.max_market_checks * 2, config.max_concurrent_positions * 4, 10)
        bounded_candidates = snapshot_candidates[:scan_limit]
        opportunities = await strategy.identify_quick_flip_opportunities(bounded_candidates, available_capital)
        if not opportunities:
            return {"opportunities_found": 0, "eligible_markets": len(markets), "snapshot_candidates": len(snapshot_candidates), "bounded_candidates": len(bounded_candidates)}
        execution_results = await strategy.execute_quick_flip_opportunities(opportunities)
        management_results = await strategy.manage_active_positions()
        return {**execution_results, **management_results, "opportunities_analyzed": len(opportunities), "eligible_markets": len(markets), "snapshot_candidates": len(snapshot_candidates), "bounded_candidates": len(bounded_candidates), "strategy": "quick_flip_scalping"}
    except Exception as exc:
        return {"error": str(exc)}


async def manage_live_quick_flip_positions(db_manager: DatabaseManager, kalshi_client: KalshiClient, *, config: Optional[QuickFlipConfig] = None) -> Dict:
    strategy = QuickFlipScalpingStrategy(db_manager=db_manager, kalshi_client=kalshi_client, xai_client=None, config=config or QuickFlipConfig())
    return await strategy.manage_live_positions(persisted_only=True)


async def reconcile_live_quick_flip_positions(db_manager: DatabaseManager, kalshi_client: KalshiClient, *, config: Optional[QuickFlipConfig] = None) -> Dict[str, int]:
    strategy = QuickFlipScalpingStrategy(db_manager=db_manager, kalshi_client=kalshi_client, xai_client=None, config=config or QuickFlipConfig())
    return await strategy.reconcile_persisted_live_positions()
