"""
Beast Auto V1 — autonomous supervisor around the existing decision/risk/
execution pipeline.

This module does NOT reimplement decision-making, sizing, or risk gating.
It wraps the existing `LiveTradeDecisionLoop` using its existing
`execute_position_fn` dependency-injection seam, so every trade Auto places
passes through the same execution gates a Manual trade would. Auto adds only:

  - a dedicated fail-closed kill switch, checked at cycle start and again
    immediately before each buy order reaches the exchange
  - deterministic, persisted client_order_ids for buy orders
  - a consecutive-failure counter that halts Auto fail-closed
  - a process-wide OpenRouter low-credit circuit breaker
  - a conservative Auto-only daily AI spend ceiling (default $0.50, override
    with AUTO_AI_DAILY_COST_LIMIT)
  - graceful SIGINT/SIGTERM shutdown
  - startup reconciliation of unresolved order intents without auto-resubmission

Quick Flip is intentionally out of scope. Sell orders are not wrapped with a
deterministic client_order_id in V1.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.auto.guarded_kalshi_client import AutoGuardedKalshiClient
from src.auto.intent_ledger import AutoIntentLedger, deterministic_client_order_id
from src.auto.kill_switch import AutoKillSwitch
from src.clients.openrouter_cost_guard import prompt_credit_exhausted
from src.config.settings import AutoConfig
from src.jobs.execute import execute_position
from src.jobs.live_trade import LiveTradeDecisionLoop

logger = logging.getLogger("beast_auto")


def _auto_ai_daily_limit_from_env() -> float:
    """Auto-only AI spend ceiling. Zero disables this extra Auto gate."""
    raw = os.getenv("AUTO_AI_DAILY_COST_LIMIT", "0.50")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        logger.warning(
            "Invalid AUTO_AI_DAILY_COST_LIMIT=%r; using conservative $0.50 default",
            raw,
        )
        return 0.50


@dataclass
class CycleOutcome:
    status: str
    detail: Optional[str] = None


class AutoRunner:
    """Unattended supervisor around one LiveTradeDecisionLoop instance."""

    def __init__(
        self,
        *,
        db_manager: Any,
        kalshi_client: Any,
        config: Optional[AutoConfig] = None,
        model_router: Any = None,
        strategy_label: str = "live_trade",
        decision_loop_factory: Optional[Callable[..., LiveTradeDecisionLoop]] = None,
    ) -> None:
        self.db_manager = db_manager
        self.config = config or AutoConfig()
        self.strategy_label = strategy_label
        self.kill_switch = AutoKillSwitch(self.config.kill_switch_path)
        self.max_ai_daily_cost_usd = _auto_ai_daily_limit_from_env()

        self.kalshi_client = AutoGuardedKalshiClient(kalshi_client, self.kill_switch)
        self.ledger = AutoIntentLedger(
            db_manager=db_manager,
            kalshi_client=self.kalshi_client,
            strategy=strategy_label,
        )
        self._consecutive_failures = 0
        self._stop_event = asyncio.Event()

        factory = decision_loop_factory or LiveTradeDecisionLoop
        self.decision_loop = factory(
            db_manager=db_manager,
            kalshi_client=self.kalshi_client,
            model_router=model_router,
            execute_position_fn=self._auto_execute_position,
            manage_quick_flip_positions_each_cycle=False,
        )

    def _current_shared_ai_cost_usd(self) -> Optional[float]:
        """Read the shared on-disk AI usage tracker through ModelRouter."""
        router = getattr(self.decision_loop, "model_router", None)
        if router is None:
            return None
        try:
            loader = getattr(router, "_load_daily_tracker", None)
            tracker = loader() if callable(loader) else getattr(router, "daily_tracker", None)
            if tracker is None:
                return None
            return float(getattr(tracker, "total_cost", 0.0) or 0.0)
        except Exception as exc:
            logger.warning("Could not read shared AI cost tracker: %s", exc)
            return None

    def _ai_budget_outcome(self) -> Optional[CycleOutcome]:
        if self.max_ai_daily_cost_usd <= 0:
            return None
        current = self._current_shared_ai_cost_usd()
        if current is None or current < self.max_ai_daily_cost_usd:
            return None
        return CycleOutcome(
            status="ai_budget_exhausted",
            detail=(
                f"Auto AI daily spend limit reached: ${current:.4f} >= "
                f"${self.max_ai_daily_cost_usd:.2f}"
            ),
        )

    async def _auto_execute_position(self, **kwargs: Any) -> bool:
        position = kwargs["position"]
        live_mode = kwargs.get("live_mode", False)

        if not live_mode:
            return await execute_position(**kwargs)

        if self.kill_switch.is_active():
            logger.warning(
                "Auto kill switch active — blocking buy for %s (%s)",
                position.market_id,
                position.side,
            )
            return False

        client_order_id = deterministic_client_order_id(
            ticker=position.market_id,
            side=position.side,
            position_id=position.id,
        )
        await self.ledger.record_before_submit(
            client_order_id=client_order_id,
            ticker=position.market_id,
            side=position.side,
            position_id=position.id,
        )
        await self.ledger.mark_submitted(client_order_id=client_order_id)

        try:
            success = await execute_position(**kwargs, client_order_id=client_order_id)
        except Exception as exc:
            await self.ledger.mark_terminal(
                client_order_id=client_order_id,
                status="failed",
                error=str(exc),
            )
            raise

        await self.ledger.mark_terminal(
            client_order_id=client_order_id,
            status="filled" if success else "voided",
        )
        return success

    async def reconcile_on_startup(self) -> List[Dict[str, Any]]:
        unresolved = await self.db_manager.list_unresolved_auto_order_intents(
            strategy=self.strategy_label
        )
        results = []
        for intent in unresolved:
            reconciled = await self.ledger.reconcile(intent)
            logger.info(
                "Startup reconciliation: %s found_on_exchange=%s status=%s safe_to_retry=%s",
                reconciled.client_order_id,
                reconciled.found_on_exchange,
                reconciled.exchange_status,
                reconciled.safe_to_retry,
            )
            results.append(reconciled.__dict__)
        return results

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop_event.set)
            except (NotImplementedError, RuntimeError):
                logger.warning("Could not install signal handler for %s", sig)

    async def _persist_heartbeat(
        self,
        *,
        loop_status: str,
        summary: str,
        error: Optional[str] = None,
    ) -> None:
        try:
            from src.utils.database import LiveTradeRuntimeState

            await self.db_manager.upsert_live_trade_runtime_state(
                LiveTradeRuntimeState(
                    strategy=self.strategy_label,
                    worker="auto_runner",
                    heartbeat_at=datetime.now(timezone.utc).isoformat(),
                    loop_status=loop_status,
                    last_summary=summary,
                    error=error,
                )
            )
        except Exception as exc:
            logger.error("Failed to persist Auto heartbeat: %s", exc)

    async def _cost_stop_outcome(self) -> Optional[CycleOutcome]:
        if prompt_credit_exhausted():
            detail = "OpenRouter credit cannot fund the live-trade prompt"
            await self._persist_heartbeat(
                loop_status="killed",
                summary=detail,
                error="openrouter_prompt_credit_exhausted",
            )
            return CycleOutcome(status="credit_exhausted", detail=detail)

        budget = self._ai_budget_outcome()
        if budget is not None:
            await self._persist_heartbeat(
                loop_status="killed",
                summary=budget.detail or "Auto AI daily spend limit reached",
                error="auto_ai_daily_budget_exhausted",
            )
            return budget
        return None

    async def run_one_cycle(self) -> CycleOutcome:
        if self.kill_switch.is_active():
            await self._persist_heartbeat(
                loop_status="killed",
                summary="Kill switch active",
            )
            return CycleOutcome(status="skipped_kill_switch")

        cost_stop = await self._cost_stop_outcome()
        if cost_stop is not None:
            return cost_stop

        if await self.db_manager.is_strategy_halted_today(strategy=self.strategy_label):
            await self._persist_heartbeat(
                loop_status="halted",
                summary="Strategy halt active",
            )
            return CycleOutcome(status="skipped_halted")

        try:
            summary = await asyncio.wait_for(self.decision_loop.run_once(), timeout=300)

            cost_stop = await self._cost_stop_outcome()
            if cost_stop is not None:
                return cost_stop

            await self._persist_heartbeat(
                loop_status="idle",
                summary=getattr(summary, "skipped_reason", None) or "Cycle completed",
            )
            return CycleOutcome(status="ok")
        except Exception as exc:
            logger.exception("Auto cycle failed")
            await self._persist_heartbeat(
                loop_status="error",
                summary="Cycle failed",
                error=str(exc),
            )
            return CycleOutcome(status="failed", detail=str(exc))

    async def run_forever(self) -> None:
        self._install_signal_handlers()
        await self.reconcile_on_startup()

        while not self._stop_event.is_set():
            outcome = await self.run_one_cycle()

            if outcome.status in {"credit_exhausted", "ai_budget_exhausted"}:
                self.kill_switch.activate(
                    reason=outcome.detail or "Auto AI cost circuit breaker opened",
                    activated_by="auto_runner",
                )
                logger.error("Auto halted fail-closed: %s", outcome.detail)
                break

            if outcome.status == "failed":
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.config.max_consecutive_failures:
                    self.kill_switch.activate(
                        reason=(
                            f"{self._consecutive_failures} consecutive cycle "
                            f"failures (last: {outcome.detail})"
                        ),
                        activated_by="auto_runner",
                    )
                    logger.error(
                        "Auto halted fail-closed after %d consecutive failures",
                        self._consecutive_failures,
                    )
                    break
            else:
                self._consecutive_failures = 0

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.config.cadence_seconds,
                )
            except asyncio.TimeoutError:
                pass

        logger.info("Auto runner stopped gracefully")
