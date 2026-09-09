"""
Beast Auto V1 — autonomous supervisor around the existing decision/risk/
execution pipeline.

This module does NOT reimplement decision-making, sizing, or risk gating.
It wraps the existing `LiveTradeDecisionLoop` (src/jobs/live_trade.py,
UNCHANGED) using its existing `execute_position_fn` dependency-injection
seam, so every trade Auto places passes through exactly the same gates a
Manual trade would. Auto adds only:

  - a dedicated fail-closed kill switch, checked at cycle start and again
    immediately before each buy order reaches `execute_position()`
  - deterministic, persisted client_order_ids for buy orders, keyed to the
    stable DB position.id (see intent_ledger.py for why)
  - a consecutive-failure counter that halts Auto fail-closed
  - a process-wide OpenRouter low-credit circuit breaker that halts Auto
    after an unaffordable-input 402 instead of hammering the API each cycle
  - graceful SIGINT/SIGTERM shutdown
  - startup reconciliation of any order intent left in a non-terminal
    state by a prior crash (detection + bookkeeping only — V1 does not
    automatically resubmit; see reconcile_on_startup() docstring)

Quick Flip is intentionally out of scope: this runner only ever triggers
`run_once()`'s live-trade path. `manage_quick_flip_positions_each_cycle`
is left False, matching the LiveTradeDecisionLoop default, so Auto never
touches Quick Flip positions or intents.

Sell orders (place_sell_limit_order) are NOT wrapped and receive no
deterministic client_order_id in V1 — see the risk report for why this is
a known, accepted limitation rather than an oversight.
"""

from __future__ import annotations

import asyncio
import logging
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


@dataclass
class CycleOutcome:
    status: str  # "ok" | "skipped_kill_switch" | "skipped_halted" | "credit_exhausted" | "failed"
    detail: Optional[str] = None


class AutoRunner:
    """
    Owns one LiveTradeDecisionLoop instance, configured with a wrapped
    execute_position_fn that adds the kill-switch check and deterministic
    client_order_id, then drives it on a timed loop with fail-closed
    failure handling.
    """

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

        # Wrap the raw client ONCE, here, before it's used anywhere else.
        # Everything downstream -- the decision loop's own Kalshi calls,
        # the intent ledger's reconciliation reads, and the execute_position
        # wrapper below -- receives this guarded instance. SELL passes
        # through unchanged; BUY is gated immediately before the real
        # place_order() call, closing the race window that existed between
        # this point and the actual POST.
        self.kalshi_client = AutoGuardedKalshiClient(kalshi_client, self.kill_switch)

        self.ledger = AutoIntentLedger(
            db_manager=db_manager, kalshi_client=self.kalshi_client, strategy=strategy_label
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
                client_order_id=client_order_id, status="failed", error=str(exc)
            )
            raise

        await self.ledger.mark_terminal(
            client_order_id=client_order_id,
            status="filled" if success else "voided",
        )
        return success

    async def reconcile_on_startup(self) -> List[Dict[str, Any]]:
        """
        Check every non-terminal Auto order intent against Kalshi's own
        order records and update local bookkeeping to match reality.

        V1 scope: detection and bookkeeping ONLY. This does NOT
        automatically resubmit an order, even when reconciliation
        determines a retry would be safe.
        """
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
        self, *, loop_status: str, summary: str, error: Optional[str] = None
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

    async def run_one_cycle(self) -> CycleOutcome:
        if self.kill_switch.is_active():
            await self._persist_heartbeat(loop_status="killed", summary="Kill switch active")
            return CycleOutcome(status="skipped_kill_switch")

        # If a prior request in this process already proved the account cannot
        # fund the input prompt, fail closed before another cycle touches the API.
        if prompt_credit_exhausted():
            detail = "OpenRouter credit cannot fund the live-trade prompt"
            await self._persist_heartbeat(
                loop_status="killed",
                summary=detail,
                error="openrouter_prompt_credit_exhausted",
            )
            return CycleOutcome(status="credit_exhausted", detail=detail)

        if await self.db_manager.is_strategy_halted_today(strategy=self.strategy_label):
            await self._persist_heartbeat(loop_status="halted", summary="Strategy halt active")
            return CycleOutcome(status="skipped_halted")

        try:
            summary = await asyncio.wait_for(self.decision_loop.run_once(), timeout=300)

            # OpenRouterClient returns None on request failure, so the decision
            # loop may complete with a skip instead of raising. Check the latch
            # after the cycle as well; this is what converts today's repeated
            # 402 pattern into one failed request followed by a hard Auto halt.
            if prompt_credit_exhausted():
                detail = "OpenRouter credit cannot fund the live-trade prompt"
                await self._persist_heartbeat(
                    loop_status="killed",
                    summary=detail,
                    error="openrouter_prompt_credit_exhausted",
                )
                return CycleOutcome(status="credit_exhausted", detail=detail)

            await self._persist_heartbeat(
                loop_status="idle",
                summary=getattr(summary, "skipped_reason", None) or "Cycle completed",
            )
            return CycleOutcome(status="ok")
        except Exception as exc:
            logger.exception("Auto cycle failed")
            await self._persist_heartbeat(loop_status="error", summary="Cycle failed", error=str(exc))
            return CycleOutcome(status="failed", detail=str(exc))

    async def run_forever(self) -> None:
        self._install_signal_handlers()
        await self.reconcile_on_startup()

        while not self._stop_event.is_set():
            outcome = await self.run_one_cycle()

            if outcome.status == "credit_exhausted":
                self.kill_switch.activate(
                    reason=outcome.detail or "OpenRouter prompt credit exhausted",
                    activated_by="auto_runner",
                )
                logger.error(
                    "Auto halted fail-closed because OpenRouter cannot fund the input prompt"
                )
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
                    self._stop_event.wait(), timeout=self.config.cadence_seconds
                )
            except asyncio.TimeoutError:
                pass

        logger.info("Auto runner stopped gracefully")
