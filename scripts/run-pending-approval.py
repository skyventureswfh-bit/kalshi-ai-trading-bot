#!/usr/bin/env python3
"""Run one live-data Beast decision cycle and stop before any real order is sent.

This launcher intentionally sets the decision loop to live-data semantics so the
scout, specialists, debate, sizing, and pre-execution checks see the same runtime
context as a live cycle. The final guardrail is then replaced with an unconditional
manual-approval hold. No BUY can reach ``execute_position`` from this script.

The resulting execution decision is persisted with the reason
``PENDING MANUAL APPROVAL`` so the dashboard can surface the candidate for review.
"""

from __future__ import annotations

import asyncio

from src.clients.kalshi_client import KalshiClient
from src.config.settings import settings
from src.jobs.live_trade import LiveTradeDecisionLoop
from src.utils.database import DatabaseManager


async def _pending_approval_guardrail(**_kwargs):
    """Fail closed immediately before execution until the operator approves."""
    return (
        False,
        "PENDING MANUAL APPROVAL — candidate prepared; no real order was sent.",
    )


async def _main() -> None:
    db_manager = DatabaseManager()
    kalshi_client = KalshiClient()

    previous_live = bool(getattr(settings.trading, "live_trading_enabled", False))
    previous_shadow = bool(getattr(settings.trading, "shadow_mode_enabled", False))

    # Match live decision semantics while the injected guardrail guarantees
    # execution cannot proceed to the exchange.
    settings.trading.live_trading_enabled = True
    settings.trading.shadow_mode_enabled = False

    loop = LiveTradeDecisionLoop(
        db_manager=db_manager,
        kalshi_client=kalshi_client,
        guardrail_fn=_pending_approval_guardrail,
        manage_quick_flip_positions_each_cycle=False,
    )

    try:
        initialize = getattr(db_manager, "initialize", None)
        if callable(initialize):
            result = initialize()
            if asyncio.iscoroutine(result):
                await result

        print("BEAST MANUAL-APPROVAL MODE")
        print("Live data/decision semantics are active; order execution is hard-blocked.")
        summary = await loop.run_once()
        print(
            "Decision cycle complete: "
            f"events={getattr(summary, 'events_scanned', 0)}, "
            f"shortlisted={getattr(summary, 'shortlisted_events', 0)}, "
            f"specialists={getattr(summary, 'specialist_candidates', 0)}, "
            f"executed={getattr(summary, 'executed_positions', 0)}"
        )
        print("Any approved candidate remains PENDING MANUAL APPROVAL; no real order was sent.")
    finally:
        settings.trading.live_trading_enabled = previous_live
        settings.trading.shadow_mode_enabled = previous_shadow
        await loop.close()
        await kalshi_client.close()
        close_db = getattr(db_manager, "close", None)
        if callable(close_db):
            result = close_db()
            if asyncio.iscoroutine(result):
                await result


if __name__ == "__main__":
    asyncio.run(_main())
