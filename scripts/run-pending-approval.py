#!/usr/bin/env python3
"""Continuously hunt live Kalshi markets while hard-blocking real order execution.

This launcher uses live-data decision semantics so the scout, specialists,
debate, sizing, and pre-execution checks see realistic production context. The
final guardrail is replaced with an unconditional manual-approval hold, so no
BUY can reach ``execute_position`` from this script.

Qualified candidates are persisted as ``PENDING MANUAL APPROVAL`` for review in
the dashboard. The hunter repeats until the operator stops it with Ctrl+C.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

# Allow direct execution from ``scripts/`` without requiring PYTHONPATH=.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clients.kalshi_client import KalshiClient
from src.config.settings import settings
from src.jobs.live_trade import LiveTradeDecisionLoop
from src.utils.database import DatabaseManager


DEFAULT_INTERVAL_SECONDS = 300.0
PENDING_REASON = "PENDING MANUAL APPROVAL — candidate prepared; no real order was sent."


async def _pending_approval_guardrail(**_kwargs: Any) -> tuple[bool, str]:
    """Fail closed immediately before execution until the operator approves."""
    return False, PENDING_REASON


def _summary_value(summary: Any, *names: str) -> int:
    """Read a count from either dict-style or attribute-style run summaries."""
    for name in names:
        if isinstance(summary, dict) and name in summary:
            try:
                return int(summary.get(name) or 0)
            except (TypeError, ValueError):
                return 0
        if hasattr(summary, name):
            try:
                return int(getattr(summary, name) or 0)
            except (TypeError, ValueError):
                return 0
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Beast continuously with live data and manual approval required."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=float(
            os.getenv("BEAST_APPROVAL_SCAN_INTERVAL_SECONDS", DEFAULT_INTERVAL_SECONDS)
        ),
        help="Seconds between decision cycles (default: 300).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one decision cycle and exit.",
    )
    return parser.parse_args()


async def _run_cycle(loop: LiveTradeDecisionLoop, cycle_number: int) -> None:
    print(f"\n=== BEAST HUNT CYCLE {cycle_number} ===")
    summary = await loop.run_once()
    events = _summary_value(summary, "events_considered", "events_scanned")
    shortlisted = _summary_value(summary, "events_shortlisted", "shortlisted_events")
    specialists = _summary_value(summary, "specialists_completed", "specialist_candidates")
    executed = _summary_value(summary, "trades_executed", "executed_positions")
    print(
        "Decision cycle complete: "
        f"events={events}, shortlisted={shortlisted}, "
        f"specialists={specialists}, executed={executed}"
    )
    print("Qualified candidates remain PENDING MANUAL APPROVAL; no real order was sent.")


async def _main(args: argparse.Namespace) -> None:
    interval = max(30.0, float(args.interval))
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

        print("BEAST CONTINUOUS MANUAL-APPROVAL MODE")
        print("Live data/decision semantics are active; real order execution is hard-blocked.")
        if args.once:
            print("One-cycle mode selected.")
        else:
            print(f"Hunter interval: {interval:.0f} seconds. Press Ctrl+C to stop cleanly.")

        cycle_number = 1
        while True:
            try:
                await _run_cycle(loop, cycle_number)
            except Exception as exc:
                # Keep the hunter alive across transient provider/model failures. The
                # live-trade loop itself remains fail-closed and cannot execute here.
                print(f"Cycle {cycle_number} failed safely: {exc}")

            if args.once:
                break

            cycle_number += 1
            print(f"Next hunt in {interval:.0f} seconds...")
            await asyncio.sleep(interval)
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
    try:
        asyncio.run(_main(_parse_args()))
    except KeyboardInterrupt:
        print("\nBeast hunter stopped by operator. No automatic order execution was enabled.")
