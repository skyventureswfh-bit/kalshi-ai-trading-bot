#!/usr/bin/env python3
"""Seed one harmless synthetic PENDING MANUAL APPROVAL record.

This script is only for verifying the dashboard approval plumbing. It does not
call Kalshi, does not place an order, and does not invoke the trading executor.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.database import DatabaseManager, LiveTradeDecision


TEST_SUMMARY = (
    "PENDING MANUAL APPROVAL — TEST ONLY; synthetic candidate for approval-queue verification."
)
TEST_RATIONALE = (
    "Harmless dashboard plumbing test. No Kalshi market is attached and no exchange order can be sent by this script."
)


async def main() -> None:
    db = DatabaseManager()
    await db.initialize()
    try:
        run_id = f"manual-approval-test-{uuid4().hex[:10]}"
        decision = LiveTradeDecision(
            created_at=datetime.now(timezone.utc),
            run_id=run_id,
            step="manual_approval_test",
            strategy="live_trade",
            status="pending",
            title="BEAST MANUAL APPROVAL — TEST CANDIDATE",
            focus_type="test",
            action="BUY",
            side="YES",
            confidence=0.99,
            edge_pct=0.0,
            limit_price=0.01,
            quantity=1.0,
            runtime_mode="manual-approval-test",
            paper_trade=False,
            live_trade=False,
            summary=TEST_SUMMARY,
            rationale=TEST_RATIONALE,
            payload_json='{"synthetic": true, "exchange_order_allowed": false}',
        )
        decision_id = await db.add_live_trade_decision(decision)
        print("BEAST MANUAL APPROVAL TEST SEEDED")
        print(f"Decision id: {decision_id}")
        print("No market was contacted and no real order was sent.")
        print("Refresh http://localhost:3000/live-trade/approval and approve or reject the TEST CANDIDATE.")
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
