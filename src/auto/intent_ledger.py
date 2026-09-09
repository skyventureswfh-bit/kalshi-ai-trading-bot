"""
Idempotent order-intent tracking for Beast Auto.

Why position.id and not a per-cycle run_id:
    LiveTradeDecisionLoop.run_once() generates a fresh run_id
    (`uuid.uuid4().hex[:12]`) on every call, including retries after a
    crash. Deriving the client_order_id from that run_id would produce a
    NEW id on every restart — exactly the failure mode idempotency is
    supposed to prevent.

    position.id is the DB primary key created once via
    `db_manager.add_position(position)`, BEFORE execution is attempted, and
    the existing pipeline already treats (market_id, side) as unique at
    that point (add_position returns None for a duplicate). It is stable
    across a crash/restart because the position row itself persists.

    client_order_id = uuid5(NAMESPACE, f"{ticker}|{side}|{position.id}")

    is therefore identical every time it's computed for the same position,
    including after a crash and restart, without needing any new
    "decision identity" concept.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Fixed namespace so IDs are stable across process restarts and machines.
# This is a private, arbitrary UUID — not a Kalshi- or Auto-meaningful value.
AUTO_CLIENT_ORDER_ID_NAMESPACE = uuid.UUID("7b6f9e2a-2f7a-4b3a-9b2e-4b6f2b7e9c1a")


def deterministic_client_order_id(*, ticker: str, side: str, position_id: int) -> str:
    """Compute the stable client_order_id for a given position identity."""
    key = f"{ticker}|{side}|{position_id}"
    return str(uuid.uuid5(AUTO_CLIENT_ORDER_ID_NAMESPACE, key))


@dataclass
class ReconciledIntent:
    client_order_id: str
    local_status: str
    found_on_exchange: bool
    exchange_status: Optional[str] = None
    safe_to_retry: bool = False


class AutoIntentLedger:
    """
    Thin wrapper around the additive `auto_order_intents` table plus the
    Kalshi-side reconciliation check. Auto's runner is the only caller.
    """

    def __init__(self, db_manager: Any, kalshi_client: Any, strategy: str = "live_trade") -> None:
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.strategy = strategy

    async def record_before_submit(
        self, *, client_order_id: str, ticker: str, side: str, position_id: Optional[int]
    ) -> None:
        """Persist the intent BEFORE the order is submitted to Kalshi.
        Idempotent — safe to call again with the same client_order_id."""
        await self.db_manager.record_auto_order_intent(
            client_order_id=client_order_id,
            strategy=self.strategy,
            ticker=ticker,
            side=side,
            position_id=position_id,
        )

    async def mark_submitted(self, *, client_order_id: str) -> None:
        await self.db_manager.update_auto_order_intent_status(
            client_order_id=client_order_id, status="submitted"
        )

    async def mark_terminal(
        self, *, client_order_id: str, status: str, error: Optional[str] = None
    ) -> None:
        """status should be one of: 'filled', 'voided', 'failed'."""
        await self.db_manager.update_auto_order_intent_status(
            client_order_id=client_order_id, status=status, last_error=error
        )

    async def reconcile(self, intent: Dict[str, Any]) -> ReconciledIntent:
        """
        Check a non-terminal intent against Kalshi's own order records.

        This does NOT retry or resubmit anything. It only determines, from
        the exchange's point of view, what actually happened, so the
        caller (runner.py) can decide whether a retry using the SAME
        client_order_id would be safe:

          - found on exchange, terminal status (filled/canceled/rejected):
            mark our local record terminal to match. Never safe to retry
            with this ID once Kalshi has seen it as filled.
          - found on exchange, still resting/open: leave as 'submitted'.
            Not safe to retry (an order already exists).
          - NOT found on exchange at all: the original submission most
            likely never reached Kalshi (crash before/during the POST).
            Safe to retry using the same client_order_id, since Kalshi has
            no record of it and place_order is otherwise idempotent per
            the exchange's own client_order_id de-duplication.
        """
        client_order_id = intent["client_order_id"]
        ticker = intent["ticker"]
        try:
            response = await self.kalshi_client.get_orders(ticker=ticker, limit=200)
        except Exception as exc:
            # Cannot reach Kalshi to check — do NOT guess. Fail closed:
            # never treat an unreconciled intent as safe to retry.
            return ReconciledIntent(
                client_order_id=client_order_id,
                local_status=intent.get("status", "pending"),
                found_on_exchange=False,
                exchange_status=f"reconciliation_error: {exc}",
                safe_to_retry=False,
            )

        orders = response.get("orders", []) if isinstance(response, dict) else []
        match = next(
            (o for o in orders if o.get("client_order_id") == client_order_id),
            None,
        )
        if match is None:
            return ReconciledIntent(
                client_order_id=client_order_id,
                local_status=intent.get("status", "pending"),
                found_on_exchange=False,
                safe_to_retry=True,
            )

        exchange_status = str(match.get("status") or "").lower()
        terminal_statuses = {"filled", "canceled", "cancelled", "rejected", "expired"}
        if exchange_status in terminal_statuses:
            await self.mark_terminal(
                client_order_id=client_order_id,
                status="filled" if exchange_status == "filled" else "voided",
            )
        return ReconciledIntent(
            client_order_id=client_order_id,
            local_status=intent.get("status", "pending"),
            found_on_exchange=True,
            exchange_status=exchange_status,
            safe_to_retry=False,
        )
