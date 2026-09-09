"""
Auto-only guarded Kalshi client.

Closes the race window identified in review: the earlier kill-switch check
in AutoRunner._auto_execute_position happens before the quote fetch and
safety re-check inside execute_position() -- several awaits, and
therefore several context-switch points, before the real place_order()
call. This wrapper moves the authoritative, final check to the literal
call site immediately preceding the real HTTP request.

Design constraints (per locked review):
  - Does NOT modify src/clients/kalshi_client.py.
  - Manual never constructs or sees this class -- it continues calling
    KalshiClient directly, unmodified, unaffected.
  - SELL orders always pass straight through, unchanged, no gating.
  - BUY orders (and any order whose action cannot be determined) are
    gated: if the kill switch is active, OR its state cannot be
    confirmed (missing is fine and means "not active" -- see
    AutoKillSwitch.is_active() for the fail-closed contract on
    unreadable/corrupt state), the real place_order() is never called.
  - Every other KalshiClient method (get_orders, get_balance, get_series,
    cancel_order, etc.) is transparently delegated, untouched.
"""

from __future__ import annotations

from typing import Any

from src.auto.kill_switch import AutoKillSwitch
from src.clients.kalshi_client import KalshiAPIError


class AutoKillSwitchBlocked(KalshiAPIError):
    """
    Raised in place of delegating to the real exchange call when the Auto
    kill switch is active (or its state could not be confirmed) at the
    moment a BUY would otherwise reach Kalshi.

    Subclasses KalshiAPIError deliberately: execute_position() already
    catches KalshiAPIError from the live-buy block and returns False with
    a clean log line (see src/jobs/execute.py, unmodified). Raising this
    type means the existing error handling absorbs the block correctly
    with zero changes to execute.py.
    """


class AutoGuardedKalshiClient:
    """Wraps a real KalshiClient. Auto-only; Manual never touches this."""

    def __init__(self, kalshi_client: Any, kill_switch: AutoKillSwitch) -> None:
        # Leading underscore attributes set via __dict__ directly so
        # __getattr__ below (which only fires for MISSING attributes)
        # doesn't recurse trying to look them up on the wrapped client.
        self.__dict__["_kalshi_client"] = kalshi_client
        self.__dict__["_kill_switch"] = kill_switch

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes not found on this wrapper itself --
        # i.e. everything except place_order, __init__, __aenter__, __aexit__.
        return getattr(self._kalshi_client, name)

    async def place_order(self, *args: Any, **kwargs: Any) -> Any:
        action = kwargs.get("action")
        if action is None and len(args) >= 4:
            # KalshiClient.place_order(self, ticker, client_order_id, side, action, ...)
            action = args[3]
        is_sell = str(action or "").strip().lower() == "sell"

        if not is_sell:
            # BUY, or an indeterminate action: never assume it's safe to
            # skip the check. is_active() itself fails closed on a
            # missing-vs-corrupt state file distinction; we just act on it.
            if self._kill_switch.is_active():
                raise AutoKillSwitchBlocked(
                    "Auto kill switch active (or its state could not be "
                    "confirmed) immediately before order submission -- "
                    "BUY blocked, never sent to Kalshi."
                )

        return await self._kalshi_client.place_order(*args, **kwargs)

    async def __aenter__(self) -> "AutoGuardedKalshiClient":
        await self._kalshi_client.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self._kalshi_client.__aexit__(exc_type, exc_val, exc_tb)
