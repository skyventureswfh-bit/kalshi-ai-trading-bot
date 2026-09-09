"""
Dedicated fail-closed kill switch for Beast Auto.

Design goals (per the locked Beast Auto V1 plan):
  - Fail closed: if the switch state cannot be determined, treat Auto as
    killed. Missing-file is the one exception — a fresh install with no
    kill-switch file yet must default to "not killed", or Auto could never
    start. Any other read failure (corrupt JSON, permission error, etc.)
    is treated as killed.
  - Checked at the start of every Auto cycle AND immediately before every
    Auto buy order is submitted (see src/auto/runner.py).
  - Independent of Manual. Activating/deactivating this switch has zero
    effect on Manual's pending-approval path or on existing strategy_halts.

This is intentionally file-based rather than a new DB table: it needs to
work even if the database is unavailable, and a human operator flipping it
by hand (or via `cli.py auto kill` / `cli.py auto resume`) should not
require the trading DB to be up.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class KillSwitchState:
    active: bool
    reason: Optional[str] = None
    activated_at: Optional[float] = None
    activated_by: Optional[str] = None


class AutoKillSwitch:
    """File-backed, fail-closed kill switch for the Auto supervisor."""

    def __init__(self, path: str) -> None:
        self.path = path

    def is_active(self) -> bool:
        """
        Return True if Auto must not submit new buy orders.

        Fail-closed: missing file => not active (False). Any other error
        reading/parsing the file => active (True).
        """
        if not os.path.exists(self.path):
            return False
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return bool(data.get("active", True))
        except Exception:
            # Corrupt or unreadable state file: fail closed.
            return True

    def read_state(self) -> KillSwitchState:
        """Return the full kill-switch state for status reporting/tests."""
        if not os.path.exists(self.path):
            return KillSwitchState(active=False)
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return KillSwitchState(
                active=bool(data.get("active", True)),
                reason=data.get("reason"),
                activated_at=data.get("activated_at"),
                activated_by=data.get("activated_by"),
            )
        except Exception as exc:
            return KillSwitchState(active=True, reason=f"unreadable state file: {exc}")

    def activate(self, *, reason: str, activated_by: str = "manual") -> None:
        """Activate the kill switch. Fails safe: raises on write failure
        rather than silently pretending the switch is now active."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload = {
            "active": True,
            "reason": reason,
            "activated_at": time.time(),
            "activated_by": activated_by,
        }
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    def deactivate(self, *, deactivated_by: str = "manual") -> None:
        """Deactivate the kill switch. Writes an explicit inactive record
        rather than deleting the file, so `read_state()` still reports who
        last touched it."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload = {
            "active": False,
            "reason": None,
            "activated_at": None,
            "activated_by": deactivated_by,
        }
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
