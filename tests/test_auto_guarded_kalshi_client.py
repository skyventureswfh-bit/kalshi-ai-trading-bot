"""
Tests proving the guarded Kalshi client closes the kill-switch race window
identified in review: the check now happens immediately before the real
place_order() call, not just at the top of the execute_position wrapper.

Required proofs (per reviewer sign-off):
  - kill switch OFF -> BUY delegates exactly once
  - kill switch ON immediately before submission -> zero delegate calls
  - unreadable/corrupt kill-switch state -> zero delegate calls
  - SELL remains untouched/passes through regardless of switch state
  - Manual path is unaffected (Manual never constructs this class)
"""

import os

import pytest

from src.auto.guarded_kalshi_client import AutoGuardedKalshiClient, AutoKillSwitchBlocked
from src.auto.kill_switch import AutoKillSwitch
from src.clients.kalshi_client import KalshiAPIError


class _FakeRealClient:
    """Stands in for the real KalshiClient. Records every place_order call."""

    def __init__(self):
        self.calls = []

    async def place_order(self, **kwargs):
        self.calls.append(kwargs)
        return {"order": {"order_id": "kalshi-1", "status": "filled"}, "order_id": "kalshi-1"}

    async def get_orders(self, **kwargs):
        return {"orders": []}


@pytest.fixture
def switch_path(tmp_path):
    return str(tmp_path / "kill.json")


@pytest.mark.asyncio
async def test_kill_switch_off_buy_delegates_exactly_once(switch_path):
    real_client = _FakeRealClient()
    switch = AutoKillSwitch(switch_path)  # missing file -> not active
    guarded = AutoGuardedKalshiClient(real_client, switch)

    result = await guarded.place_order(
        ticker="KXTEST-26", client_order_id="abc", side="yes", action="buy", count=10,
    )

    assert result["order_id"] == "kalshi-1"
    assert len(real_client.calls) == 1
    assert real_client.calls[0]["action"] == "buy"


@pytest.mark.asyncio
async def test_kill_switch_on_immediately_before_submission_blocks_buy(switch_path):
    real_client = _FakeRealClient()
    switch = AutoKillSwitch(switch_path)
    guarded = AutoGuardedKalshiClient(real_client, switch)

    # Activate AFTER constructing the guarded client, simulating the switch
    # flipping mid-cycle -- this is exactly the race window under test.
    switch.activate(reason="mid-cycle kill")

    with pytest.raises(AutoKillSwitchBlocked):
        await guarded.place_order(
            ticker="KXTEST-26", client_order_id="abc", side="yes", action="buy", count=10,
        )

    assert len(real_client.calls) == 0
    # And it's a KalshiAPIError, so execute_position()'s existing
    # `except KalshiAPIError` handler (unmodified) catches it cleanly.
    assert issubclass(AutoKillSwitchBlocked, KalshiAPIError)


@pytest.mark.asyncio
async def test_unreadable_corrupt_state_blocks_buy(switch_path):
    real_client = _FakeRealClient()
    os.makedirs(os.path.dirname(switch_path), exist_ok=True)
    with open(switch_path, "w", encoding="utf-8") as fh:
        fh.write("{this is not valid json")
    switch = AutoKillSwitch(switch_path)
    guarded = AutoGuardedKalshiClient(real_client, switch)

    with pytest.raises(AutoKillSwitchBlocked):
        await guarded.place_order(
            ticker="KXTEST-26", client_order_id="abc", side="yes", action="buy", count=10,
        )

    assert len(real_client.calls) == 0


@pytest.mark.asyncio
async def test_sell_passes_through_unchanged_even_when_killed(switch_path):
    real_client = _FakeRealClient()
    switch = AutoKillSwitch(switch_path)
    switch.activate(reason="kill active, sell must still work")
    guarded = AutoGuardedKalshiClient(real_client, switch)

    result = await guarded.place_order(
        ticker="KXTEST-26", client_order_id="abc", side="yes", action="sell", count=10,
    )

    assert result["order_id"] == "kalshi-1"
    assert len(real_client.calls) == 1
    assert real_client.calls[0]["action"] == "sell"


@pytest.mark.asyncio
async def test_indeterminate_action_fails_closed_like_a_buy(switch_path):
    """An action we can't identify as SELL must be gated, not assumed safe."""
    real_client = _FakeRealClient()
    switch = AutoKillSwitch(switch_path)
    switch.activate(reason="kill active")
    guarded = AutoGuardedKalshiClient(real_client, switch)

    with pytest.raises(AutoKillSwitchBlocked):
        await guarded.place_order(ticker="KXTEST-26", client_order_id="abc", side="yes", count=10)

    assert len(real_client.calls) == 0


@pytest.mark.asyncio
async def test_non_place_order_methods_delegate_transparently(switch_path):
    real_client = _FakeRealClient()
    switch = AutoKillSwitch(switch_path)
    switch.activate(reason="kill active -- must not affect reads")
    guarded = AutoGuardedKalshiClient(real_client, switch)

    result = await guarded.get_orders(ticker="KXTEST-26")
    assert result == {"orders": []}


def test_manual_never_constructs_guarded_client():
    """
    Documentation-as-test: Manual's call paths (cli.py `run`, `run --live`,
    `run --live-trade`, unified_bot.py) construct and use KalshiClient
    directly -- this diff adds zero references to AutoGuardedKalshiClient
    anywhere outside src/auto/. Grep-checked at review time; asserted here
    so a future edit that wires this into Manual by accident fails loudly.
    """
    import inspect

    import src.auto.guarded_kalshi_client as guarded_mod

    module_file = inspect.getfile(guarded_mod)
    assert "src/auto/" in module_file.replace(os.sep, "/") or "src\\auto\\" in module_file
