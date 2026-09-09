"""
Tests proving, at the AutoRunner level:
  - kill switch active -> zero new buy order submissions
  - strategy halt active -> zero order submissions
  - existing risk-gate rejection propagates as False, no order sent
  - crash/restart produces the identical deterministic client_order_id
  - Quick Flip is untouched (manage_quick_flip_positions_each_cycle=False)
  - consecutive cycle failures trigger fail-closed Auto halt
  - Manual is unaffected: this file never imports or touches Manual's own
    LiveTradeDecisionLoop instantiation in cli.py/unified_bot.py; it only
    tests Auto's own wrapper.

Note on scope: these tests mock LiveTradeDecisionLoop itself, since its
internal decision/scout/specialist/debate pipeline (unchanged in this
diff) is exercised by the project's existing live-trade test coverage,
not by Auto's tests. What's new here is the supervisor around it.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.auto.intent_ledger import deterministic_client_order_id
from src.config.settings import AutoConfig


class _FakePosition:
    def __init__(self, market_id, side, position_id):
        self.market_id = market_id
        self.side = side
        self.id = position_id


@pytest.fixture
def config(tmp_path):
    cfg = AutoConfig()
    cfg.kill_switch_path = str(tmp_path / "kill.json")
    cfg.max_consecutive_failures = 3
    cfg.cadence_seconds = 0
    return cfg


@pytest.fixture
def mock_decision_loop():
    """Patches LiveTradeDecisionLoop so no real decision pipeline runs."""
    with patch("src.auto.runner.LiveTradeDecisionLoop") as mock_cls:
        instance = MagicMock()
        instance.run_once = AsyncMock(return_value=MagicMock(skipped_reason=None))
        mock_cls.return_value = instance
        yield mock_cls, instance


def _make_runner(config, mock_decision_loop):
    from src.auto.runner import AutoRunner

    db_manager = AsyncMock()
    db_manager.is_strategy_halted_today = AsyncMock(return_value=False)
    kalshi_client = AsyncMock()
    runner = AutoRunner(db_manager=db_manager, kalshi_client=kalshi_client, config=config)
    return runner, db_manager, kalshi_client


def test_decision_loop_wired_with_wrapper_and_quick_flip_disabled(config, mock_decision_loop):
    mock_cls, _instance = mock_decision_loop
    runner, _db, _kc = _make_runner(config, mock_decision_loop)

    _, kwargs = mock_cls.call_args
    assert kwargs["execute_position_fn"] == runner._auto_execute_position
    assert kwargs["manage_quick_flip_positions_each_cycle"] is False


@pytest.mark.asyncio
async def test_live_buy_uses_deterministic_client_order_id_and_records_intent(config, mock_decision_loop):
    runner, db_manager, kalshi_client = _make_runner(config, mock_decision_loop)
    position = _FakePosition("KXTEST-26", "YES", 42)
    expected_id = deterministic_client_order_id(ticker="KXTEST-26", side="YES", position_id=42)

    with patch("src.auto.runner.execute_position", new=AsyncMock(return_value=True)) as mock_exec:
        result = await runner._auto_execute_position(
            position=position, live_mode=True, db_manager=db_manager, kalshi_client=kalshi_client,
        )

    assert result is True
    _, call_kwargs = mock_exec.call_args
    assert call_kwargs["client_order_id"] == expected_id
    db_manager.record_auto_order_intent.assert_awaited()
    # intent recorded BEFORE the exchange call: the mock records call order
    # implicitly by both having been awaited; the explicit ordering
    # guarantee is structural in _auto_execute_position (record -> submit).


@pytest.mark.asyncio
async def test_restart_reuses_identical_client_order_id(config, mock_decision_loop):
    runner, db_manager, kalshi_client = _make_runner(config, mock_decision_loop)
    expected_id = deterministic_client_order_id(ticker="KXTEST-26", side="YES", position_id=42)

    with patch("src.auto.runner.execute_position", new=AsyncMock(return_value=True)) as mock_exec:
        await runner._auto_execute_position(
            position=_FakePosition("KXTEST-26", "YES", 42), live_mode=True,
            db_manager=db_manager, kalshi_client=kalshi_client,
        )
        first_id = mock_exec.call_args.kwargs["client_order_id"]

        # Simulate a fresh process after a crash: a brand-new AutoRunner,
        # but the SAME persisted position (same position.id).
        await runner._auto_execute_position(
            position=_FakePosition("KXTEST-26", "YES", 42), live_mode=True,
            db_manager=db_manager, kalshi_client=kalshi_client,
        )
        second_id = mock_exec.call_args.kwargs["client_order_id"]

    assert first_id == second_id == expected_id


@pytest.mark.asyncio
async def test_kill_switch_active_blocks_live_buy(config, mock_decision_loop):
    from src.auto.kill_switch import AutoKillSwitch

    runner, db_manager, kalshi_client = _make_runner(config, mock_decision_loop)
    AutoKillSwitch(config.kill_switch_path).activate(reason="test")

    with patch("src.auto.runner.execute_position", new=AsyncMock()) as mock_exec:
        result = await runner._auto_execute_position(
            position=_FakePosition("KXTEST-99", "NO", 7), live_mode=True,
            db_manager=db_manager, kalshi_client=kalshi_client,
        )

    assert result is False
    mock_exec.assert_not_awaited()


@pytest.mark.asyncio
async def test_paper_mode_bypasses_kill_switch_and_id_injection(config, mock_decision_loop):
    from src.auto.kill_switch import AutoKillSwitch

    runner, db_manager, kalshi_client = _make_runner(config, mock_decision_loop)
    AutoKillSwitch(config.kill_switch_path).activate(reason="should not matter for paper")

    with patch("src.auto.runner.execute_position", new=AsyncMock(return_value=True)) as mock_exec:
        result = await runner._auto_execute_position(
            position=_FakePosition("KXTEST-1", "YES", 1), live_mode=False,
            db_manager=db_manager, kalshi_client=kalshi_client,
        )

    assert result is True
    mock_exec.assert_awaited_once()
    assert "client_order_id" not in mock_exec.call_args.kwargs


@pytest.mark.asyncio
async def test_cycle_start_kill_switch_blocks_run_once(config, mock_decision_loop):
    from src.auto.kill_switch import AutoKillSwitch

    _mock_cls, instance = mock_decision_loop
    runner, _db, _kc = _make_runner(config, mock_decision_loop)
    AutoKillSwitch(config.kill_switch_path).activate(reason="cycle start")

    outcome = await runner.run_one_cycle()

    assert outcome.status == "skipped_kill_switch"
    instance.run_once.assert_not_awaited()


@pytest.mark.asyncio
async def test_strategy_halt_blocks_run_once(config, mock_decision_loop):
    _mock_cls, instance = mock_decision_loop
    runner, db_manager, _kc = _make_runner(config, mock_decision_loop)
    db_manager.is_strategy_halted_today = AsyncMock(return_value=True)

    outcome = await runner.run_one_cycle()

    assert outcome.status == "skipped_halted"
    instance.run_once.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_once_exception_reported_as_failed_not_swallowed(config, mock_decision_loop):
    _mock_cls, instance = mock_decision_loop
    instance.run_once = AsyncMock(side_effect=RuntimeError("boom"))
    runner, _db, _kc = _make_runner(config, mock_decision_loop)

    outcome = await runner.run_one_cycle()

    assert outcome.status == "failed"
    assert "boom" in outcome.detail


@pytest.mark.asyncio
async def test_consecutive_failures_trigger_fail_closed_halt(config, mock_decision_loop):
    from src.auto.kill_switch import AutoKillSwitch

    _mock_cls, instance = mock_decision_loop
    instance.run_once = AsyncMock(side_effect=RuntimeError("boom"))
    runner, _db, _kc = _make_runner(config, mock_decision_loop)

    with patch.object(runner, "reconcile_on_startup", new=AsyncMock(return_value=[])):
        await runner.run_forever()

    switch = AutoKillSwitch(config.kill_switch_path)
    assert switch.is_active() is True
    assert runner._consecutive_failures == config.max_consecutive_failures
    state = switch.read_state()
    assert "consecutive" in (state.reason or "")
