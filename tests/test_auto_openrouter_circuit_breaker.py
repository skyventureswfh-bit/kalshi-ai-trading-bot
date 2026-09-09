from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.auto.kill_switch import AutoKillSwitch
from src.auto.runner import AutoRunner
from src.config.settings import AutoConfig


def _runner(tmp_path):
    config = AutoConfig()
    config.kill_switch_path = str(tmp_path / "kill.json")
    config.cadence_seconds = 1
    config.max_consecutive_failures = 3

    db = AsyncMock()
    db.is_strategy_halted_today = AsyncMock(return_value=False)
    db.list_unresolved_auto_order_intents = AsyncMock(return_value=[])
    kalshi = AsyncMock()
    loop = MagicMock()
    loop.run_once = AsyncMock(return_value=MagicMock(skipped_reason="ai unavailable"))

    runner = AutoRunner(
        db_manager=db,
        kalshi_client=kalshi,
        config=config,
        decision_loop_factory=lambda **kwargs: loop,
    )
    return runner, loop, config


@pytest.mark.asyncio
async def test_credit_latch_blocks_next_cycle_before_decision_loop(tmp_path):
    runner, loop, _config = _runner(tmp_path)

    with patch("src.auto.runner.prompt_credit_exhausted", return_value=True):
        outcome = await runner.run_one_cycle()

    assert outcome.status == "credit_exhausted"
    loop.run_once.assert_not_awaited()


@pytest.mark.asyncio
async def test_credit_exhaustion_halts_auto_and_activates_kill_switch(tmp_path):
    runner, _loop, config = _runner(tmp_path)

    with patch("src.auto.runner.prompt_credit_exhausted", return_value=True):
        await runner.run_forever()

    state = AutoKillSwitch(config.kill_switch_path).read_state()
    assert state.active is True
    assert "OpenRouter" in (state.reason or "")
