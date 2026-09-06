from unittest.mock import AsyncMock

import pytest

import src.utils.position_limits as position_limits_module


@pytest.mark.asyncio
async def test_enforcement_helper_allows_only_below_limit(monkeypatch):
    db_manager = AsyncMock()
    kalshi_client = AsyncMock()

    manager = AsyncMock()
    manager.max_positions = 15
    manager._get_position_count.return_value = 14

    monkeypatch.setattr(
        position_limits_module,
        "PositionLimitsManager",
        lambda db, client: manager,
    )

    assert await position_limits_module.enforce_limits_if_needed(db_manager, kalshi_client) is True
    manager.enforce_position_limits.assert_not_awaited()


@pytest.mark.asyncio
async def test_enforcement_helper_blocks_at_exact_limit(monkeypatch):
    db_manager = AsyncMock()
    kalshi_client = AsyncMock()

    manager = AsyncMock()
    manager.max_positions = 15
    manager._get_position_count.return_value = 15

    monkeypatch.setattr(
        position_limits_module,
        "PositionLimitsManager",
        lambda db, client: manager,
    )

    assert await position_limits_module.enforce_limits_if_needed(db_manager, kalshi_client) is False
    manager.enforce_position_limits.assert_not_awaited()


@pytest.mark.asyncio
async def test_enforcement_helper_stays_blocked_until_confirmed_exits_reduce_count(monkeypatch):
    db_manager = AsyncMock()
    kalshi_client = AsyncMock()

    manager = AsyncMock()
    manager.max_positions = 15
    manager._get_position_count.return_value = 17
    manager.enforce_position_limits.return_value = {
        "action": "exit_required",
        "positions_to_reduce": 2,
    }

    monkeypatch.setattr(
        position_limits_module,
        "PositionLimitsManager",
        lambda db, client: manager,
    )

    assert await position_limits_module.enforce_limits_if_needed(db_manager, kalshi_client) is False
    manager.enforce_position_limits.assert_awaited_once()
