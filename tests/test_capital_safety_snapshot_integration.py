from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import src.utils.cash_reserves as cash_reserves_module
from src.utils.account_snapshot import AccountSafetySnapshot
from src.utils.cash_reserves import CashReservesManager
from src.utils.position_limits import PositionLimitResult


@pytest.mark.asyncio
async def test_cash_reserve_gate_reuses_one_snapshot_for_drawdown_and_position_limits(monkeypatch):
    client = AsyncMock()
    db_manager = SimpleNamespace(db_path=":memory:")
    manager = CashReservesManager(db_manager, client)

    snapshot = AccountSafetySnapshot(available_cash=70.0, portfolio_value=100.0)
    snapshot_getter = AsyncMock(return_value=snapshot)
    monkeypatch.setattr(
        cash_reserves_module,
        "get_account_safety_snapshot",
        snapshot_getter,
    )

    manager._daily_loss_status = AsyncMock(
        return_value={
            "daily_realized_pnl": 0.0,
            "daily_realized_loss": 0.0,
            "daily_loss_limit": 3.0,
            "daily_loss_cap_pct": 3.0,
            "daily_loss_cap_dollars": 0.0,
            "daily_loss_halt": False,
        }
    )

    drawdown_check = AsyncMock(
        return_value=SimpleNamespace(
            can_trade=True,
            reason="Drawdown within limit",
        )
    )
    monkeypatch.setattr(cash_reserves_module, "check_drawdown_guard", drawdown_check)

    position_check = AsyncMock(
        return_value=PositionLimitResult(
            can_trade=True,
            reason="Position limits satisfied",
            current_positions=2,
            max_positions=15,
            current_portfolio_usage=30.0,
            max_position_size=5.0,
            recommended_actions=[],
        )
    )
    monkeypatch.setattr(
        cash_reserves_module.PositionLimitsManager,
        "check_position_limits",
        position_check,
    )

    result = await manager.check_cash_reserves(proposed_trade_value=2.0)

    assert result.can_trade is True
    snapshot_getter.assert_awaited_once_with(client)
    drawdown_check.assert_awaited_once_with(db_manager, client, snapshot=snapshot)
    position_check.assert_awaited_once_with(
        proposed_position_size=2.0,
        portfolio_value=100.0,
        available_cash=70.0,
        snapshot=snapshot,
    )
    client.get_balance.assert_not_awaited()
    client.get_positions.assert_not_awaited()


@pytest.mark.asyncio
async def test_position_limit_failure_blocks_capital_gate_with_shared_snapshot(monkeypatch):
    client = AsyncMock()
    db_manager = SimpleNamespace(db_path=":memory:")
    manager = CashReservesManager(db_manager, client)

    snapshot = AccountSafetySnapshot(available_cash=70.0, portfolio_value=100.0)
    monkeypatch.setattr(
        cash_reserves_module,
        "get_account_safety_snapshot",
        AsyncMock(return_value=snapshot),
    )
    manager._daily_loss_status = AsyncMock(
        return_value={
            "daily_realized_pnl": 0.0,
            "daily_realized_loss": 0.0,
            "daily_loss_limit": 3.0,
            "daily_loss_cap_pct": 3.0,
            "daily_loss_cap_dollars": 0.0,
            "daily_loss_halt": False,
        }
    )
    monkeypatch.setattr(
        cash_reserves_module,
        "check_drawdown_guard",
        AsyncMock(return_value=SimpleNamespace(can_trade=True, reason="ok")),
    )
    monkeypatch.setattr(
        cash_reserves_module.PositionLimitsManager,
        "check_position_limits",
        AsyncMock(
            return_value=PositionLimitResult(
                can_trade=False,
                reason="Position count 15 at/above limit 15",
                current_positions=15,
                max_positions=15,
                current_portfolio_usage=30.0,
                max_position_size=5.0,
                recommended_actions=["Close a confirmed position before adding another"],
            )
        ),
    )

    result = await manager.check_cash_reserves(proposed_trade_value=2.0)

    assert result.can_trade is False
    assert "POSITION LIMIT HALT" in result.reason
    client.get_balance.assert_not_awaited()
    client.get_positions.assert_not_awaited()
