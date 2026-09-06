from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.utils.account_snapshot import AccountSafetySnapshot
from src.utils.cash_reserves import CashReservesManager


@pytest.mark.asyncio
async def test_conflicting_portfolio_value_fails_closed_before_other_safety_checks():
    manager = CashReservesManager(
        db_manager=SimpleNamespace(db_path=":memory:"),
        kalshi_client=AsyncMock(),
    )
    snapshot = AccountSafetySnapshot(
        available_cash=70.0,
        portfolio_value=100.0,
        marked_position_value=30.0,
    )

    with patch.object(manager, "_daily_loss_status", new=AsyncMock()) as daily_check:
        result = await manager.check_cash_reserves(
            proposed_trade_value=1.0,
            portfolio_value=125.0,
            snapshot=snapshot,
        )

    assert result.can_trade is False
    assert result.emergency_status is True
    assert "conflicts with shared account snapshot" in result.reason
    daily_check.assert_not_awaited()


@pytest.mark.asyncio
async def test_matching_explicit_portfolio_value_uses_shared_snapshot():
    manager = CashReservesManager(
        db_manager=SimpleNamespace(db_path=":memory:"),
        kalshi_client=AsyncMock(),
    )
    snapshot = AccountSafetySnapshot(
        available_cash=70.0,
        portfolio_value=100.0,
        marked_position_value=30.0,
    )

    daily_status = {
        "daily_realized_pnl": 0.0,
        "daily_realized_loss": 0.0,
        "daily_loss_limit": 3.0,
        "daily_loss_cap_pct": 3.0,
        "daily_loss_cap_dollars": 0.0,
        "daily_loss_halt": False,
    }
    drawdown = SimpleNamespace(
        can_trade=True,
        reason="drawdown safe",
        drawdown_pct=0.0,
        limit_pct=15.0,
        high_watermark=100.0,
    )
    position_limits = SimpleNamespace(
        can_trade=True,
        reason="position limits safe",
        recommended_actions=[],
    )

    with (
        patch.object(manager, "_daily_loss_status", new=AsyncMock(return_value=daily_status)),
        patch("src.utils.cash_reserves.check_drawdown_guard", new=AsyncMock(return_value=drawdown)),
        patch("src.utils.cash_reserves.PositionLimitsManager.check_position_limits", new=AsyncMock(return_value=position_limits)),
    ):
        result = await manager.check_cash_reserves(
            proposed_trade_value=1.0,
            portfolio_value=100.0,
            snapshot=snapshot,
        )

    assert result.can_trade is True
    assert result.portfolio_value == 100.0
    assert result.current_cash == 70.0
