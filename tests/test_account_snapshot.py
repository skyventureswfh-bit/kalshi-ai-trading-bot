from unittest.mock import AsyncMock

import pytest

from src.utils.account_snapshot import (
    AccountSafetySnapshot,
    get_account_safety_snapshot,
)


@pytest.mark.asyncio
async def test_snapshot_uses_balance_marked_value_without_positions_call():
    client = AsyncMock()
    client.get_balance.return_value = {
        "balance": 5000,
        "portfolio_value": 2500,
    }

    snapshot = await get_account_safety_snapshot(client)

    assert isinstance(snapshot, AccountSafetySnapshot)
    assert snapshot.available_cash == pytest.approx(50.0)
    assert snapshot.portfolio_value == pytest.approx(75.0)
    client.get_balance.assert_awaited_once()
    client.get_positions.assert_not_awaited()


@pytest.mark.asyncio
async def test_snapshot_falls_back_to_positions_once_when_marked_value_missing():
    client = AsyncMock()
    client.get_balance.return_value = {"balance": 5000, "portfolio_value": 0}
    client.get_positions.return_value = {
        "event_positions": [
            {"market_exposure": 1000},
            {"market_exposure": 500},
        ]
    }

    snapshot = await get_account_safety_snapshot(client)

    assert snapshot.available_cash == pytest.approx(50.0)
    assert snapshot.portfolio_value >= snapshot.available_cash
    client.get_balance.assert_awaited_once()
    client.get_positions.assert_awaited_once()


@pytest.mark.asyncio
async def test_snapshot_fails_closed_on_zero_portfolio_value():
    client = AsyncMock()
    client.get_balance.return_value = {"balance": 0, "portfolio_value": 0}
    client.get_positions.return_value = {"event_positions": []}

    with pytest.raises(ValueError, match="portfolio value is zero or negative"):
        await get_account_safety_snapshot(client)
