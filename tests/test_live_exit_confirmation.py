from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from src.utils.database import Position
from src.utils.live_exit import execute_confirmed_live_exit


def _position(quantity: float = 5) -> Position:
    return Position(
        market_id="KXTEST-YES",
        side="YES",
        quantity=quantity,
        entry_price=0.40,
        timestamp=datetime.now(),
        rationale="test",
        live=True,
        strategy="test",
    )


def _market(best_bid: float = 0.55):
    # Normalization helper can read yes_bid_dollars directly.
    return {"yes_bid_dollars": best_bid}


@pytest.mark.asyncio
async def test_terminal_status_without_fill_quantity_fails_closed():
    client = AsyncMock()
    client.place_order.return_value = {
        "order": {
            "order_id": "order-1",
            "status": "filled",
            # Deliberately no filled_count / fill_count fields.
        }
    }

    result = await execute_confirmed_live_exit(
        position=_position(5),
        market_info=_market(),
        kalshi_client=client,
    )

    assert result.filled is False
    assert result.filled_quantity == 0
    assert "full requested exit quantity" in result.reason


@pytest.mark.asyncio
async def test_partial_fill_does_not_confirm_full_exit():
    client = AsyncMock()
    client.place_order.return_value = {
        "order": {
            "order_id": "order-2",
            "status": "filled",
            "filled_count": 3,
            "yes_price_dollars": 0.55,
        }
    }

    result = await execute_confirmed_live_exit(
        position=_position(5),
        market_info=_market(),
        kalshi_client=client,
    )

    assert result.filled is False
    assert result.filled_quantity == 3


@pytest.mark.asyncio
async def test_full_reported_fill_confirms_exit():
    client = AsyncMock()
    client.place_order.return_value = {
        "order": {
            "order_id": "order-3",
            "status": "filled",
            "filled_count": 5,
            "yes_price_dollars": 0.55,
        }
    }

    result = await execute_confirmed_live_exit(
        position=_position(5),
        market_info=_market(),
        kalshi_client=client,
    )

    assert result.filled is True
    assert result.filled_quantity == 5
