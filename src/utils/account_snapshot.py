"""Shared Kalshi account snapshot for safety checks.

Safety modules should reason about one consistent account state per decision
instead of independently re-fetching balance and positions several times.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.clients.kalshi_client import KalshiClient
from src.utils.kalshi_normalization import (
    get_balance_dollars,
    get_portfolio_value_dollars,
    get_position_exposure_dollars,
)


@dataclass(frozen=True)
class AccountSafetySnapshot:
    available_cash: float
    portfolio_value: float


async def get_account_safety_snapshot(
    kalshi_client: KalshiClient,
) -> AccountSafetySnapshot:
    """Fetch one consistent cash/portfolio snapshot for a safety decision."""
    balance = await kalshi_client.get_balance()
    available_cash = get_balance_dollars(balance)
    marked_value = get_portfolio_value_dollars(balance)

    if marked_value > 0:
        portfolio_value = available_cash + marked_value
    else:
        positions_response = await kalshi_client.get_positions()
        positions = (
            positions_response.get("event_positions", [])
            if isinstance(positions_response, dict)
            else []
        )
        position_value = sum(
            get_position_exposure_dollars(position)
            for position in positions
            if isinstance(position, dict)
        )
        portfolio_value = available_cash + position_value

    if portfolio_value <= 0:
        raise ValueError("portfolio value is zero or negative")

    return AccountSafetySnapshot(
        available_cash=float(available_cash),
        portfolio_value=float(portfolio_value),
    )
