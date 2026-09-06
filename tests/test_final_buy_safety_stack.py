from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import src.clients.kalshi_client as kalshi_client_module
import src.utils.cash_reserves as cash_reserves_module
from src.clients.kalshi_client import KalshiAPIError, KalshiClient
from src.utils.account_snapshot import AccountSafetySnapshot
from src.utils.position_limits import PositionLimitResult


class DummyAsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def request(self, *args, **kwargs):
        raise AssertionError("network request should be mocked in this test")

    async def aclose(self):
        return None


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(kalshi_client_module.httpx, "AsyncClient", DummyAsyncClient)
    return KalshiClient(api_key="test-key", base_url="https://demo-api.kalshi.co")


def _safe_daily_status():
    return {
        "daily_realized_pnl": 0.0,
        "daily_realized_loss": 0.0,
        "daily_loss_limit": 3.0,
        "daily_loss_cap_pct": 3.0,
        "daily_loss_cap_dollars": 0.0,
        "daily_loss_halt": False,
    }


def _position_result(can_trade=True, reason="Position limits satisfied"):
    return PositionLimitResult(
        can_trade=can_trade,
        reason=reason,
        current_positions=2 if can_trade else 15,
        max_positions=15,
        current_portfolio_usage=30.0,
        max_position_size=5.0,
        recommended_actions=[] if can_trade else ["Reduce exposure before adding risk"],
    )


def _patch_combined_safety(monkeypatch, *, cash=70.0, portfolio=100.0, drawdown_ok=True, positions_ok=True):
    snapshot = AccountSafetySnapshot(available_cash=cash, portfolio_value=portfolio)
    snapshot_getter = AsyncMock(return_value=snapshot)
    monkeypatch.setattr(cash_reserves_module, "get_account_safety_snapshot", snapshot_getter)
    monkeypatch.setattr(
        cash_reserves_module.CashReservesManager,
        "_daily_loss_status",
        AsyncMock(return_value=_safe_daily_status()),
    )
    monkeypatch.setattr(
        cash_reserves_module,
        "check_drawdown_guard",
        AsyncMock(
            return_value=SimpleNamespace(
                can_trade=drawdown_ok,
                reason="Drawdown within limit" if drawdown_ok else "DRAWDOWN HALT: test",
            )
        ),
    )
    monkeypatch.setattr(
        cash_reserves_module.PositionLimitsManager,
        "check_position_limits",
        AsyncMock(
            return_value=_position_result(
                can_trade=positions_ok,
                reason="Position limits satisfied" if positions_ok else "Position count 15 at/above limit 15",
            )
        ),
    )
    return snapshot_getter


@pytest.mark.asyncio
async def test_final_buy_gate_runs_full_stack_once_and_posts_when_all_checks_pass(client, monkeypatch):
    snapshot_getter = _patch_combined_safety(monkeypatch)
    request_mock = AsyncMock(return_value={"order": {"order_id": "sent"}})
    monkeypatch.setattr(client, "_make_authenticated_request", request_mock)

    response = await client.place_order(
        ticker="STACK-OK",
        client_order_id="stack-ok",
        side="yes",
        action="buy",
        count=2,
        type_="limit",
        yes_price_dollars="0.40",
    )

    assert response["order"]["order_id"] == "sent"
    snapshot_getter.assert_awaited_once_with(client)
    request_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_final_buy_gate_blocks_exchange_post_on_drawdown_halt(client, monkeypatch):
    _patch_combined_safety(monkeypatch, drawdown_ok=False)
    request_mock = AsyncMock(return_value={"order": {"order_id": "must-not-send"}})
    monkeypatch.setattr(client, "_make_authenticated_request", request_mock)

    with pytest.raises(KalshiAPIError, match="capital safety gate"):
        await client.place_order(
            ticker="STACK-DD",
            client_order_id="stack-dd",
            side="yes",
            action="buy",
            count=1,
            type_="limit",
            yes_price_dollars="0.20",
        )

    request_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_final_buy_gate_blocks_exchange_post_on_position_limit(client, monkeypatch):
    _patch_combined_safety(monkeypatch, positions_ok=False)
    request_mock = AsyncMock(return_value={"order": {"order_id": "must-not-send"}})
    monkeypatch.setattr(client, "_make_authenticated_request", request_mock)

    with pytest.raises(KalshiAPIError, match="capital safety gate"):
        await client.place_order(
            ticker="STACK-POS",
            client_order_id="stack-pos",
            side="yes",
            action="buy",
            count=1,
            type_="limit",
            yes_price_dollars="0.20",
        )

    request_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_final_buy_gate_blocks_trade_that_breaks_protected_reserve(client, monkeypatch):
    _patch_combined_safety(monkeypatch, cash=34.0, portfolio=100.0)
    request_mock = AsyncMock(return_value={"order": {"order_id": "must-not-send"}})
    monkeypatch.setattr(client, "_make_authenticated_request", request_mock)

    with pytest.raises(KalshiAPIError, match="capital safety gate"):
        await client.place_order(
            ticker="STACK-RESERVE",
            client_order_id="stack-reserve",
            side="yes",
            action="buy",
            count=2,
            type_="limit",
            yes_price_dollars="0.75",
        )

    request_mock.assert_not_awaited()
