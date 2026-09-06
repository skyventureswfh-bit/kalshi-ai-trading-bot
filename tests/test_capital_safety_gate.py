from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import src.clients.kalshi_client as kalshi_client_module
from src.clients.kalshi_client import KalshiAPIError, KalshiClient
from src.utils.cash_reserves import CashReservesManager


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
    instance = KalshiClient(api_key="test-key", base_url="https://demo-api.kalshi.co")
    yield instance


@pytest.mark.asyncio
async def test_live_buy_gate_blocks_exchange_post_when_reserve_check_rejects(client, monkeypatch):
    request_mock = AsyncMock(return_value={"order": {"order_id": "should-not-send"}})
    monkeypatch.setattr(client, "_make_authenticated_request", request_mock)

    reserve_check = AsyncMock(
        return_value=SimpleNamespace(
            can_trade=False,
            reason="DAILY LOSS HALT",
        )
    )
    monkeypatch.setattr(CashReservesManager, "check_cash_reserves", reserve_check)

    with pytest.raises(KalshiAPIError, match="capital safety gate"):
        await client.place_order(
            ticker="TEST-1",
            client_order_id="abc",
            side="yes",
            action="buy",
            count=2,
            type_="limit",
            yes_price_dollars="0.40",
        )

    reserve_check.assert_awaited_once_with(proposed_trade_value=0.8)
    request_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_live_buy_gate_allows_exchange_post_when_safety_check_passes(client, monkeypatch):
    request_mock = AsyncMock(return_value={"order": {"order_id": "sent"}})
    monkeypatch.setattr(client, "_make_authenticated_request", request_mock)

    reserve_check = AsyncMock(
        return_value=SimpleNamespace(
            can_trade=True,
            reason="Safety checks passed",
        )
    )
    monkeypatch.setattr(CashReservesManager, "check_cash_reserves", reserve_check)

    response = await client.place_order(
        ticker="TEST-2",
        client_order_id="def",
        side="no",
        action="buy",
        count=3,
        type_="limit",
        no_price_dollars="0.25",
    )

    assert response["order"]["order_id"] == "sent"
    reserve_check.assert_awaited_once_with(proposed_trade_value=0.75)
    request_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_sell_order_bypasses_new_money_gate_so_exits_are_never_blocked(client, monkeypatch):
    request_mock = AsyncMock(return_value={"order": {"order_id": "sell-sent"}})
    monkeypatch.setattr(client, "_make_authenticated_request", request_mock)

    reserve_check = AsyncMock(side_effect=AssertionError("sell must not call reserve gate"))
    monkeypatch.setattr(CashReservesManager, "check_cash_reserves", reserve_check)

    response = await client.place_order(
        ticker="TEST-3",
        client_order_id="ghi",
        side="yes",
        action="sell",
        count=1,
        type_="limit",
        yes_price_dollars="0.55",
    )

    assert response["order"]["order_id"] == "sell-sent"
    reserve_check.assert_not_awaited()
    request_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_buy_max_cost_is_included_in_final_safety_exposure(client, monkeypatch):
    request_mock = AsyncMock(return_value={"order": {"order_id": "sent"}})
    monkeypatch.setattr(client, "_make_authenticated_request", request_mock)

    reserve_check = AsyncMock(
        return_value=SimpleNamespace(
            can_trade=True,
            reason="Safety checks passed",
        )
    )
    monkeypatch.setattr(CashReservesManager, "check_cash_reserves", reserve_check)

    await client.place_order(
        ticker="TEST-4",
        client_order_id="jkl",
        side="yes",
        action="buy",
        count=1,
        type_="limit",
        yes_price_dollars="0.20",
        buy_max_cost=75,
    )

    reserve_check.assert_awaited_once_with(proposed_trade_value=0.75)
    request_mock.assert_awaited_once()
