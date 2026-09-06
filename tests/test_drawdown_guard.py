from types import SimpleNamespace

import pytest

import src.utils.drawdown_guard as drawdown_module
from src.utils.drawdown_guard import DrawdownGuard


class DummyKalshiClient:
    def __init__(self, values):
        self.values = iter(values)

    async def get_balance(self):
        value = next(self.values)
        return {"balance": int(value * 100), "portfolio_value": 0}

    async def get_positions(self):
        return {"event_positions": []}


@pytest.mark.asyncio
async def test_drawdown_guard_tracks_high_watermark_and_halts(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_DRAWDOWN_PCT", "15")
    monkeypatch.setattr(
        drawdown_module,
        "settings",
        SimpleNamespace(trading=SimpleNamespace(live_trading_enabled=False)),
    )
    db_manager = SimpleNamespace(db_path=str(tmp_path / "drawdown.db"))
    client = DummyKalshiClient([100.0, 120.0, 100.0])
    guard = DrawdownGuard(db_manager, client)

    first = await guard.check()
    assert first.can_trade is True
    assert first.high_watermark == pytest.approx(100.0)
    assert first.drawdown_pct == pytest.approx(0.0)

    second = await guard.check()
    assert second.can_trade is True
    assert second.high_watermark == pytest.approx(120.0)
    assert second.drawdown_pct == pytest.approx(0.0)

    third = await guard.check()
    assert third.can_trade is False
    assert third.high_watermark == pytest.approx(120.0)
    assert third.drawdown_pct == pytest.approx((20.0 / 120.0) * 100.0)
    assert "DRAWDOWN HALT" in third.reason


@pytest.mark.asyncio
async def test_drawdown_guard_fails_closed_when_portfolio_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_DRAWDOWN_PCT", "15")
    monkeypatch.setattr(
        drawdown_module,
        "settings",
        SimpleNamespace(trading=SimpleNamespace(live_trading_enabled=True)),
    )
    db_manager = SimpleNamespace(db_path=str(tmp_path / "drawdown.db"))
    client = DummyKalshiClient([0.0])

    status = await DrawdownGuard(db_manager, client).check()

    assert status.can_trade is False
    assert "failed closed" in status.reason.lower()
