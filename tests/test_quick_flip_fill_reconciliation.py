from datetime import datetime, timezone

from src.strategies.quick_flip_scalping import QuickFlipScalpingStrategy
from src.utils.database import Position


def _strategy():
    return QuickFlipScalpingStrategy(
        db_manager=object(),
        kalshi_client=object(),
        xai_client=None,
    )


def _position():
    return Position(
        market_id="TEST-MARKET",
        side="YES",
        quantity=4,
        entry_price=0.25,
        timestamp=datetime.now(timezone.utc),
        rationale="test",
        live=True,
        strategy="quick_flip_scalping",
    )


def test_fill_reconciliation_dedupes_live_and_history_overlap():
    strategy = _strategy()
    position = _position()
    fill = {
        "fill_id": "fill-1",
        "ticker": "TEST-MARKET",
        "side": "yes",
        "action": "sell",
        "count": 2,
        "yes_price_dollars": "0.31",
        "ts": 1_800_000_000,
    }

    unique = strategy._dedupe_position_fills([fill, dict(fill)], position)

    assert unique == [fill]


def test_fill_reconciliation_rejects_other_side_and_other_ticker():
    strategy = _strategy()
    position = _position()
    matching = {
        "fill_id": "matching",
        "ticker": "TEST-MARKET",
        "side": "yes",
        "action": "sell",
        "count": 1,
    }
    other_side = {
        "fill_id": "wrong-side",
        "ticker": "TEST-MARKET",
        "side": "no",
        "action": "sell",
        "count": 4,
    }
    other_ticker = {
        "fill_id": "wrong-ticker",
        "ticker": "SOME-OTHER-MARKET",
        "side": "yes",
        "action": "sell",
        "count": 4,
    }

    unique = strategy._dedupe_position_fills(
        [other_side, matching, other_ticker],
        position,
    )

    assert unique == [matching]


def test_fill_reconciliation_dedupes_when_fill_id_is_missing():
    strategy = _strategy()
    position = _position()
    fill = {
        "order_id": "order-1",
        "client_order_id": "client-1",
        "ticker": "TEST-MARKET",
        "side": "yes",
        "action": "sell",
        "count": 2,
        "yes_price_dollars": "0.31",
        "ts": 1_800_000_000,
    }

    unique = strategy._dedupe_position_fills([fill, dict(fill)], position)

    assert unique == [fill]
