from src.auto.market_discovery import discover_open_btc_market


def test_discovers_nearest_usable_open_market():
    payload = {"markets": [
        {"ticker": "LATER", "floor_strike": 102,
         "expected_expiration_time": "2026-09-11T01:30:00Z"},
        {"ticker": "NEXT", "floor_strike": 100,
         "expected_expiration_time": "2026-09-11T01:15:00Z"},
        {"ticker": "OLD", "floor_strike": 99,
         "expected_expiration_time": "2026-09-11T00:45:00Z"},
    ]}
    result = discover_open_btc_market(
        now_epoch=1789088400,
        fetch_json=lambda _url: payload,
    )
    assert result.ticker == "NEXT"
    assert result.target == 100

