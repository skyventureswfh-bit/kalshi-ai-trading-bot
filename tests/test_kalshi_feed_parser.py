import pytest

from src.auto.kalshi_feed_parser import LiveOrderBook, parse_cfbenchmarks_value, parse_orderbook_snapshot
from src.auto.market_recorder import UnsafeObservation


def test_parses_authoritative_final_minute_average():
    frame = {
        "type": "cfbenchmarks_value", "seq": 42,
        "msg": {"index_id": "BRTI", "received_at": 1710000000141,
                "data": '{"time":1710000000123,"value":"68000.12"}',
                "last_60s_windowed_average_15min": {"value": "68000.23", "window_size": 14}},
    }
    item = parse_cfbenchmarks_value(frame, ticker="KXBTC15M-TEST", target=68001, seconds_remaining=46)
    assert item.brti == 68000.12
    assert item.official_final_minute_average == 68000.23
    assert item.official_average_window_size == 14
    assert item.event_epoch == 1710000000.123
    assert item.upstream_received_epoch == 1710000000.141


def test_orderbook_converts_opposite_bid_to_ask():
    frame = {"type": "orderbook_snapshot", "seq": 2,
             "msg": {"market_ticker": "KXBTC15M-TEST",
                     "yes_dollars_fp": [["0.4800", "10"]],
                     "no_dollars_fp": [["0.4900", "8"]]}}
    item = parse_orderbook_snapshot(frame, target=68001, seconds_remaining=46,
                                    received_epoch=1710000000.2)
    assert item.yes_bid_cents == 48
    assert item.yes_ask_cents == 51
    assert item.no_bid_cents == 49
    assert item.no_ask_cents == 52


def test_live_book_applies_delta_and_rejects_sequence_gap():
    book = LiveOrderBook()
    book.load_snapshot({"type": "orderbook_snapshot", "sid": 2, "seq": 2,
                        "msg": {"market_ticker": "KXBTC15M-TEST",
                                "yes_dollars_fp": [["0.4800", "10"]],
                                "no_dollars_fp": [["0.4900", "8"]]}})
    event_time = book.apply_delta({"type": "orderbook_delta", "sid": 2, "seq": 3,
                                   "msg": {"market_ticker": "KXBTC15M-TEST",
                                           "price_dollars": "0.5000", "delta_fp": "4",
                                           "side": "yes", "ts_ms": 1710000000300}})
    assert event_time == 1710000000.3
    assert book.top() == (50.0, 51.0, 49.0, 50.0)
    with pytest.raises(UnsafeObservation, match="sequence gap"):
        book.apply_delta({"type": "orderbook_delta", "sid": 2, "seq": 5,
                          "msg": {"market_ticker": "KXBTC15M-TEST",
                                  "price_dollars": "0.5100", "delta_fp": "1",
                                  "side": "yes", "ts_ms": 1710000000400}})


def test_live_book_uses_exact_fixed_point_quantities():
    book = LiveOrderBook()
    book.load_snapshot({"type": "orderbook_snapshot", "sid": 2, "seq": 2,
                        "msg": {"market_ticker": "KXBTC15M-TEST",
                                "yes_dollars_fp": [["0.4800", "0.30"]],
                                "no_dollars_fp": []}})
    book.apply_delta({"type": "orderbook_delta", "sid": 2, "seq": 3,
                      "msg": {"market_ticker": "KXBTC15M-TEST",
                              "price_dollars": "0.4800", "delta_fp": "-0.10",
                              "side": "yes", "ts_ms": 1710000000300}})
    book.apply_delta({"type": "orderbook_delta", "sid": 2, "seq": 4,
                      "msg": {"market_ticker": "KXBTC15M-TEST",
                              "price_dollars": "0.4800", "delta_fp": "-0.20",
                              "side": "yes", "ts_ms": 1710000000400}})
    assert book.top() == (None, None, None, None)
