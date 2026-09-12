import json

import pytest

from src.auto.market_recorder import JsonlMarketRecorder
from src.auto.observation_listener import ObservationListener, ObservationListenerConfig


def _listener(tmp_path):
    calls = []

    def signer(timestamp, method, path):
        calls.append((timestamp, method, path))
        return "signed"

    config = ObservationListenerConfig(
        ticker="KXBTC15M-TEST", target=68001, expiration_epoch=1710000060,
        api_key_id="key-id",
    )
    listener = ObservationListener(
        config=config,
        recorder=JsonlMarketRecorder(str(tmp_path / "observations.jsonl")),
        sign_request=signer,
        clock=lambda: 1710000000.2,
    )
    return listener, calls


def test_auth_signs_documented_websocket_path(tmp_path):
    listener, calls = _listener(tmp_path)
    headers = listener.authentication_headers()
    assert headers["KALSHI-ACCESS-KEY"] == "key-id"
    assert headers["KALSHI-ACCESS-SIGNATURE"] == "signed"
    assert calls[0][1:] == ("GET", "/trade-api/ws/v2")


def test_subscribes_only_to_observation_channels(tmp_path):
    listener, _ = _listener(tmp_path)
    serialized = json.dumps(listener.subscription_commands("KXBTC15M-TEST"))
    assert "ticker" in serialized
    assert "orderbook_delta" not in serialized
    assert "cfbenchmarks_value" in serialized
    assert "BRTI" in serialized
    for forbidden in ("portfolio", "fill", "buy", "sell"):
        assert forbidden not in serialized


def test_processes_market_ticker_into_recorder(tmp_path):
    listener, _ = _listener(tmp_path)
    frame = {
        "type": "ticker", "sid": 7,
        "msg": {"market_ticker": "KXBTC15M-TEST",
                "yes_bid_dollars": "0.5800", "yes_ask_dollars": "0.6100",
                "ts_ms": 1710000000100},
    }
    assert listener.process_frame(frame, local_received_epoch=1710000000.2)
    saved = json.loads(listener.recorder.path.read_text())
    assert saved["source"] == "kalshi_ticker"
    assert saved["yes_ask_cents"] == 61


def test_processes_cf_frame_into_recorder(tmp_path):
    listener, _ = _listener(tmp_path)
    frame = {
        "type": "cfbenchmarks_value", "seq": 42,
        "msg": {"index_id": "BRTI", "received_at": 1710000000141,
                "data": '{"time":1710000000123,"value":"68000.12"}',
                "last_60s_windowed_average_15min": {
                    "value": "68000.23", "window_size": 14}},
    }
    assert listener.process_frame(frame, local_received_epoch=1710000000.2) is True
    saved = json.loads(listener.recorder.path.read_text())
    assert saved["official_final_minute_average"] == 68000.23
    assert saved["seconds_remaining"] == pytest.approx(59.859)


def test_complete_official_window_stops_session(tmp_path):
    listener, _ = _listener(tmp_path)
    frame = {
        "type": "cfbenchmarks_value", "seq": 60,
        "msg": {"index_id": "BRTI", "received_at": 1710000000141,
                "data": '{"time":1710000000123,"value":"68002.00"}',
                "last_60s_windowed_average_15min": {
                    "value": "68001.25", "window_size": 60}},
    }
    listener.process_frame(frame, local_received_epoch=1710000000.2)
    assert listener._stop.is_set()


def test_fresh_snapshot_resets_orderbook_sequence_epoch(tmp_path):
    listener, _ = _listener(tmp_path)
    first = {"type": "orderbook_snapshot", "sid": 2, "seq": 10,
             "msg": {"market_ticker": "KXBTC15M-TEST",
                     "yes_dollars_fp": [["0.4800", "10"]],
                     "no_dollars_fp": [["0.4900", "8"]]}}
    fresh = {"type": "orderbook_snapshot", "sid": 3, "seq": 2,
             "msg": {"market_ticker": "KXBTC15M-TEST",
                     "yes_dollars_fp": [["0.5000", "12"]],
                     "no_dollars_fp": [["0.4700", "9"]]}}
    assert listener.process_frame(first, local_received_epoch=1710000000.2)
    assert listener.process_frame(fresh, local_received_epoch=1710000000.3)
    assert len(listener.recorder.path.read_text().splitlines()) == 2


def test_cf_clock_calibrates_orderbook_receive_time(tmp_path):
    listener, _ = _listener(tmp_path)
    cf = {
        "type": "cfbenchmarks_value", "seq": 42,
        "msg": {"index_id": "BRTI", "received_at": 1710000000141,
                "data": '{"time":1710000000123,"value":"68000.12"}'},
    }
    snapshot = {
        "type": "orderbook_snapshot", "sid": 2, "seq": 2,
        "msg": {"market_ticker": "KXBTC15M-TEST",
                "yes_dollars_fp": [["0.4800", "10"]],
                "no_dollars_fp": [["0.4900", "8"]]},
    }
    # The laptop clock is 2.5 seconds ahead of Kalshi's receipt clock.
    assert listener.process_frame(cf, local_received_epoch=1710000002.641)
    assert listener.server_clock_offset_seconds == pytest.approx(2.5)
    assert listener.process_frame(snapshot, local_received_epoch=1710000002.741)
    saved = [json.loads(line) for line in listener.recorder.path.read_text().splitlines()]
    assert saved[-1]["received_epoch"] == pytest.approx(1710000000.241)
    assert saved[-1]["seconds_remaining"] == pytest.approx(59.759)
