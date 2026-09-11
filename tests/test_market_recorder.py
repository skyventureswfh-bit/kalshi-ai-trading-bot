import json

import pytest

from src.auto.market_recorder import JsonlMarketRecorder, MarketObservation, UnsafeObservation


def _observation(**overrides):
    values = dict(received_epoch=100.2, event_epoch=100.1, upstream_received_epoch=None, source="kalshi",
                  ticker="KXBTC15M-TEST", target=77157.41, seconds_remaining=42.0,
                  yes_bid_cents=48, yes_ask_cents=50, sequence=1)
    values.update(overrides)
    return MarketObservation(**values)


def test_recorder_appends_versioned_observation(tmp_path):
    path = tmp_path / "observations.jsonl"
    recorder = JsonlMarketRecorder(str(path))
    recorder.record(_observation())
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1
    assert saved["ticker"] == "KXBTC15M-TEST"
    assert saved["yes_ask_cents"] == 50


def test_recorder_rejects_stale_or_reordered_data_without_appending(tmp_path):
    path = tmp_path / "observations.jsonl"
    recorder = JsonlMarketRecorder(str(path))
    recorder.record(_observation())
    with pytest.raises(UnsafeObservation, match="stale"):
        recorder.record(_observation(event_epoch=101.0, received_epoch=104.0, sequence=2))
    with pytest.raises(UnsafeObservation, match="out-of-order"):
        recorder.record(_observation(event_epoch=99.0, received_epoch=99.1, sequence=2))
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_recorder_has_no_execution_surface(tmp_path):
    recorder = JsonlMarketRecorder(str(tmp_path / "observations.jsonl"))
    forbidden = {"buy", "sell", "place_order", "execute_position"}
    assert forbidden.isdisjoint(dir(recorder))


def test_authoritative_snapshot_can_reset_source_sequence(tmp_path):
    path = tmp_path / "observations.jsonl"
    recorder = JsonlMarketRecorder(str(path))
    recorder.record(_observation(sequence=10))
    recorder.reset_source_ordering("kalshi")
    recorder.record(_observation(event_epoch=101.0, received_epoch=101.1, sequence=2))
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_synchronized_pair_enforces_cross_source_skew(tmp_path):
    path = tmp_path / "observations.jsonl"
    recorder = JsonlMarketRecorder(str(path))
    kalshi = _observation()
    brti = _observation(source="cf_brti", event_epoch=101.0, received_epoch=101.1,
                        sequence=None, yes_bid_cents=None, yes_ask_cents=None, brti=77158.0)
    with pytest.raises(UnsafeObservation, match="cross-source timestamp skew"):
        recorder.record_synchronized(kalshi, brti)
    assert not path.exists()
