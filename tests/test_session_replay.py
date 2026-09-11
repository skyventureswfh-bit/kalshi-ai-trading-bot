import json

import pytest

from src.auto.session_replay import IncompleteCapture, summarize_capture


def test_replay_requires_official_complete_average(tmp_path):
    path = tmp_path / "capture.jsonl"
    path.write_text(json.dumps({"ticker": "KX", "target": 100, "brti": 99,
                               "official_average_window_size": 59,
                               "official_final_minute_average": 99.5}) + "\n")
    with pytest.raises(IncompleteCapture, match="60-sample"):
        summarize_capture(str(path))


def test_replay_reports_result_margin_and_book_movement(tmp_path):
    path = tmp_path / "capture.jsonl"
    rows = [
        {"source": "kalshi_orderbook", "yes_bid_cents": 39, "yes_ask_cents": 41},
        {"source": "kalshi_orderbook", "yes_bid_cents": 79, "yes_ask_cents": 81},
        {"source": "kalshi_cfbenchmarks_brti", "ticker": "KX", "target": 100,
         "brti": 102, "official_average_window_size": 60,
         "official_final_minute_average": 101.25},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = summarize_capture(str(path))
    assert result.result == "UP"
    assert result.official_margin == 1.25
    assert result.first_yes_mid_cents == 40
    assert result.last_yes_mid_cents == 80

