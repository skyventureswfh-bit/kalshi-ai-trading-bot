import json

from cli import build_parser
from src.auto.market_recorder import MarketObservation
from src.auto.wave_replay import discover_captures, replay_campaign, replay_capture


def _row(source, epoch, *, brti=None, bid=59, ask=60, remaining=100):
    base = dict(
        received_epoch=epoch,
        event_epoch=epoch,
        upstream_received_epoch=epoch,
        source=source,
        ticker="KXTEST",
        target=100.0,
        seconds_remaining=remaining,
        brti=brti,
        yes_bid_cents=None,
        yes_ask_cents=None,
        no_bid_cents=None,
        no_ask_cents=None,
        official_final_minute_average=None,
        official_average_window_size=None,
        sequence=None,
        schema_version=1,
    )
    if source == "kalshi_ticker":
        base.update(
            yes_bid_cents=bid, yes_ask_cents=ask,
            no_bid_cents=100 - ask, no_ask_cents=100 - bid,
        )
    return base


def _complete_capture(path):
    rows = []
    for index in range(9):
        epoch = 1000.0 + index * 2
        brti = _row(
            "kalshi_cfbenchmarks_brti", epoch, brti=100 + index,
            remaining=100 - index * 2,
        )
        if index == 8:
            brti["official_final_minute_average"] = 105.0
            brti["official_average_window_size"] = 60
        rows.append(brti)
        # Stay executable while momentum matures, then reach the profit rail.
        if index < 7:
            bid, ask = 58 + index * 0.25, 59 + index * 0.25
        else:
            bid, ask = 89, 90
        rows.append(_row(
            "kalshi_ticker", epoch + 0.1, bid=bid, ask=ask,
            remaining=100 - index * 2,
        ))
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_cli_exposes_offline_wave_replay():
    args = build_parser().parse_args(["replay-wave", "--limit", "4"])
    assert args.command == "replay-wave"
    assert args.limit == 4
    assert args.func.__name__ == "cmd_replay_wave"


def test_discovery_excludes_wave_decision_journals(tmp_path):
    (tmp_path / "one.jsonl").write_text("{}\n")
    (tmp_path / "one.wave.jsonl").write_text("{}\n")
    assert discover_captures(str(tmp_path)) == [tmp_path / "one.jsonl"]


def test_replay_uses_locked_paper_engine_and_aggregates(tmp_path):
    capture = tmp_path / "round.jsonl"
    _complete_capture(capture)
    result = replay_capture(capture)
    assert result["ticker"] == "KXTEST"
    assert result["result"] == "UP"
    assert result["quote_observations"] == 9
    assert result["position_open_at_end"] is False
    assert result["scorecard"]["trades"] == 1

    campaign = replay_campaign([capture])
    assert campaign["mode"] == "OFFLINE_WAVE_REPLAY_NO_ORDERS"
    assert campaign["completed_rounds"] == 1
    assert campaign["rounds_with_entries"] == 1
    assert campaign["scorecard"]["trades"] == 1


def test_campaign_reports_incomplete_capture_instead_of_counting_it(tmp_path):
    capture = tmp_path / "incomplete.jsonl"
    capture.write_text(json.dumps(_row(
        "kalshi_cfbenchmarks_brti", 1000.0, brti=101.0,
    )) + "\n")
    campaign = replay_campaign([capture])
    assert campaign["completed_rounds"] == 0
    assert len(campaign["skipped_captures"]) == 1
    assert campaign["scorecard"]["trades"] == 0
