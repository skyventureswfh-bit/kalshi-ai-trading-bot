import asyncio
from argparse import Namespace

import pytest

from cli import build_parser as build_cli_parser
from src.auto import capture_session
from src.auto.capture_session import (
    _campaign_summary,
    _expiration_epoch,
    _validate_rounds,
    build_parser,
)


def test_launcher_can_start_in_auto_discovery_mode():
    args = build_parser().parse_args([])
    assert args.series == "KXBTC15M"
    assert args.ticker is None
    assert args.rounds == 1


def test_launcher_parses_market_without_credentials_on_command_line():
    args = build_parser().parse_args([
        "--ticker", "KXBTC15M-TEST", "--target", "77157.41",
        "--expires", "2026-09-11T01:15:00Z", "--output", "capture.jsonl",
    ])
    assert args.target == 77157.41
    assert _expiration_epoch(args.expires) > 0
    assert "key" not in vars(args)
    assert "private_key" not in vars(args)


def test_unified_cli_exposes_read_only_btc_observer():
    args = build_cli_parser().parse_args(["observe-btc"])
    assert args.command == "observe-btc"
    assert args.series == "KXBTC15M"
    assert args.output is None
    assert args.rounds == 1
    assert args.func.__name__ == "cmd_observe_btc"


def test_launcher_accepts_fixed_multi_round_paper_campaign():
    args = build_cli_parser().parse_args([
        "observe-btc", "--wave-paper", "--rounds", "20",
    ])
    assert args.rounds == 20


def test_multi_round_rejects_paths_that_would_overwrite():
    args = Namespace(
        rounds=2, ticker=None, target=None, expires=None,
        output="same.jsonl", wave_journal=None,
    )
    with pytest.raises(RuntimeError, match="custom output"):
        _validate_rounds(args)


def test_campaign_summary_combines_round_scorecards():
    results = [
        {"ticker": "A", "result": "UP", "quote_observations": 50,
         "feed_safety": {"dropped_observations": 0},
         "wave_paper": {"trades": 1, "wins": 1, "losses": 0,
                        "average_net_win_dollars": 3.2,
                        "average_net_loss_dollars": 0.0,
                        "net_pnl_dollars": 3.2, "total_fees_dollars": 0.2,
                        "total_slippage_dollars": 0.0}},
        {"ticker": "B", "result": "DOWN", "quote_observations": 60,
         "feed_safety": {"dropped_observations": 1},
         "wave_paper": {"trades": 1, "wins": 0, "losses": 1,
                        "average_net_win_dollars": 0.0,
                        "average_net_loss_dollars": -3.0,
                        "net_pnl_dollars": -3.0, "total_fees_dollars": 0.1,
                        "total_slippage_dollars": 0.0}},
    ]
    summary = _campaign_summary(results, 2)
    assert summary["completed_rounds"] == 2
    assert summary["trades"] == 2
    assert summary["win_rate"] == 0.5
    assert summary["quote_observations"] == 110
    assert summary["dropped_observations"] == 1
    assert summary["average_net_win_dollars"] == pytest.approx(3.2)
    assert summary["average_net_loss_dollars"] == pytest.approx(-3.0)
    assert summary["net_pnl_dollars"] == pytest.approx(0.2)


def test_capture_rounds_rolls_over_fixed_number_without_execution(monkeypatch):
    calls = []

    async def fake_capture(args):
        number = len(calls) + 1
        calls.append(number)
        return {
            "ticker": f"ROUND-{number}", "result": "UP",
            "quote_observations": 10,
            "feed_safety": {"dropped_observations": 0},
            "wave_paper": {"trades": 0, "wins": 0, "losses": 0,
                           "net_pnl_dollars": 0.0},
        }

    monkeypatch.setattr(capture_session, "capture", fake_capture)
    args = Namespace(
        rounds=3, ticker=None, target=None, expires=None,
        output=None, wave_journal=None,
    )
    result = asyncio.run(capture_session.capture_rounds(args))
    assert calls == [1, 2, 3]
    assert result["paper_campaign"]["completed_rounds"] == 3
    assert result["paper_campaign"]["quote_observations"] == 30
