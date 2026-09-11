import pytest

from src.auto.capture_session import _expiration_epoch, build_parser


def test_launcher_can_start_in_auto_discovery_mode():
    args = build_parser().parse_args([])
    assert args.series == "KXBTC15M"
    assert args.ticker is None


def test_launcher_parses_market_without_credentials_on_command_line():
    args = build_parser().parse_args([
        "--ticker", "KXBTC15M-TEST", "--target", "77157.41",
        "--expires", "2026-09-11T01:15:00Z", "--output", "capture.jsonl",
    ])
    assert args.target == 77157.41
    assert _expiration_epoch(args.expires) > 0
    assert "key" not in vars(args)
    assert "private_key" not in vars(args)
