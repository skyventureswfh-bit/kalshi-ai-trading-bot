"""Command-line launcher for one read-only BTC 15-minute capture session."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from src.auto.market_recorder import JsonlMarketRecorder
from src.auto.market_discovery import discover_open_btc_market
from src.auto.observation_listener import ObservationListener, ObservationListenerConfig
from src.auto.session_replay import summarize_capture


class RequestSigner:
    def __init__(self, private_key_path: str) -> None:
        with Path(private_key_path).open("rb") as handle:
            self._key = serialization.load_pem_private_key(handle.read(), password=None)

    def __call__(self, timestamp: str, method: str, path: str) -> str:
        message = f"{timestamp}{method.upper()}{path.split('?')[0]}".encode("utf-8")
        signature = self._key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")


def _expiration_epoch(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture one Kalshi BTC market; never trade")
    parser.add_argument("--series", default="KXBTC15M")
    parser.add_argument("--ticker")
    parser.add_argument("--target", type=float)
    parser.add_argument("--expires", help="UTC ISO time, for example 2026-09-11T01:15:00Z")
    parser.add_argument("--output")
    return parser


async def capture(args: argparse.Namespace) -> dict:
    api_key_id = os.environ.get("KALSHI_API_KEY", "").strip()
    private_key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "").strip()
    if not api_key_id or not private_key_path:
        raise RuntimeError("KALSHI_API_KEY and KALSHI_PRIVATE_KEY_PATH are required")
    supplied = (args.ticker, args.target, args.expires)
    if any(value is not None for value in supplied) and not all(value is not None for value in supplied):
        raise RuntimeError("ticker, target, and expires must be supplied together")
    if all(value is not None for value in supplied):
        ticker, target, expiration = args.ticker, args.target, _expiration_epoch(args.expires)
    else:
        discovered = discover_open_btc_market(series_ticker=args.series)
        ticker, target, expiration = discovered.ticker, discovered.target, discovered.expiration_epoch
    output = args.output or f"data/beast_observations/{ticker}.jsonl"
    listener = ObservationListener(
        config=ObservationListenerConfig(
            ticker=ticker,
            target=target,
            expiration_epoch=expiration,
            api_key_id=api_key_id,
        ),
        recorder=JsonlMarketRecorder(output),
        sign_request=RequestSigner(private_key_path),
    )
    await listener.run_forever()
    return summarize_capture(output).as_dict()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(asyncio.run(capture(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
