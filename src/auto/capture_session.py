"""Command-line launcher for read-only BTC 15-minute capture sessions."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from src.auto.market_recorder import JsonlMarketRecorder
from src.auto.market_discovery import discover_open_btc_market
from src.auto.observation_listener import ObservationListener, ObservationListenerConfig
from src.auto.session_replay import summarize_capture
from src.auto.wave_paper import WavePaperSession
from src.auto.wave_strategy import WaveAction, WaveJournal


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
    parser = argparse.ArgumentParser(description="Capture Kalshi BTC markets; never trade")
    parser.add_argument("--series", default="KXBTC15M")
    parser.add_argument("--ticker")
    parser.add_argument("--target", type=float)
    parser.add_argument("--expires", help="UTC ISO time, for example 2026-09-11T01:15:00Z")
    parser.add_argument("--output")
    parser.add_argument(
        "--wave-paper",
        action="store_true",
        help="Run the locked Wave strategy against the read-only live feed; never trade",
    )
    parser.add_argument("--wave-journal", help="Optional Wave decision/result JSONL path")
    parser.add_argument(
        "--rounds", type=int, default=1,
        help="Number of consecutive markets to capture (default: 1; maximum: 96)",
    )
    return parser


async def capture(args: argparse.Namespace) -> dict:
    from dotenv import load_dotenv

    # Match the rest of Beast: credentials live in the project-root .env file.
    # Existing process environment values still win because load_dotenv does
    # not override them by default.
    load_dotenv()
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
    wave_session = None
    on_observation = None
    wave_journal_path = None
    if args.wave_paper:
        wave_journal_path = args.wave_journal or str(Path(output).with_suffix(".wave.jsonl"))
        wave_session = WavePaperSession(journal=WaveJournal(wave_journal_path))

        def _paper_observation(item):
            event = wave_session.on_observation(item)
            if event is not None and event.decision.action in {
                WaveAction.ENTER_UP,
                WaveAction.ENTER_DOWN,
                WaveAction.EXIT_PROFIT,
                WaveAction.EXIT_STOP,
                WaveAction.EXIT_TIME,
            }:
                decision = event.decision
                print(
                    f"WAVE PAPER {decision.action.value}: {decision.side or '-'} "
                    f"@ {decision.executable_price_cents or 0:.1f}c — {decision.reason}",
                    file=sys.stderr,
                    flush=True,
                )

        on_observation = _paper_observation
    listener = ObservationListener(
        config=ObservationListenerConfig(
            ticker=ticker,
            target=target,
            expiration_epoch=expiration,
            api_key_id=api_key_id,
        ),
        recorder=JsonlMarketRecorder(output),
        sign_request=RequestSigner(private_key_path),
        on_observation=on_observation,
    )
    await listener.run_forever()
    result = summarize_capture(output).as_dict()
    result["feed_safety"] = {
        "dropped_observations": listener.dropped_observations,
        "orderbook_resyncs": listener.orderbook_resyncs,
        "last_rejection_reason": listener.last_rejection_reason,
        "server_clock_offset_seconds": listener.server_clock_offset_seconds,
    }
    if wave_session is not None:
        result["wave_paper"] = asdict(wave_session.scorecard)
        result["wave_journal"] = wave_journal_path
    return result


def _validate_rounds(args: argparse.Namespace) -> int:
    rounds = int(getattr(args, "rounds", 1))
    if not 1 <= rounds <= 96:
        raise RuntimeError("rounds must be between 1 and 96")
    if rounds > 1 and any((args.ticker, args.target, args.expires)):
        raise RuntimeError("multiple rounds require automatic market discovery")
    if rounds > 1 and (args.output or args.wave_journal):
        raise RuntimeError("custom output paths are supported only for one round")
    return rounds


def _campaign_summary(results: list[dict], requested_rounds: int) -> dict:
    markets = []
    trades = wins = losses = 0
    quote_observations = dropped_observations = 0
    net_pnl = fees = slippage = 0.0
    net_wins = net_losses = 0.0
    for result in results:
        score = result.get("wave_paper") or {}
        safety = result.get("feed_safety") or {}
        round_trades = int(score.get("trades") or 0)
        round_wins = int(score.get("wins") or 0)
        round_losses = int(score.get("losses") or 0)
        trades += round_trades
        wins += round_wins
        losses += round_losses
        round_quotes = int(result.get("quote_observations") or 0)
        round_dropped = int(safety.get("dropped_observations") or 0)
        quote_observations += round_quotes
        dropped_observations += round_dropped
        net_pnl += float(score.get("net_pnl_dollars") or 0.0)
        fees += float(score.get("total_fees_dollars") or 0.0)
        slippage += float(score.get("total_slippage_dollars") or 0.0)
        net_wins += float(score.get("average_net_win_dollars") or 0.0) * round_wins
        net_losses += float(score.get("average_net_loss_dollars") or 0.0) * round_losses
        markets.append({
            "ticker": result.get("ticker"),
            "result": result.get("result"),
            "quote_observations": round_quotes,
            "dropped_observations": round_dropped,
            "trades": round_trades,
            "wins": round_wins,
            "losses": round_losses,
            "net_pnl_dollars": float(score.get("net_pnl_dollars") or 0.0),
        })
    return {
        "requested_rounds": requested_rounds,
        "completed_rounds": len(results),
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / trades if trades else 0.0,
        "average_net_win_dollars": net_wins / wins if wins else 0.0,
        "average_net_loss_dollars": net_losses / losses if losses else 0.0,
        "total_fees_dollars": fees,
        "total_slippage_dollars": slippage,
        "net_pnl_dollars": net_pnl,
        "quote_observations": quote_observations,
        "dropped_observations": dropped_observations,
        "markets": markets,
    }


async def capture_rounds(args: argparse.Namespace) -> dict:
    rounds = _validate_rounds(args)
    if rounds == 1:
        return await capture(args)
    results = []
    for number in range(1, rounds + 1):
        result = await capture(args)
        results.append(result)
        score = result.get("wave_paper") or {}
        print(
            f"PAPER ROUND {number}/{rounds} complete: {result.get('ticker')} "
            f"result={result.get('result')} trades={score.get('trades', 0)} "
            f"net=${float(score.get('net_pnl_dollars') or 0.0):.2f}",
            file=sys.stderr,
            flush=True,
        )
    return {"paper_campaign": _campaign_summary(results, rounds)}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(asyncio.run(capture_rounds(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
