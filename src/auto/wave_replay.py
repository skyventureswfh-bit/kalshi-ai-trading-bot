"""Offline replay of recorded BTC Wave captures.

The replay path is intentionally read-only: it loads the append-only market
recordings and feeds them through the same :class:`WavePaperSession` used by
live paper observation.  It never constructs a Kalshi client and therefore
cannot place an order.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path
from typing import Iterable

from src.auto.market_recorder import MarketObservation
from src.auto.session_replay import IncompleteCapture, summarize_capture
from src.auto.wave_paper import WavePaperSession
from src.auto.wave_strategy import WaveAction, build_scorecard


_OBSERVATION_FIELDS = {field.name for field in fields(MarketObservation)}


def discover_captures(directory: str = "data/beast_observations") -> list[Path]:
    """Return raw capture files, excluding the decision journals."""
    root = Path(directory)
    if not root.exists():
        return []
    return sorted(
        path for path in root.glob("*.jsonl")
        if not path.name.endswith(".wave.jsonl")
    )


def _load_observation(row: dict) -> MarketObservation:
    payload = {key: value for key, value in row.items() if key in _OBSERVATION_FIELDS}
    return MarketObservation(**payload)


def replay_capture(path: str | Path) -> dict:
    """Replay one complete capture through the locked Wave paper strategy."""
    capture_path = Path(path)
    settlement = summarize_capture(str(capture_path))
    session = WavePaperSession()
    actions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    entries: list[dict] = []
    exits: list[dict] = []

    for line in capture_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = session.on_observation(_load_observation(json.loads(line)))
        if event is None:
            continue
        decision = event.decision
        actions[decision.action.value] += 1
        if decision.action == WaveAction.WAIT:
            reasons[decision.reason] += 1
        if decision.action in {WaveAction.ENTER_UP, WaveAction.ENTER_DOWN}:
            entries.append(asdict(decision))
        elif decision.action in {
            WaveAction.EXIT_PROFIT, WaveAction.EXIT_STOP, WaveAction.EXIT_TIME,
        }:
            exits.append(asdict(decision))

    position_open = session.strategy.position_for(settlement.ticker) is not None
    return {
        **settlement.as_dict(),
        "capture": str(capture_path),
        "actions": dict(sorted(actions.items())),
        "wait_reason_counts": dict(sorted(reasons.items())),
        "top_wait_reasons": [
            {"reason": reason, "observations": count}
            for reason, count in reasons.most_common(5)
        ],
        "entries": entries,
        "exits": exits,
        "position_open_at_end": position_open,
        "scorecard": asdict(session.scorecard),
        "results": [asdict(result) | {"net_pnl_dollars": result.net_pnl_dollars}
                    for result in session.results],
    }


def replay_campaign(paths: Iterable[str | Path]) -> dict:
    """Replay complete captures and aggregate an evidence-first scorecard."""
    rounds: list[dict] = []
    skipped: list[dict] = []
    all_results = []
    action_totals: Counter[str] = Counter()
    reason_totals: Counter[str] = Counter()

    for path in paths:
        try:
            result = replay_capture(path)
        except (IncompleteCapture, json.JSONDecodeError, TypeError, ValueError) as exc:
            skipped.append({"capture": str(path), "reason": str(exc)})
            continue
        rounds.append(result)
        action_totals.update(result["actions"])
        reason_totals.update(result["wait_reason_counts"])
        for item in result["results"]:
            all_results.append(item)

    # Recreate the immutable result objects only once at the aggregation edge.
    from src.auto.wave_strategy import WaveTradeResult

    scorecard = build_scorecard([
        WaveTradeResult(
            ticker=item["ticker"],
            side=item["side"],
            gross_pnl_dollars=item["gross_pnl_dollars"],
            fees_dollars=item["fees_dollars"],
            slippage_dollars=item["slippage_dollars"],
        )
        for item in all_results
    ])
    return {
        "mode": "OFFLINE_WAVE_REPLAY_NO_ORDERS",
        "captures_found": len(rounds) + len(skipped),
        "completed_rounds": len(rounds),
        "skipped_captures": skipped,
        "rounds_with_entries": sum(bool(row["entries"]) for row in rounds),
        "rounds_without_entries": sum(not row["entries"] for row in rounds),
        "open_positions_at_end": sum(row["position_open_at_end"] for row in rounds),
        "action_totals": dict(sorted(action_totals.items())),
        "top_wait_reasons": [
            {"reason": reason, "observations": count}
            for reason, count in reason_totals.most_common(8)
        ],
        "scorecard": asdict(scorecard),
        "rounds": rounds,
    }
