"""Summarize one completed Beast observation file without trading advice."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List


class IncompleteCapture(ValueError):
    """The recording cannot prove a complete official settlement window."""


@dataclass(frozen=True)
class ReplaySummary:
    ticker: str
    target: float
    official_final_average: float
    official_margin: float
    result: str
    brti_observations: int
    orderbook_observations: int
    first_yes_mid_cents: float | None
    last_yes_mid_cents: float | None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_capture(path: str) -> ReplaySummary:
    rows: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise IncompleteCapture("capture is empty")

    brti = [row for row in rows if row.get("brti") is not None]
    books = [row for row in rows if row.get("source") == "kalshi_orderbook"]
    complete = [row for row in brti if (row.get("official_average_window_size") or 0) >= 60]
    if not complete:
        raise IncompleteCapture("missing official 60-sample final-minute average")
    final = complete[-1]
    target = float(final["target"])
    average = float(final["official_final_minute_average"])
    mids = [_yes_mid(row) for row in books]
    mids = [value for value in mids if value is not None]
    return ReplaySummary(
        ticker=str(final["ticker"]),
        target=target,
        official_final_average=average,
        official_margin=average - target,
        result="UP" if average > target else "DOWN" if average < target else "TIE",
        brti_observations=len(brti),
        orderbook_observations=len(books),
        first_yes_mid_cents=mids[0] if mids else None,
        last_yes_mid_cents=mids[-1] if mids else None,
    )


def _yes_mid(row: Dict[str, Any]) -> float | None:
    bid = row.get("yes_bid_cents")
    ask = row.get("yes_ask_cents")
    if bid is None or ask is None:
        return None
    return (float(bid) + float(ask)) / 2.0

