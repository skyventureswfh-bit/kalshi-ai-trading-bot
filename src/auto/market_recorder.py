"""Append-only synchronized market observations for offline Beast replay."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Dict, Optional


class UnsafeObservation(ValueError):
    """Timestamps or market values are unsafe to record."""


@dataclass(frozen=True)
class MarketObservation:
    received_epoch: float
    event_epoch: float
    upstream_received_epoch: Optional[float]
    source: str
    ticker: str
    target: float
    seconds_remaining: float
    brti: Optional[float] = None
    yes_bid_cents: Optional[float] = None
    yes_ask_cents: Optional[float] = None
    no_bid_cents: Optional[float] = None
    no_ask_cents: Optional[float] = None
    official_final_minute_average: Optional[float] = None
    official_average_window_size: Optional[int] = None
    sequence: Optional[int] = None


class JsonlMarketRecorder:
    """Validate and durably append observations as newline-delimited JSON."""

    def __init__(self, path: str, *, max_staleness_seconds: float = 2.0) -> None:
        self.path = Path(path)
        self.max_staleness_seconds = max_staleness_seconds
        self._last_event_by_source: Dict[str, float] = {}
        self._last_sequence_by_source: Dict[str, int] = {}

    def record(self, observation: MarketObservation) -> None:
        self._validate(observation)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(observation)
        payload["schema_version"] = 1
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        previous_event = self._last_event_by_source.get(observation.source)
        self._last_event_by_source[observation.source] = (
            observation.event_epoch
            if previous_event is None
            else max(previous_event, observation.event_epoch)
        )
        if observation.sequence is not None:
            self._last_sequence_by_source[observation.source] = observation.sequence

    def record_synchronized(
        self,
        first: MarketObservation,
        second: MarketObservation,
        *,
        max_event_skew_seconds: float = 0.5,
    ) -> None:
        """Record a cross-source pair only when it describes the same instant."""
        self._validate(first)
        self._validate(second)
        if first.source == second.source:
            raise UnsafeObservation("synchronized pair requires two sources")
        if first.ticker != second.ticker or first.target != second.target:
            raise UnsafeObservation("synchronized pair market mismatch")
        if abs(first.event_epoch - second.event_epoch) > max_event_skew_seconds:
            raise UnsafeObservation("cross-source timestamp skew")
        self.record(first)
        self.record(second)

    def reset_source_ordering(self, source: str) -> None:
        """Start a new authoritative stream epoch for one source.

        WebSocket sequence numbers can restart after a reconnect.  A fresh
        order-book snapshot is the boundary that makes that reset safe.
        """
        self._last_event_by_source.pop(source, None)
        self._last_sequence_by_source.pop(source, None)

    def _validate(self, item: MarketObservation) -> None:
        numeric = (item.received_epoch, item.event_epoch, item.target, item.seconds_remaining)
        if not all(isfinite(value) for value in numeric):
            raise UnsafeObservation("non-finite observation value")
        if (item.upstream_received_epoch is not None
                and not isfinite(item.upstream_received_epoch)):
            raise UnsafeObservation("non-finite upstream receive time")
        if not item.source or not item.ticker:
            raise UnsafeObservation("source and ticker are required")
        # Prefer Kalshi's own receipt timestamp when the channel supplies it.
        # This measures feed latency without depending on the laptop clock.
        freshness_epoch = (
            item.upstream_received_epoch
            if item.upstream_received_epoch is not None
            else item.received_epoch
        )
        if freshness_epoch < item.event_epoch:
            raise UnsafeObservation("received time precedes event time")
        if freshness_epoch - item.event_epoch > self.max_staleness_seconds:
            raise UnsafeObservation("stale observation")
        previous_event = self._last_event_by_source.get(item.source)
        previous_sequence = self._last_sequence_by_source.get(item.source)
        # Kalshi can publish a newer sequenced frame whose embedded source time
        # regresses slightly.  Sequence establishes message order; timestamp
        # order is required only when no sequence proves the ordering.
        if (previous_event is not None and item.event_epoch < previous_event
                and (item.sequence is None or previous_sequence is None)):
            raise UnsafeObservation("out-of-order source timestamp")
        if item.sequence is not None:
            if previous_sequence is not None and item.sequence <= previous_sequence:
                raise UnsafeObservation("duplicate or out-of-order source sequence")
        book_fields = ("yes_bid_cents", "yes_ask_cents", "no_bid_cents", "no_ask_cents")
        for name in book_fields:
            value = getattr(item, name)
            if value is not None and not 0 <= value <= 100:
                raise UnsafeObservation(f"{name} must be between 0 and 100")
        if item.brti is None and item.official_final_minute_average is None and all(
            getattr(item, name) is None for name in book_fields
        ):
            raise UnsafeObservation("observation contains neither BRTI nor order-book data")
