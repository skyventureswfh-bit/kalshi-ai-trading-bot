"""Pure final-minute settlement math for offline Beast observation/replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Dict, Optional


class UnsafeSettlementData(ValueError):
    """Settlement inputs are incomplete or temporally unsafe."""


@dataclass(frozen=True)
class SettlementProjection:
    observed_seconds: int
    remaining_seconds: int
    observed_average: float
    projected_average_if_flat: float
    required_remaining_average: Optional[float]
    projected_margin: float


@dataclass
class FinalMinuteAccumulator:
    """Accumulate one observation per UTC second in a fixed final window.

    Kalshi's documented quarter-hour window is ``(start, close]``. Therefore
    valid second buckets are ``start+1`` through ``start+window_seconds``.
    """

    window_start_epoch: int
    target: float
    window_seconds: int = 60
    max_staleness_seconds: float = 2.0
    _values: Dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _latest_event_epoch: Optional[float] = field(default=None, init=False, repr=False)

    def add(self, *, event_epoch: float, value: float, received_epoch: float) -> None:
        if not all(isfinite(item) for item in (event_epoch, value, received_epoch)):
            raise UnsafeSettlementData("non-finite settlement observation")
        if received_epoch < event_epoch:
            raise UnsafeSettlementData("received time precedes event time")
        if received_epoch - event_epoch > self.max_staleness_seconds:
            raise UnsafeSettlementData("stale settlement observation")
        bucket = int(event_epoch)
        if not self.window_start_epoch < bucket <= self.window_start_epoch + self.window_seconds:
            raise UnsafeSettlementData("observation outside final-minute window")
        if self._latest_event_epoch is not None and event_epoch < self._latest_event_epoch:
            raise UnsafeSettlementData("out-of-order settlement observation")
        self._values[bucket] = value
        self._latest_event_epoch = event_epoch

    @property
    def observed_seconds(self) -> int:
        return len(self._values)

    @property
    def is_complete(self) -> bool:
        return self.observed_seconds == self.window_seconds

    def projection(self) -> SettlementProjection:
        if not self._values:
            raise UnsafeSettlementData("no settlement observations")
        total = sum(self._values.values())
        observed = len(self._values)
        remaining = self.window_seconds - observed
        latest = self._values[max(self._values)]
        projected = (total + latest * remaining) / self.window_seconds
        required = None if remaining == 0 else (
            (self.target * self.window_seconds - total) / remaining
        )
        return SettlementProjection(
            observed_seconds=observed,
            remaining_seconds=remaining,
            observed_average=total / observed,
            projected_average_if_flat=projected,
            required_remaining_average=required,
            projected_margin=projected - self.target,
        )

    def final_average(self) -> float:
        if not self.is_complete:
            raise UnsafeSettlementData(
                f"incomplete settlement window: {self.observed_seconds}/{self.window_seconds} seconds"
            )
        expected = set(range(self.window_start_epoch + 1, self.window_start_epoch + self.window_seconds + 1))
        if set(self._values) != expected:
            raise UnsafeSettlementData("settlement window contains missing seconds")
        return sum(self._values.values()) / self.window_seconds
