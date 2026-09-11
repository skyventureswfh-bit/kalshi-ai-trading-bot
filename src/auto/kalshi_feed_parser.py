"""Pure parsers for documented Kalshi observation WebSocket messages."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

from src.auto.market_recorder import MarketObservation, UnsafeObservation


class OrderBookDesynchronized(UnsafeObservation):
    """The incremental book is unsafe until a fresh snapshot is received."""


class LiveOrderBook:
    """Maintain top-of-book from one snapshot followed by ordered deltas."""

    def __init__(self) -> None:
        self.ticker: Optional[str] = None
        self.sid: Optional[int] = None
        self.sequence: Optional[int] = None
        # Kalshi sends fixed-point decimal strings.  Keep them exact internally;
        # binary floats can turn 0.3 - 0.1 - 0.2 into a tiny negative quantity.
        self.yes: Dict[Decimal, Decimal] = {}
        self.no: Dict[Decimal, Decimal] = {}

    def load_snapshot(self, frame: Dict[str, Any]) -> None:
        if frame.get("type") != "orderbook_snapshot":
            raise UnsafeObservation("expected orderbook snapshot")
        msg = frame.get("msg") or {}
        self.ticker = str(msg["market_ticker"])
        self.sid = int(frame["sid"])
        self.sequence = int(frame["seq"])
        self.yes = _levels(msg.get("yes_dollars_fp") or [])
        self.no = _levels(msg.get("no_dollars_fp") or [])

    def apply_delta(self, frame: Dict[str, Any]) -> float:
        if self.sequence is None:
            raise OrderBookDesynchronized("orderbook delta received before snapshot")
        if frame.get("type") != "orderbook_delta" or int(frame["sid"]) != self.sid:
            raise OrderBookDesynchronized("orderbook delta stream mismatch")
        sequence = int(frame["seq"])
        if sequence != self.sequence + 1:
            raise OrderBookDesynchronized("orderbook sequence gap")
        msg = frame.get("msg") or {}
        if str(msg["market_ticker"]) != self.ticker:
            raise OrderBookDesynchronized("orderbook delta market mismatch")
        side = str(msg["side"]).lower()
        levels = self.yes if side == "yes" else self.no if side == "no" else None
        if levels is None:
            raise OrderBookDesynchronized("unknown orderbook side")
        price = Decimal(str(msg["price_dollars"])) * Decimal("100")
        quantity = levels.get(price, Decimal("0")) + Decimal(str(msg["delta_fp"]))
        if quantity < 0:
            raise OrderBookDesynchronized("orderbook quantity became negative")
        if quantity == 0:
            levels.pop(price, None)
        else:
            levels[price] = quantity
        self.sequence = sequence
        return float(msg["ts_ms"]) / 1000.0

    def top(self) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        yes_bid = max(self.yes, default=None)
        no_bid = max(self.no, default=None)
        return (
            None if yes_bid is None else float(yes_bid),
            None if no_bid is None else float(Decimal("100") - no_bid),
            None if no_bid is None else float(no_bid),
            None if yes_bid is None else float(Decimal("100") - yes_bid),
        )


def parse_cfbenchmarks_value(
    frame: Dict[str, Any], *, ticker: str, target: float, seconds_remaining: float,
    local_received_epoch: Optional[float] = None,
) -> MarketObservation:
    if frame.get("type") != "cfbenchmarks_value":
        raise UnsafeObservation("unexpected CF Benchmarks frame type")
    msg = frame.get("msg") or {}
    if msg.get("index_id") != "BRTI":
        raise UnsafeObservation("CF Benchmarks frame is not BRTI")
    raw = json.loads(msg.get("data") or "{}")
    average = msg.get("last_60s_windowed_average_15min")
    return MarketObservation(
        event_epoch=float(raw["time"]) / 1000.0,
        received_epoch=(float(local_received_epoch) if local_received_epoch is not None
                        else float(msg["received_at"]) / 1000.0),
        source="kalshi_cfbenchmarks_brti",
        upstream_received_epoch=float(msg["received_at"]) / 1000.0,
        ticker=ticker,
        target=target,
        seconds_remaining=seconds_remaining,
        brti=float(raw["value"]),
        official_final_minute_average=float(average["value"]) if average else None,
        official_average_window_size=int(average["window_size"]) if average else None,
        sequence=int(frame["seq"]),
    )


def parse_orderbook_snapshot(
    frame: Dict[str, Any], *, target: float, seconds_remaining: float, received_epoch: float
) -> MarketObservation:
    if frame.get("type") != "orderbook_snapshot":
        raise UnsafeObservation("unexpected orderbook frame type")
    msg = frame.get("msg") or {}
    yes_bid, yes_ask, no_bid, no_ask = _top_of_book(msg)
    return MarketObservation(
        event_epoch=received_epoch,
        received_epoch=received_epoch,
        upstream_received_epoch=None,
        source="kalshi_orderbook",
        ticker=str(msg["market_ticker"]),
        target=target,
        seconds_remaining=seconds_remaining,
        yes_bid_cents=yes_bid,
        yes_ask_cents=yes_ask,
        no_bid_cents=no_bid,
        no_ask_cents=no_ask,
        sequence=int(frame["seq"]),
    )


def _top_of_book(msg: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    yes_levels = msg.get("yes_dollars_fp") or []
    no_levels = msg.get("no_dollars_fp") or []
    yes_bid = max((float(level[0]) * 100 for level in yes_levels), default=None)
    no_bid = max((float(level[0]) * 100 for level in no_levels), default=None)
    yes_ask = None if no_bid is None else 100.0 - no_bid
    no_ask = None if yes_bid is None else 100.0 - yes_bid
    return yes_bid, yes_ask, no_bid, no_ask


def _levels(rows: Any) -> Dict[Decimal, Decimal]:
    return {
        Decimal(str(price)) * Decimal("100"): Decimal(str(quantity))
        for price, quantity in rows
        if Decimal(str(quantity)) > 0
    }
