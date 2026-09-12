"""Authenticated, read-only Kalshi WebSocket listener for Beast observations.

Only the documented market-ticker and CF Benchmarks channels are subscribed.
There are deliberately no portfolio, order, fill, buy, or sell operations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from src.auto.kalshi_feed_parser import (
    LiveOrderBook,
    OrderBookDesynchronized,
    parse_cfbenchmarks_value,
    parse_market_ticker,
    parse_orderbook_snapshot,
)
from src.auto.market_recorder import JsonlMarketRecorder, MarketObservation, UnsafeObservation


PRODUCTION_WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
WS_SIGNING_PATH = "/trade-api/ws/v2"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObservationListenerConfig:
    ticker: str
    target: float
    expiration_epoch: float
    api_key_id: str
    websocket_url: str = PRODUCTION_WS_URL
    reconnect_initial_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    settlement_grace_seconds: float = 15.0


class ObservationListener:
    def __init__(
        self,
        *,
        config: ObservationListenerConfig,
        recorder: JsonlMarketRecorder,
        sign_request: Callable[[str, str, str], str],
        websocket_connect: Optional[Callable[..., Any]] = None,
        clock: Callable[[], float] = time.time,
        on_observation: Optional[Callable[[MarketObservation], None]] = None,
    ) -> None:
        self.config = config
        self.recorder = recorder
        self.sign_request = sign_request
        self.websocket_connect = websocket_connect
        self.clock = clock
        self.on_observation = on_observation
        self._stop = asyncio.Event()
        self._orderbook = LiveOrderBook()
        self.dropped_observations = 0
        self.orderbook_resyncs = 0
        self.last_rejection_reason: Optional[str] = None
        self.server_clock_offset_seconds: Optional[float] = None

    def stop(self) -> None:
        self._stop.set()

    def authentication_headers(self) -> Dict[str, str]:
        timestamp = str(int(self.clock() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self.config.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": self.sign_request(timestamp, "GET", WS_SIGNING_PATH),
        }

    @staticmethod
    def subscription_commands(ticker: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        return (
            {"id": 1, "cmd": "subscribe", "params": {
                "channels": ["ticker"], "market_ticker": ticker}},
            {"id": 2, "cmd": "subscribe", "params": {
                "channels": ["cfbenchmarks_value"], "index_ids": ["BRTI"]}},
        )

    async def run_forever(self) -> None:
        connector = self.websocket_connect
        if connector is None:
            from websockets.asyncio.client import connect
            connector = connect
        delay = self.config.reconnect_initial_seconds
        while not self._stop.is_set():
            if self.clock() >= self.config.expiration_epoch + self.config.settlement_grace_seconds:
                raise UnsafeObservation("official final-minute average did not complete")
            try:
                async with connector(
                    self.config.websocket_url,
                    additional_headers=self.authentication_headers(),
                    ping_interval=20,
                    ping_timeout=20,
                ) as socket:
                    for command in self.subscription_commands(self.config.ticker):
                        await socket.send(json.dumps(command))
                    delay = self.config.reconnect_initial_seconds
                    while not self._stop.is_set():
                        remaining = (self.config.expiration_epoch
                                     + self.config.settlement_grace_seconds - self.clock())
                        if remaining <= 0:
                            raise UnsafeObservation("official final-minute average did not complete")
                        try:
                            raw = await asyncio.wait_for(socket.recv(), timeout=min(5.0, remaining))
                        except asyncio.TimeoutError:
                            continue
                        try:
                            self.process_frame(json.loads(raw), local_received_epoch=self.clock())
                        except OrderBookDesynchronized as exc:
                            # Reconnect to force Kalshi to send a new authoritative
                            # snapshot before any more deltas reach the strategy.
                            self.orderbook_resyncs += 1
                            self.last_rejection_reason = str(exc)
                            self._orderbook = LiveOrderBook()
                            LOGGER.warning("orderbook desynchronized; reconnecting: %s", exc)
                            break
            except asyncio.CancelledError:
                raise
            except UnsafeObservation:
                raise
            except Exception:
                if self._stop.is_set():
                    return
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.config.reconnect_max_seconds)

    def process_frame(self, frame: Dict[str, Any], *, local_received_epoch: float) -> bool:
        frame_type = frame.get("type")
        if frame_type == "cfbenchmarks_value":
            msg = frame.get("msg") or {}
            upstream_received = float(msg["received_at"]) / 1000.0
            offset_sample = local_received_epoch - upstream_received
            # Network transit is positive, so the smallest observed sample is
            # the safest estimate of the laptop's wall-clock offset.
            if abs(offset_sample) <= 300.0:
                self.server_clock_offset_seconds = (
                    offset_sample
                    if self.server_clock_offset_seconds is None
                    else min(self.server_clock_offset_seconds, offset_sample)
                )
        aligned_received_epoch = local_received_epoch
        if self.server_clock_offset_seconds is not None:
            aligned_received_epoch -= self.server_clock_offset_seconds
        seconds_remaining = max(0.0, self.config.expiration_epoch - aligned_received_epoch)
        if frame_type == "cfbenchmarks_value":
            item = parse_cfbenchmarks_value(
                frame, ticker=self.config.ticker, target=self.config.target,
                seconds_remaining=seconds_remaining, local_received_epoch=aligned_received_epoch,
            )
        elif frame_type == "ticker":
            item = parse_market_ticker(
                frame, ticker=self.config.ticker, target=self.config.target,
                seconds_remaining=seconds_remaining, received_epoch=aligned_received_epoch,
            )
        elif frame_type == "orderbook_snapshot":
            self._orderbook.load_snapshot(frame)
            self.recorder.reset_source_ordering("kalshi_orderbook")
            item = parse_orderbook_snapshot(
                frame, target=self.config.target, seconds_remaining=seconds_remaining,
                received_epoch=aligned_received_epoch,
            )
        elif frame_type == "orderbook_delta":
            event_epoch = self._orderbook.apply_delta(frame)
            yes_bid, yes_ask, no_bid, no_ask = self._orderbook.top()
            item = MarketObservation(
                received_epoch=aligned_received_epoch,
                event_epoch=event_epoch,
                upstream_received_epoch=None,
                source="kalshi_orderbook",
                ticker=self.config.ticker,
                target=self.config.target,
                seconds_remaining=seconds_remaining,
                yes_bid_cents=yes_bid,
                yes_ask_cents=yes_ask,
                no_bid_cents=no_bid,
                no_ask_cents=no_ask,
                sequence=int(frame["seq"]),
            )
        elif frame_type in {"subscribed", "ok"}:
            return False
        elif frame_type == "error":
            raise UnsafeObservation(f"Kalshi WebSocket error: {frame.get('msg')}")
        else:
            return False
        try:
            self.recorder.record(item)
        except UnsafeObservation as exc:
            # A single stale or timestamp-reordered observation is not a reason
            # to kill a read-only capture.  It must never reach the strategy,
            # but the listener can safely keep waiting for the next frame.
            self.dropped_observations += 1
            self.last_rejection_reason = str(exc)
            upstream_age = (
                None
                if item.upstream_received_epoch is None
                else item.upstream_received_epoch - item.event_epoch
            )
            LOGGER.warning(
                "dropped unsafe market observation: source=%s receive_age=%.3fs "
                "upstream_age=%s reason=%s",
                item.source,
                item.received_epoch - item.event_epoch,
                "n/a" if upstream_age is None else f"{upstream_age:.3f}s",
                exc,
            )
            return False
        if self.on_observation is not None:
            self.on_observation(item)
        if (item.official_average_window_size is not None
                and item.official_average_window_size >= 60):
            self._stop.set()
        return True
