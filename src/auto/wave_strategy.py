"""Deterministic BTC 15-minute "catch the wave" decision core.

This module is deliberately exchange-agnostic.  It decides; it does not place
orders.  The production adapter must pass real, executable asks for entry and
bids for exit.  Keeping the strategy pure makes replay and paper validation
possible before the live client is allowed to use it.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from math import floor, isfinite
from pathlib import Path
from typing import Iterable, Optional, Sequence

from src.utils.trade_pricing import estimate_kalshi_fee


class WaveAction(str, Enum):
    WAIT = "WAIT"
    ENTER_UP = "ENTER_UP"
    ENTER_DOWN = "ENTER_DOWN"
    HOLD = "HOLD"
    EXIT_PROFIT = "EXIT_PROFIT"
    EXIT_STOP = "EXIT_STOP"
    EXIT_TIME = "EXIT_TIME"
    SKIP_USED = "SKIP_USED"


@dataclass(frozen=True)
class WaveConfig:
    """Locked baseline. Prices are cents per contract; money is dollars."""

    capital_per_trade: float = 10.0
    entry_min_cents: float = 58.0
    entry_max_cents: float = 62.0
    take_profit_cents: float = 89.0
    max_loss_dollars: float = 3.0
    min_samples: int = 5
    min_sample_span_seconds: float = 12.0
    min_seconds_remaining: float = 45.0
    max_spread_cents: float = 5.0
    min_formula_score: float = 0.66
    min_combined_score: float = 0.68
    formula_weight: float = 0.90
    jeff_weight: float = 0.10

    def __post_init__(self) -> None:
        if self.capital_per_trade <= 0 or self.max_loss_dollars <= 0:
            raise ValueError("capital and max loss must be positive")
        if not 0 < self.entry_min_cents <= self.entry_max_cents < self.take_profit_cents <= 100:
            raise ValueError("entry and take-profit prices are inconsistent")
        if self.max_loss_dollars >= self.capital_per_trade:
            raise ValueError("max loss must be smaller than capital per trade")
        if self.min_samples < 3 or self.min_sample_span_seconds <= 0:
            raise ValueError("momentum confirmation requires at least three timed samples")
        if abs(self.formula_weight + self.jeff_weight - 1.0) > 1e-9:
            raise ValueError("formula and Jeff weights must total 1.0")


@dataclass(frozen=True)
class WaveTick:
    ticker: str
    event_epoch: float
    seconds_remaining: float
    target: float
    underlying: float
    yes_bid_cents: float
    yes_ask_cents: float
    no_bid_cents: float
    no_ask_cents: float

    def __post_init__(self) -> None:
        numeric = (
            self.event_epoch,
            self.seconds_remaining,
            self.target,
            self.underlying,
            self.yes_bid_cents,
            self.yes_ask_cents,
            self.no_bid_cents,
            self.no_ask_cents,
        )
        if not self.ticker or not all(isfinite(value) for value in numeric):
            raise ValueError("wave tick contains missing or non-finite data")
        if self.seconds_remaining < 0:
            raise ValueError("seconds remaining cannot be negative")
        for bid, ask in (
            (self.yes_bid_cents, self.yes_ask_cents),
            (self.no_bid_cents, self.no_ask_cents),
        ):
            if not 0 <= bid <= ask <= 100:
                raise ValueError("invalid executable bid/ask")


@dataclass(frozen=True)
class WavePosition:
    ticker: str
    side: str
    contracts: int
    entry_price_cents: float
    cost_dollars: float
    entry_fee_dollars: float


@dataclass(frozen=True)
class WaveDecision:
    action: WaveAction
    reason: str
    side: Optional[str] = None
    contracts: int = 0
    executable_price_cents: Optional[float] = None
    formula_score: Optional[float] = None
    jeff_score: Optional[float] = None
    combined_score: Optional[float] = None
    marked_pnl_dollars: Optional[float] = None
    estimated_fee_dollars: Optional[float] = None


@dataclass(frozen=True)
class WaveTradeResult:
    ticker: str
    side: str
    gross_pnl_dollars: float
    fees_dollars: float = 0.0
    slippage_dollars: float = 0.0

    @property
    def net_pnl_dollars(self) -> float:
        return self.gross_pnl_dollars - self.fees_dollars - self.slippage_dollars


@dataclass(frozen=True)
class WaveScorecard:
    trades: int
    wins: int
    losses: int
    win_rate: float
    average_net_win_dollars: float
    average_net_loss_dollars: float
    total_fees_dollars: float
    total_slippage_dollars: float
    net_pnl_dollars: float


def build_scorecard(results: Sequence[WaveTradeResult]) -> WaveScorecard:
    """Summarize confirmed closed trades; break-even trades are not wins."""
    net = [result.net_pnl_dollars for result in results]
    wins = [value for value in net if value > 0]
    losses = [value for value in net if value < 0]
    trades = len(results)
    return WaveScorecard(
        trades=trades,
        wins=len(wins),
        losses=len(losses),
        win_rate=(len(wins) / trades) if trades else 0.0,
        average_net_win_dollars=(sum(wins) / len(wins)) if wins else 0.0,
        average_net_loss_dollars=(sum(losses) / len(losses)) if losses else 0.0,
        total_fees_dollars=sum(result.fees_dollars for result in results),
        total_slippage_dollars=sum(result.slippage_dollars for result in results),
        net_pnl_dollars=sum(net),
    )


def _mid(tick: WaveTick, side: str) -> float:
    if side == "UP":
        return (tick.yes_bid_cents + tick.yes_ask_cents) / 2.0
    return (tick.no_bid_cents + tick.no_ask_cents) / 2.0


def _ask(tick: WaveTick, side: str) -> float:
    return tick.yes_ask_cents if side == "UP" else tick.no_ask_cents


def _bid(tick: WaveTick, side: str) -> float:
    return tick.yes_bid_cents if side == "UP" else tick.no_bid_cents


def _spread(tick: WaveTick, side: str) -> float:
    return _ask(tick, side) - _bid(tick, side)


def _efficiency(values: Sequence[float]) -> float:
    """Net movement divided by path length: 1 is a clean one-way wave."""
    path = sum(abs(right - left) for left, right in zip(values, values[1:]))
    return 0.0 if path <= 0 else min(1.0, abs(values[-1] - values[0]) / path)


def _direction_consistency(values: Sequence[float], sign: int) -> float:
    deltas = [right - left for left, right in zip(values, values[1:])]
    moving = [delta for delta in deltas if abs(delta) > 1e-9]
    if not moving:
        return 0.0
    aligned = sum(1 for delta in moving if delta * sign > 0)
    return aligned / len(moving)


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


class WaveStrategy:
    """Stateful one-trade-per-market Wave engine."""

    def __init__(self, config: Optional[WaveConfig] = None) -> None:
        self.config = config or WaveConfig()
        self._used_markets: set[str] = set()
        self._positions: dict[str, WavePosition] = {}

    def restore_used_markets(self, tickers: Iterable[str]) -> None:
        """Restore persisted market locks after a restart."""
        self._used_markets.update(str(ticker) for ticker in tickers if ticker)

    def position_for(self, ticker: str) -> Optional[WavePosition]:
        return self._positions.get(ticker)

    def evaluate(self, history: Sequence[WaveTick]) -> WaveDecision:
        if not history:
            return WaveDecision(WaveAction.WAIT, "no observations")
        ticker = history[-1].ticker
        if any(tick.ticker != ticker for tick in history):
            return WaveDecision(WaveAction.WAIT, "mixed-market history rejected")
        if any(right.event_epoch <= left.event_epoch for left, right in zip(history, history[1:])):
            return WaveDecision(WaveAction.WAIT, "out-of-order or duplicate observations")

        current = history[-1]
        position = self._positions.get(ticker)
        if position is not None:
            return self._manage_position(position, current)
        if ticker in self._used_markets:
            return WaveDecision(WaveAction.SKIP_USED, "one-trade market lock already consumed")
        return self._evaluate_entry(history)

    def confirm_entry(self, decision: WaveDecision, ticker: str) -> WavePosition:
        """Commit an entry only after the execution adapter confirms a fill."""
        if decision.action not in (WaveAction.ENTER_UP, WaveAction.ENTER_DOWN):
            raise ValueError("decision is not an entry")
        side = "UP" if decision.action == WaveAction.ENTER_UP else "DOWN"
        if ticker in self._used_markets or ticker in self._positions:
            raise ValueError("market has already been traded")
        if not decision.contracts or decision.executable_price_cents is None:
            raise ValueError("entry decision is incomplete")
        entry_fee = estimate_kalshi_fee(
            decision.executable_price_cents / 100.0,
            decision.contracts,
            maker=False,
        )
        cost = decision.contracts * decision.executable_price_cents / 100.0 + entry_fee
        position = WavePosition(
            ticker=ticker,
            side=side,
            contracts=decision.contracts,
            entry_price_cents=decision.executable_price_cents,
            cost_dollars=cost,
            entry_fee_dollars=entry_fee,
        )
        self._positions[ticker] = position
        self._used_markets.add(ticker)
        return position

    def confirm_exit(self, ticker: str) -> WavePosition:
        """Remove a position after the execution adapter confirms the sell."""
        try:
            return self._positions.pop(ticker)
        except KeyError as exc:
            raise ValueError("no open Wave position for market") from exc

    def _evaluate_entry(self, history: Sequence[WaveTick]) -> WaveDecision:
        cfg = self.config
        current = history[-1]
        if current.seconds_remaining < cfg.min_seconds_remaining:
            return WaveDecision(WaveAction.WAIT, "too little time remains for a controlled exit")
        if len(history) < cfg.min_samples:
            return WaveDecision(WaveAction.WAIT, "collecting momentum samples")
        recent = list(history[-max(cfg.min_samples, 8):])
        if recent[-1].event_epoch - recent[0].event_epoch < cfg.min_sample_span_seconds:
            return WaveDecision(WaveAction.WAIT, "momentum sample span is too short")

        underlying = [tick.underlying for tick in recent]
        net_move = underlying[-1] - underlying[0]
        if abs(net_move) < 1e-9:
            return WaveDecision(WaveAction.WAIT, "underlying has no directional movement")
        sign = 1 if net_move > 0 else -1
        side = "UP" if sign > 0 else "DOWN"
        side_mids = [_mid(tick, side) for tick in recent]
        if side_mids[-1] <= side_mids[0]:
            return WaveDecision(WaveAction.WAIT, "contract price does not confirm underlying direction", side=side)

        ask = _ask(current, side)
        spread = _spread(current, side)
        if not cfg.entry_min_cents <= ask <= cfg.entry_max_cents:
            return WaveDecision(
                WaveAction.WAIT,
                f"confirmed {side} wave, waiting for executable ask in entry band",
                side=side,
                executable_price_cents=ask,
            )
        if spread > cfg.max_spread_cents:
            return WaveDecision(WaveAction.WAIT, "spread is too wide for a controlled exit", side=side)

        underlying_efficiency = _efficiency(underlying)
        contract_efficiency = _efficiency(side_mids)
        consistency = _direction_consistency(underlying, sign)
        distance = [tick.underlying - tick.target for tick in recent]
        moving_away = _direction_consistency(distance, sign)
        formula = _bounded(
            0.35 * underlying_efficiency
            + 0.25 * contract_efficiency
            + 0.25 * consistency
            + 0.15 * moving_away
        )

        # "Jeff Factor": a small common-sense tie-breaker for a clean,
        # executable, not-yet-overpriced wave. It may reject a marginal setup;
        # it never overrides the formula, price band, spread, or risk limits.
        spread_quality = _bounded(1.0 - spread / cfg.max_spread_cents)
        entry_room = _bounded(
            (cfg.entry_max_cents - ask)
            / max(1e-9, cfg.entry_max_cents - cfg.entry_min_cents)
        )
        cleanliness = (underlying_efficiency + contract_efficiency + consistency) / 3.0
        jeff = _bounded(0.60 * cleanliness + 0.25 * spread_quality + 0.15 * entry_room)
        combined = cfg.formula_weight * formula + cfg.jeff_weight * jeff

        if formula < cfg.min_formula_score or combined < cfg.min_combined_score:
            return WaveDecision(
                WaveAction.WAIT,
                "direction exists but the wave is not clean enough",
                side=side,
                executable_price_cents=ask,
                formula_score=formula,
                jeff_score=jeff,
                combined_score=combined,
            )

        contracts = floor(cfg.capital_per_trade * 100.0 / ask)
        while contracts > 0:
            entry_fee = estimate_kalshi_fee(ask / 100.0, contracts, maker=False)
            if contracts * ask / 100.0 + entry_fee <= cfg.capital_per_trade + 1e-9:
                break
            contracts -= 1
        if contracts < 1:
            return WaveDecision(WaveAction.WAIT, "capital cannot buy one contract", side=side)
        action = WaveAction.ENTER_UP if side == "UP" else WaveAction.ENTER_DOWN
        return WaveDecision(
            action,
            "momentum confirmed; executable entry is inside the locked band",
            side=side,
            contracts=contracts,
            executable_price_cents=ask,
            formula_score=formula,
            jeff_score=jeff,
            combined_score=combined,
            estimated_fee_dollars=entry_fee,
        )

    def _manage_position(self, position: WavePosition, current: WaveTick) -> WaveDecision:
        bid = _bid(current, position.side)
        liquid_value = position.contracts * bid / 100.0
        exit_fee = estimate_kalshi_fee(bid / 100.0, position.contracts, maker=False)
        pnl = liquid_value - position.cost_dollars - exit_fee
        if bid >= self.config.take_profit_cents:
            return WaveDecision(
                WaveAction.EXIT_PROFIT,
                "executable bid reached the locked profit target",
                side=position.side,
                contracts=position.contracts,
                executable_price_cents=bid,
                marked_pnl_dollars=pnl,
                estimated_fee_dollars=exit_fee,
            )
        if pnl <= -self.config.max_loss_dollars:
            return WaveDecision(
                WaveAction.EXIT_STOP,
                "executable liquidation value reached the loss seatbelt",
                side=position.side,
                contracts=position.contracts,
                executable_price_cents=bid,
                marked_pnl_dollars=pnl,
                estimated_fee_dollars=exit_fee,
            )
        if current.seconds_remaining <= self.config.min_seconds_remaining:
            return WaveDecision(
                WaveAction.EXIT_TIME,
                "time rail reached; do not drift into settlement",
                side=position.side,
                contracts=position.contracts,
                executable_price_cents=bid,
                marked_pnl_dollars=pnl,
                estimated_fee_dollars=exit_fee,
            )
        return WaveDecision(
            WaveAction.HOLD,
            "position remains between the locked exit rails",
            side=position.side,
            contracts=position.contracts,
            executable_price_cents=bid,
            marked_pnl_dollars=pnl,
            estimated_fee_dollars=exit_fee,
        )


class WaveJournal:
    """Append-only audit journal for decisions and confirmed fills."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def append(self, *, event: str, ticker: str, payload: object) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = asdict(payload) if hasattr(payload, "__dataclass_fields__") else payload
        row = {"schema_version": 1, "event": event, "ticker": ticker, "payload": body}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
