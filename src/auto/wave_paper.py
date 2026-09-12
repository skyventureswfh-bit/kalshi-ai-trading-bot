"""Paper-forward adapter connecting synchronized observations to WaveStrategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.auto.market_recorder import MarketObservation
from src.auto.wave_strategy import (
    WaveAction,
    WaveDecision,
    WaveJournal,
    WavePosition,
    WaveScorecard,
    WaveStrategy,
    WaveTick,
    WaveTradeResult,
    build_scorecard,
)


@dataclass(frozen=True)
class PaperWaveEvent:
    decision: WaveDecision
    position: Optional[WavePosition] = None
    result: Optional[WaveTradeResult] = None


class WavePaperSession:
    """Synchronize two read-only feeds and simulate immediate bid/ask fills.

    The session never calls a Kalshi trading endpoint. Entry fills use the
    executable ask carried by the decision; exits use the executable bid.
    """

    def __init__(
        self,
        *,
        strategy: Optional[WaveStrategy] = None,
        journal: Optional[WaveJournal] = None,
        max_cross_source_skew_seconds: float = 2.0,
    ) -> None:
        self.strategy = strategy or WaveStrategy()
        self.journal = journal
        self.max_cross_source_skew_seconds = max_cross_source_skew_seconds
        self._brti: Optional[MarketObservation] = None
        self._book: Optional[MarketObservation] = None
        self._history: list[WaveTick] = []
        self._results: list[WaveTradeResult] = []
        self._last_tick_epoch: Optional[float] = None

    @property
    def results(self) -> tuple[WaveTradeResult, ...]:
        return tuple(self._results)

    @property
    def scorecard(self) -> WaveScorecard:
        return build_scorecard(self._results)

    def on_observation(self, item: MarketObservation) -> Optional[PaperWaveEvent]:
        if item.source == "kalshi_cfbenchmarks_brti" and item.brti is not None:
            if self._brti is not None and item.event_epoch < self._brti.event_epoch:
                return None
            self._brti = item
        elif item.source in {"kalshi_ticker", "kalshi_orderbook"}:
            if self._book is not None and item.event_epoch < self._book.event_epoch:
                return None
            self._book = item
        else:
            return None

        tick = self._synchronized_tick()
        if tick is None:
            return None
        self._history.append(tick)
        decision = self.strategy.evaluate(self._history)
        return self._apply_paper_decision(decision, tick)

    def _synchronized_tick(self) -> Optional[WaveTick]:
        brti = self._brti
        book = self._book
        if brti is None or book is None:
            return None
        if brti.ticker != book.ticker or brti.target != book.target:
            return None
        if abs(brti.event_epoch - book.event_epoch) > self.max_cross_source_skew_seconds:
            return None
        quote_fields = (
            book.yes_bid_cents,
            book.yes_ask_cents,
            book.no_bid_cents,
            book.no_ask_cents,
        )
        if any(value is None for value in quote_fields):
            return None
        epoch = max(brti.event_epoch, book.event_epoch)
        if self._last_tick_epoch is not None and epoch <= self._last_tick_epoch:
            return None
        self._last_tick_epoch = epoch
        return WaveTick(
            ticker=brti.ticker,
            event_epoch=epoch,
            seconds_remaining=min(brti.seconds_remaining, book.seconds_remaining),
            target=brti.target,
            underlying=float(brti.brti),
            yes_bid_cents=float(book.yes_bid_cents),
            yes_ask_cents=float(book.yes_ask_cents),
            no_bid_cents=float(book.no_bid_cents),
            no_ask_cents=float(book.no_ask_cents),
        )

    def _apply_paper_decision(self, decision: WaveDecision, tick: WaveTick) -> PaperWaveEvent:
        position: Optional[WavePosition] = None
        result: Optional[WaveTradeResult] = None
        if decision.action in (WaveAction.ENTER_UP, WaveAction.ENTER_DOWN):
            position = self.strategy.confirm_entry(decision, tick.ticker)
            self._journal("paper_entry", tick.ticker, decision)
        elif decision.action in (
            WaveAction.EXIT_PROFIT,
            WaveAction.EXIT_STOP,
            WaveAction.EXIT_TIME,
        ):
            open_position = self.strategy.position_for(tick.ticker)
            if open_position is None or decision.executable_price_cents is None:
                raise RuntimeError("paper exit decision has no matching position or price")
            contract_basis = open_position.contracts * open_position.entry_price_cents / 100.0
            gross = open_position.contracts * decision.executable_price_cents / 100.0 - contract_basis
            result = WaveTradeResult(
                ticker=tick.ticker,
                side=open_position.side,
                gross_pnl_dollars=gross,
                fees_dollars=(
                    open_position.entry_fee_dollars
                    + float(decision.estimated_fee_dollars or 0.0)
                ),
            )
            position = self.strategy.confirm_exit(tick.ticker)
            self._results.append(result)
            self._journal("paper_exit", tick.ticker, decision)
            self._journal("paper_result", tick.ticker, result)
        else:
            self._journal("decision", tick.ticker, decision)
        return PaperWaveEvent(decision=decision, position=position, result=result)

    def _journal(self, event: str, ticker: str, payload: object) -> None:
        if self.journal is not None:
            self.journal.append(event=event, ticker=ticker, payload=payload)
