import json
import tempfile
import unittest
from pathlib import Path

from src.auto.wave_strategy import (
    WaveAction,
    WaveConfig,
    WaveJournal,
    WaveStrategy,
    WaveTick,
    WaveTradeResult,
    build_scorecard,
)


def _history(*, ticker="BTC-15", direction=1, ask=60.0, spread=2.0):
    rows = []
    for index in range(5):
        yes_mid = 52.0 + direction * index * 2.0
        no_mid = 100.0 - yes_mid
        if index == 4:
            if direction > 0:
                yes_mid = ask - spread / 2.0
            else:
                no_mid = ask - spread / 2.0
        rows.append(
            WaveTick(
                ticker=ticker,
                event_epoch=1000.0 + index * 3.0,
                seconds_remaining=600.0 - index * 3.0,
                target=77_100.0,
                underlying=77_090.0 + direction * index * 5.0,
                yes_bid_cents=yes_mid - spread / 2.0,
                yes_ask_cents=yes_mid + spread / 2.0,
                no_bid_cents=no_mid - spread / 2.0,
                no_ask_cents=no_mid + spread / 2.0,
            )
        )
    return rows


def _current(history, *, yes_bid=None, yes_ask=None, no_bid=None, no_ask=None):
    last = history[-1]
    return WaveTick(
        ticker=last.ticker,
        event_epoch=last.event_epoch + 3.0,
        seconds_remaining=last.seconds_remaining - 3.0,
        target=last.target,
        underlying=last.underlying,
        yes_bid_cents=last.yes_bid_cents if yes_bid is None else yes_bid,
        yes_ask_cents=last.yes_ask_cents if yes_ask is None else yes_ask,
        no_bid_cents=last.no_bid_cents if no_bid is None else no_bid,
        no_ask_cents=last.no_ask_cents if no_ask is None else no_ask,
    )


class WaveStrategyTests(unittest.TestCase):
    def test_clean_wave_enters_correct_side_at_executable_60(self):
        for direction, action, side in (
            (1, WaveAction.ENTER_UP, "UP"),
            (-1, WaveAction.ENTER_DOWN, "DOWN"),
        ):
            with self.subTest(side=side):
                strategy = WaveStrategy()
                decision = strategy.evaluate(_history(direction=direction, ask=60.0))
                self.assertEqual(decision.action, action)
                self.assertEqual(decision.side, side)
                self.assertEqual(decision.executable_price_cents, 60.0)
                self.assertEqual(decision.contracts, 16)
                self.assertGreaterEqual(decision.formula_score, strategy.config.min_formula_score)

    def test_does_not_enter_outside_locked_58_to_62_band(self):
        decision = WaveStrategy().evaluate(_history(ask=64.0))
        self.assertEqual(decision.action, WaveAction.WAIT)
        self.assertIn("entry band", decision.reason)

    def test_needs_real_timed_confirmation_not_first_screen_guess(self):
        decision = WaveStrategy().evaluate(_history()[:4])
        self.assertEqual(decision.action, WaveAction.WAIT)
        self.assertIn("samples", decision.reason)

    def test_jeff_factor_cannot_override_hard_spread_guard(self):
        decision = WaveStrategy().evaluate(_history(ask=60.0, spread=6.0))
        self.assertEqual(decision.action, WaveAction.WAIT)
        self.assertIn("spread", decision.reason)

    def test_one_trade_per_market_survives_exit(self):
        strategy = WaveStrategy()
        history = _history()
        strategy.confirm_entry(strategy.evaluate(history), history[-1].ticker)
        exit_tick = _current(history, yes_bid=90.0, yes_ask=91.0)
        self.assertEqual(strategy.evaluate([exit_tick]).action, WaveAction.EXIT_PROFIT)
        strategy.confirm_exit(history[-1].ticker)
        self.assertEqual(strategy.evaluate(history).action, WaveAction.SKIP_USED)

    def test_profit_exit_uses_bid_not_displayed_probability(self):
        strategy = WaveStrategy()
        history = _history(ask=60.0)
        strategy.confirm_entry(strategy.evaluate(history), history[-1].ticker)
        decision = strategy.evaluate([_current(history, yes_bid=89.0, yes_ask=93.0)])
        self.assertEqual(decision.action, WaveAction.EXIT_PROFIT)
        self.assertAlmostEqual(decision.marked_pnl_dollars, 4.26)

    def test_three_dollar_loss_seatbelt_uses_actual_position_value(self):
        strategy = WaveStrategy()
        history = _history(ask=60.0)
        strategy.confirm_entry(strategy.evaluate(history), history[-1].ticker)
        # Fees are included, so the stop fires before contract value alone loses $3.
        decision = strategy.evaluate([_current(history, yes_bid=44.0, yes_ask=46.0)])
        self.assertEqual(decision.action, WaveAction.EXIT_STOP)
        self.assertLessEqual(decision.marked_pnl_dollars, -3.0)

    def test_entry_cap_includes_estimated_fee(self):
        strategy = WaveStrategy()
        decision = strategy.evaluate(_history(ask=58.0))
        position = strategy.confirm_entry(decision, "BTC-15")
        self.assertLessEqual(position.cost_dollars, 10.0)
        self.assertEqual(position.contracts, 16)

    def test_time_rail_exits_instead_of_drifting_to_settlement(self):
        strategy = WaveStrategy()
        history = _history(ask=60.0)
        strategy.confirm_entry(strategy.evaluate(history), history[-1].ticker)
        last = history[-1]
        time_tick = WaveTick(
            ticker=last.ticker,
            event_epoch=last.event_epoch + 3.0,
            seconds_remaining=45.0,
            target=last.target,
            underlying=last.underlying,
            yes_bid_cents=65.0,
            yes_ask_cents=67.0,
            no_bid_cents=33.0,
            no_ask_cents=35.0,
        )
        self.assertEqual(strategy.evaluate([time_tick]).action, WaveAction.EXIT_TIME)

    def test_market_lock_can_be_restored_after_restart(self):
        strategy = WaveStrategy()
        strategy.restore_used_markets(["BTC-15"])
        self.assertEqual(strategy.evaluate(_history()).action, WaveAction.SKIP_USED)

    def test_journal_is_append_only_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            journal_path = Path(temp_dir) / "wave.jsonl"
            decision = WaveStrategy().evaluate(_history())
            journal = WaveJournal(str(journal_path))
            journal.append(event="decision", ticker="BTC-15", payload=decision)
            journal.append(event="decision", ticker="BTC-15", payload=decision)
            rows = [json.loads(line) for line in journal_path.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["payload"]["action"], "ENTER_UP")

    def test_scorecard_uses_closed_trade_net_results(self):
        results = [
            WaveTradeResult("A", "UP", 4.50, fees_dollars=0.40, slippage_dollars=0.10),
            WaveTradeResult("B", "DOWN", -2.70, fees_dollars=0.20, slippage_dollars=0.10),
            WaveTradeResult("C", "UP", 4.20, fees_dollars=0.20),
        ]
        scorecard = build_scorecard(results)
        self.assertEqual(scorecard.trades, 3)
        self.assertEqual(scorecard.wins, 2)
        self.assertEqual(scorecard.losses, 1)
        self.assertAlmostEqual(scorecard.win_rate, 2 / 3)
        self.assertAlmostEqual(scorecard.average_net_win_dollars, 4.0)
        self.assertAlmostEqual(scorecard.average_net_loss_dollars, -3.0)
        self.assertAlmostEqual(scorecard.net_pnl_dollars, 5.0)


if __name__ == "__main__":
    unittest.main()
