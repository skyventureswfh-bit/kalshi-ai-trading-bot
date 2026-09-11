import unittest

from src.auto.market_recorder import MarketObservation
from src.auto.wave_paper import WavePaperSession
from src.auto.wave_strategy import WaveAction


def _brti(epoch, value, remaining=600.0):
    return MarketObservation(
        received_epoch=epoch,
        event_epoch=epoch,
        upstream_received_epoch=epoch,
        source="kalshi_cfbenchmarks_brti",
        ticker="BTC-15",
        target=77_100.0,
        seconds_remaining=remaining,
        brti=value,
    )


def _book(epoch, yes_bid, yes_ask, remaining=600.0):
    return MarketObservation(
        received_epoch=epoch,
        event_epoch=epoch,
        upstream_received_epoch=None,
        source="kalshi_orderbook",
        ticker="BTC-15",
        target=77_100.0,
        seconds_remaining=remaining,
        yes_bid_cents=yes_bid,
        yes_ask_cents=yes_ask,
        no_bid_cents=100.0 - yes_ask,
        no_ask_cents=100.0 - yes_bid,
    )


class WavePaperSessionTests(unittest.TestCase):
    def _feed_clean_up_wave(self, session):
        event = None
        for index in range(5):
            epoch = 1000.0 + index * 3.0
            self.assertIsNone(session.on_observation(_brti(epoch, 77_090.0 + index * 5.0)))
            event = session.on_observation(_book(epoch + 0.1, 50.0 + index * 2.0, 52.0 + index * 2.0))
        return event

    def test_synchronizes_feed_and_paper_fills_entry(self):
        session = WavePaperSession()
        event = self._feed_clean_up_wave(session)
        self.assertIsNotNone(event)
        self.assertEqual(event.decision.action, WaveAction.ENTER_UP)
        self.assertEqual(event.position.contracts, 16)
        self.assertEqual(event.position.entry_price_cents, 60.0)

    def test_stale_cross_source_pair_is_rejected(self):
        session = WavePaperSession(max_cross_source_skew_seconds=1.0)
        session.on_observation(_brti(1000.0, 77_100.0))
        self.assertIsNone(session.on_observation(_book(1002.0, 58.0, 60.0)))

    def test_paper_exit_updates_scorecard_and_market_stays_locked(self):
        session = WavePaperSession()
        self._feed_clean_up_wave(session)
        session.on_observation(_brti(1015.0, 77_115.0))
        event = session.on_observation(_book(1015.1, 89.0, 90.0))
        self.assertEqual(event.decision.action, WaveAction.EXIT_PROFIT)
        self.assertAlmostEqual(event.result.gross_pnl_dollars, 4.64)
        self.assertAlmostEqual(event.result.net_pnl_dollars, 4.26)
        self.assertEqual(session.scorecard.trades, 1)
        self.assertEqual(session.scorecard.wins, 1)
        # A later clean signal cannot enter the same ticker again.
        session.on_observation(_brti(1018.0, 77_120.0))
        locked = session.on_observation(_book(1018.1, 58.0, 60.0))
        self.assertEqual(locked.decision.action, WaveAction.SKIP_USED)


if __name__ == "__main__":
    unittest.main()
