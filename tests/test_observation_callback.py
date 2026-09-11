import tempfile
import unittest
from pathlib import Path

from src.auto.market_recorder import JsonlMarketRecorder
from src.auto.observation_listener import ObservationListener, ObservationListenerConfig


class ObservationCallbackTests(unittest.TestCase):
    def test_validated_observation_is_forwarded_after_recording(self):
        seen = []
        with tempfile.TemporaryDirectory() as temp_dir:
            listener = ObservationListener(
                config=ObservationListenerConfig(
                    ticker="KXBTC15M-TEST",
                    target=68_001.0,
                    expiration_epoch=1_710_000_060.0,
                    api_key_id="key-id",
                ),
                recorder=JsonlMarketRecorder(str(Path(temp_dir) / "observations.jsonl")),
                sign_request=lambda *_: "signature",
                clock=lambda: 1_710_000_000.2,
                on_observation=seen.append,
            )
            frame = {
                "type": "cfbenchmarks_value",
                "seq": 42,
                "msg": {
                    "index_id": "BRTI",
                    "received_at": 1_710_000_000_141,
                    "data": '{"time":1710000000123,"value":"68000.12"}',
                },
            }
            self.assertTrue(listener.process_frame(frame, local_received_epoch=1_710_000_000.2))
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0].ticker, "KXBTC15M-TEST")
            self.assertTrue(listener.recorder.path.exists())


if __name__ == "__main__":
    unittest.main()
