import tempfile
import unittest
import json
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

    def test_reordered_observation_is_dropped_without_reaching_strategy(self):
        seen = []
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "observations.jsonl"
            listener = ObservationListener(
                config=ObservationListenerConfig(
                    ticker="KXBTC15M-TEST",
                    target=68_001.0,
                    expiration_epoch=1_710_000_060.0,
                    api_key_id="key-id",
                ),
                recorder=JsonlMarketRecorder(str(path)),
                sign_request=lambda *_: "signature",
                clock=lambda: 1_710_000_000.2,
                on_observation=seen.append,
            )
            first = {
                "type": "cfbenchmarks_value",
                "seq": 42,
                "msg": {
                    "index_id": "BRTI",
                    "received_at": 1_710_000_000_141,
                    "data": json.dumps({"time": 1_710_000_000_123, "value": "68000.12"}),
                },
            }
            reordered = {
                "type": "cfbenchmarks_value",
                "seq": 43,
                "msg": {
                    "index_id": "BRTI",
                    "received_at": 1_710_000_000_191,
                    "data": json.dumps({"time": 1_710_000_000_100, "value": "68000.10"}),
                },
            }

            self.assertTrue(listener.process_frame(first, local_received_epoch=1_710_000_000.2))
            self.assertFalse(listener.process_frame(reordered, local_received_epoch=1_710_000_000.2))
            self.assertEqual(len(seen), 1)
            self.assertEqual(listener.dropped_observations, 1)
            self.assertEqual(listener.last_rejection_reason, "out-of-order source timestamp")
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
