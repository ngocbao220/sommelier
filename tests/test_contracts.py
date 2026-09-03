import json
import unittest

from pipeline.contracts import PhaseResult, Segment


class ContractTests(unittest.TestCase):
    def test_segment_serializes_without_audio_payload(self):
        segment = Segment(
            index="00001",
            start=1.23456,
            end=2.0,
            speaker="SPEAKER_00",
            source_phase="vad",
            flags={"sepreformer": True},
        )
        payload = segment.to_json()
        json.dumps(payload)
        self.assertEqual(payload["start"], 1.235)
        self.assertNotIn("enhanced_audio", payload)

    def test_phase_result_serializes(self):
        result = PhaseResult(
            phase="02_vad_silero",
            enabled=True,
            model_info={"backend": "silero"},
            started_at="2026-01-01T00:00:00Z",
            ended_at="2026-01-01T00:00:01Z",
            duration_seconds=1.0,
        )
        payload = result.to_json()
        json.dumps(payload)
        self.assertEqual(payload["phase"], "02_vad_silero")
