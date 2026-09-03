import unittest

import numpy as np

from pipeline.diarization import _pipeline_from_pretrained, annotation_to_segments
from pipeline.overlap import detect_overlaps
from pipeline.sepreformer import SepReformerAdapter
from pipeline.vad import SileroVadAdapter
from pipeline.contracts import Segment


class FakeTurn:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class FakeAnnotation:
    def itertracks(self, yield_label=False):
        self.assert_yield_label = yield_label
        return [
            (FakeTurn(0.0, 1.0), "track", "SPEAKER_00"),
            (FakeTurn(0.5, 1.5), "track", "SPEAKER_01"),
        ]


class FakeDiarizeOutput:
    def __init__(self):
        self.speaker_diarization = FakeAnnotation()


class FakePipelineWithToken:
    calls = []

    @classmethod
    def from_pretrained(cls, model_name, token=None):
        cls.calls.append((model_name, token))
        return "pipeline"


class FakePipelineWithUseAuthToken:
    calls = []

    @classmethod
    def from_pretrained(cls, model_name, use_auth_token=None):
        cls.calls.append((model_name, use_auth_token))
        return "pipeline"


class AdapterTests(unittest.TestCase):
    def test_annotation_to_segments(self):
        segments = annotation_to_segments(FakeAnnotation())
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].speaker, "SPEAKER_00")
        self.assertEqual(segments[1].start, 0.5)

    def test_annotation_to_segments_accepts_pyannote_diarize_output(self):
        segments = annotation_to_segments(FakeDiarizeOutput())
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].speaker, "SPEAKER_00")

    def test_pipeline_from_pretrained_uses_supported_token_argument(self):
        FakePipelineWithToken.calls = []
        FakePipelineWithUseAuthToken.calls = []

        self.assertEqual(_pipeline_from_pretrained(FakePipelineWithToken, "model", "hf_x"), "pipeline")
        self.assertEqual(_pipeline_from_pretrained(FakePipelineWithUseAuthToken, "model", "hf_y"), "pipeline")

        self.assertEqual(FakePipelineWithToken.calls, [("model", "hf_x")])
        self.assertEqual(FakePipelineWithUseAuthToken.calls, [("model", "hf_y")])

    def test_detect_overlaps(self):
        overlaps = detect_overlaps(
            [
                Segment("00000", 0.0, 1.0, "SPEAKER_00"),
                Segment("00001", 0.5, 1.5, "SPEAKER_01"),
            ],
            threshold_seconds=0.2,
        )
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0]["overlap_duration"], 0.5)

    def test_vad_dry_run_covers_audio(self):
        audio = {"waveform": np.zeros(16000, dtype=np.float32), "sample_rate": 16000}
        adapter = SileroVadAdapter({"model": "silero_vad", "threshold": 0.35}, "cpu", "dry_run")
        segments = adapter.run(audio)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].end, 1.0)

    def test_sepreformer_dry_run_marks_overlaps(self):
        audio = {"waveform": np.zeros(32000, dtype=np.float32), "sample_rate": 16000}
        segments = [
            Segment("00000", 0.0, 1.0, "SPEAKER_00"),
            Segment("00001", 0.5, 1.5, "SPEAKER_01"),
        ]
        adapter = SepReformerAdapter({"overlap_threshold_seconds": 0.2}, "cpu", "dry_run")
        updated, enhanced, overlaps = adapter.run(audio, segments)
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(set(enhanced), {"00000", "00001"})
        self.assertTrue(updated[0].flags["sepreformer"])
