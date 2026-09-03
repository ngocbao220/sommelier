import os
import tempfile
import unittest
import json
from pathlib import Path

from pipeline.config import load_config, prepare_config, redact_model_config


class ConfigTests(unittest.TestCase):
    def test_defaults_merge_and_redact_token_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"input": {"path": "x.wav"}}), encoding="utf-8")
            old = os.environ.get("HUGGINGFACE_TOKEN")
            os.environ["HUGGINGFACE_TOKEN"] = "hf_fake"
            try:
                config = load_config(path)
                self.assertEqual(config["input"]["path"], "x.wav")
                self.assertEqual(config["runtime"]["mode"], "real")
                self.assertEqual(
                    config["models"]["diarization"]["model"],
                    "pyannote/speaker-diarization-community-1",
                )
                redacted = redact_model_config(config)
                self.assertEqual(redacted["models"]["diarization"]["token_status"], "present")
            finally:
                if old is None:
                    os.environ.pop("HUGGINGFACE_TOKEN", None)
                else:
                    os.environ["HUGGINGFACE_TOKEN"] = old

    def test_rejects_unknown_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps({"models": {"vad": {"backend": "other"}}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "silero"):
                load_config(path)

    def test_prepare_config_resolves_paths_from_config_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "samples"
            samples.mkdir()
            model_dir = root / "models" / "SepReformer"
            model_dir.mkdir(parents=True)
            path = root / "config.json"
            path.write_text(
                json.dumps({
                    "input": {
                        "path": "samples",
                        "output_root": "outputs",
                    },
                    "models": {
                        "overlap_separation": {
                            "path": "models/SepReformer",
                        },
                    },
                }),
                encoding="utf-8",
            )

            config = prepare_config(load_config(path))

            self.assertEqual(Path(config["input"]["path"]), samples.resolve())
            self.assertEqual(Path(config["input"]["output_root"]), (root / "outputs").resolve())
            self.assertEqual(
                Path(config["models"]["overlap_separation"]["path"]),
                model_dir.resolve(),
            )
