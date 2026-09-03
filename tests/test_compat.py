import sys
import types
import unittest
from unittest.mock import patch

from pipeline.compat import (
    _install_unavailable_torchvision_stub,
    _patch_constructor_unsupported_kwargs,
    ensure_huggingface_hub_token_safe,
    ensure_torchaudio_metadata_safe,
    pyannote_model_context,
)


class CompatTests(unittest.TestCase):
    def test_torchvision_stub_allows_optional_imports_to_resolve(self):
        original_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "torchvision" or name.startswith("torchvision.")
        }
        try:
            _install_unavailable_torchvision_stub(AttributeError("broken torchvision"))

            import torchvision
            from torchvision import transforms
            from torchvision.models import resnet50

            self.assertIs(torchvision.transforms, transforms)
            self.assertFalse(torchvision.extension._has_ops())
            with self.assertRaisesRegex(RuntimeError, "broken torchvision"):
                transforms.Compose()
            with self.assertRaisesRegex(RuntimeError, "broken torchvision"):
                resnet50()
        finally:
            for name in list(sys.modules):
                if name == "torchvision" or name.startswith("torchvision."):
                    sys.modules.pop(name, None)
            sys.modules.update(original_modules)

    def test_torchaudio_metadata_is_restored_when_missing(self):
        fake_torchaudio = types.SimpleNamespace()

        with patch.dict(sys.modules, {"torchaudio": fake_torchaudio}):
            ensure_torchaudio_metadata_safe()
            metadata = fake_torchaudio.AudioMetaData(
                sample_rate=16000,
                num_frames=32000,
                num_channels=1,
                bits_per_sample=16,
                encoding="PCM_S",
            )

        self.assertEqual(metadata.sample_rate, 16000)
        self.assertEqual(metadata.num_frames, 32000)
        self.assertEqual(fake_torchaudio.list_audio_backends(), ["soundfile"])

    def test_huggingface_download_accepts_use_auth_token_alias(self):
        calls = []

        def fake_download(*args, **kwargs):
            calls.append((args, kwargs))
            return "ok"

        fake_hub = types.ModuleType("huggingface_hub")
        fake_file_download = types.ModuleType("huggingface_hub.file_download")
        fake_hub.hf_hub_download = fake_download
        fake_file_download.hf_hub_download = fake_download
        fake_hub.file_download = fake_file_download

        with patch.dict(
            sys.modules,
            {
                "huggingface_hub": fake_hub,
                "huggingface_hub.file_download": fake_file_download,
            },
        ):
            ensure_huggingface_hub_token_safe()
            self.assertEqual(fake_hub.hf_hub_download("repo", use_auth_token="hf_x"), "ok")
            self.assertEqual(fake_file_download.hf_hub_download("repo", token="hf_y"), "ok")

        self.assertEqual(calls[0][1], {"token": "hf_x"})
        self.assertEqual(calls[1][1], {"token": "hf_y"})

    def test_huggingface_download_rewrites_model_subfolder_placeholder(self):
        calls = []

        def fake_download(*args, **kwargs):
            calls.append((args, kwargs))
            return "ok"

        fake_hub = types.ModuleType("huggingface_hub")
        fake_file_download = types.ModuleType("huggingface_hub.file_download")
        fake_hub.hf_hub_download = fake_download
        fake_file_download.hf_hub_download = fake_download
        fake_hub.file_download = fake_file_download

        with patch.dict(
            sys.modules,
            {
                "huggingface_hub": fake_hub,
                "huggingface_hub.file_download": fake_file_download,
            },
        ):
            ensure_huggingface_hub_token_safe()
            with pyannote_model_context("pyannote/speaker-diarization-community-1"):
                self.assertEqual(fake_hub.hf_hub_download("$model/segmentation", filename="pytorch_model.bin"), "ok")
                self.assertEqual(
                    fake_file_download.hf_hub_download(repo_id="$model/embedding", filename="pytorch_model.bin"),
                    "ok",
                )

        self.assertEqual(calls[0][0], ("pyannote/speaker-diarization-community-1",))
        self.assertEqual(calls[0][1]["subfolder"], "segmentation")
        self.assertEqual(calls[1][1]["repo_id"], "pyannote/speaker-diarization-community-1")
        self.assertEqual(calls[1][1]["subfolder"], "embedding")

    def test_constructor_patch_drops_unsupported_kwargs(self):
        calls = []

        class FakeSpeakerDiarization:
            def __init__(self, segmentation=None):
                calls.append(segmentation)

        _patch_constructor_unsupported_kwargs(FakeSpeakerDiarization)
        FakeSpeakerDiarization(segmentation="model", plda="unsupported")

        self.assertEqual(calls, ["model"])
