import sys
import types
import unittest
from unittest.mock import patch

from pipeline.compat import _install_unavailable_torchvision_stub, ensure_torchaudio_metadata_safe


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
