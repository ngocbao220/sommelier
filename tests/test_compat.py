import sys
import unittest

from pipeline.compat import _install_unavailable_torchvision_stub


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
