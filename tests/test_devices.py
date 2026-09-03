import sys
import types
import unittest
from unittest.mock import patch

from pipeline.io_utils import resolve_device


def fake_torch(cuda_available):
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
    )


class DeviceTests(unittest.TestCase):
    def test_auto_prefers_cuda_when_available(self):
        with patch.dict(sys.modules, {"torch": fake_torch(True)}):
            self.assertEqual(resolve_device("auto"), "cuda")

    def test_auto_falls_back_to_cpu_without_cuda(self):
        with patch.dict(sys.modules, {"torch": fake_torch(False)}):
            self.assertEqual(resolve_device("auto"), "cpu")

    def test_gpu_alias_falls_back_to_cpu_without_cuda(self):
        with patch.dict(sys.modules, {"torch": fake_torch(False)}):
            with self.assertWarnsRegex(RuntimeWarning, "falling back to CPU"):
                self.assertEqual(resolve_device("gpu"), "cpu")

    def test_explicit_cpu_is_preserved(self):
        self.assertEqual(resolve_device("cpu"), "cpu")
