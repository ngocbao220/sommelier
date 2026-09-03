from __future__ import annotations

import importlib.machinery
import sys
import types
from collections import namedtuple


class _UnavailableObject:
    def __init__(self, original_error: Exception) -> None:
        self._original_error = original_error

    def __call__(self, *_args: object, **_kwargs: object) -> object:
        raise RuntimeError(
            "torchvision is installed but cannot be imported in this environment; "
            f"original error: {self._original_error}"
        )

    def __getattr__(self, _attr: str) -> "_UnavailableObject":
        return self


def ensure_pyannote_import_safe() -> None:
    ensure_torchvision_import_safe()
    ensure_torchaudio_metadata_safe()


def ensure_torchvision_import_safe() -> None:
    try:
        import torchvision  # noqa: F401
    except Exception as exc:
        _install_unavailable_torchvision_stub(exc)


def ensure_torchaudio_metadata_safe() -> None:
    try:
        import torchaudio
    except Exception:
        return
    if hasattr(torchaudio, "AudioMetaData"):
        return
    torchaudio.AudioMetaData = namedtuple(  # type: ignore[attr-defined]
        "AudioMetaData",
        ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
        defaults=[0, 0, 0, 0, "UNKNOWN"],
    )


def _install_unavailable_torchvision_stub(original_error: Exception) -> None:
    for name in list(sys.modules):
        if name == "torchvision" or name.startswith("torchvision."):
            sys.modules.pop(name, None)

    root = _unavailable_module("torchvision", original_error, package=True)
    sys.modules["torchvision"] = root
    for child in ("datasets", "extension", "io", "models", "ops", "transforms", "utils"):
        module_name = f"torchvision.{child}"
        module = _unavailable_module(module_name, original_error)
        setattr(root, child, module)
        sys.modules[module_name] = module


def _unavailable_module(name: str, original_error: Exception, package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=package)
    if package:
        module.__path__ = []
    unavailable = _UnavailableObject(original_error)

    def __getattr__(attr: str) -> object:
        if attr.startswith("__"):
            raise AttributeError(attr)
        return unavailable

    module.__getattr__ = __getattr__  # type: ignore[attr-defined]
    if name.endswith(".extension"):
        module._has_ops = lambda: False  # type: ignore[attr-defined]
    return module
