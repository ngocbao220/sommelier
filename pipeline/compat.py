from __future__ import annotations

import importlib.machinery
import inspect
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
    ensure_huggingface_hub_token_safe()


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
        pass
    else:
        torchaudio.AudioMetaData = namedtuple(  # type: ignore[attr-defined]
            "AudioMetaData",
            ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
            defaults=[0, 0, 0, 0, "UNKNOWN"],
        )
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]


def ensure_huggingface_hub_token_safe() -> None:
    try:
        import huggingface_hub
        from huggingface_hub import file_download
    except Exception:
        return
    _patch_hf_hub_download_module(huggingface_hub)
    _patch_hf_hub_download_module(file_download)


def _patch_hf_hub_download_module(module: types.ModuleType) -> None:
    original = getattr(module, "hf_hub_download", None)
    if original is None or getattr(original, "_sommelier_token_compat", False):
        return

    def hf_hub_download_token_compat(*args: object, **kwargs: object) -> object:
        if "use_auth_token" in kwargs and "token" not in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        else:
            kwargs.pop("use_auth_token", None)
        return original(*args, **kwargs)

    hf_hub_download_token_compat._sommelier_token_compat = True  # type: ignore[attr-defined]
    module.hf_hub_download = hf_hub_download_token_compat


def ensure_pyannote_pipeline_runtime_safe() -> None:
    try:
        from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
    except Exception:
        return
    _patch_constructor_unsupported_kwargs(SpeakerDiarization)


def _patch_constructor_unsupported_kwargs(cls: type) -> None:
    original = getattr(cls, "__init__", None)
    if original is None or getattr(original, "_sommelier_kwargs_compat", False):
        return
    signature = inspect.signature(original)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return
    allowed = {
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }

    def init_kwargs_compat(self: object, *args: object, **kwargs: object) -> object:
        filtered = {key: value for key, value in kwargs.items() if key in allowed}
        return original(self, *args, **filtered)

    init_kwargs_compat._sommelier_kwargs_compat = True  # type: ignore[attr-defined]
    cls.__init__ = init_kwargs_compat  # type: ignore[method-assign]


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
