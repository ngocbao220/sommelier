from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

MODEL_PHASES = [
    {"phase": "02_vad_silero", "model_key": "vad", "token_label": ""},
    {"phase": "03_diarization_pyannote", "model_key": "diarization", "token_label": "HF token"},
    {"phase": "05_sepreformer", "model_key": "overlap_separation", "token_label": "HF token"},
]

SUPPORTED_BACKENDS = {
    "vad": {"silero"},
    "diarization": {"pyannote"},
    "overlap_separation": {"sepreformer"},
}


DEFAULT_CONFIG: dict[str, Any] = {
    "input": {
        "path": "audio_pipeline/test",
        "output_root": "outputs",
        "sample_rate": 16000,
        "extensions": [".wav", ".mp3", ".flac", ".m4a", ".aac", ".opus", ".ogg"],
    },
    "runtime": {
        "mode": "real",
        "device": "auto",
        "log_level": "INFO",
        "seed": 42,
        "stop_after": "",
    },
    "debug": {
        "save_json": True,
        "save_wav": True,
        "keep_temp": False,
    },
    "models": {
        "vad": {
            "enabled": True,
            "backend": "silero",
            "model": "silero_vad",
            "source": "github",
            "threshold": 0.35,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 100,
        },
        "diarization": {
            "enabled": True,
            "backend": "pyannote",
            "model": "pyannote/speaker-diarization-community-1",
            "token_env": "HUGGINGFACE_TOKEN",
            "max_speakers": 4,
        },
        "overlap_separation": {
            "enabled": True,
            "backend": "sepreformer",
            "model": "SepReformer_Base_WSJ0",
            "path": "../SepReformer",
            "overlap_threshold_seconds": 1.0,
            "embedding_model": "pyannote/embedding",
        },
    },
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_config_file(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        user_config = json.load(handle)
    config = deep_update(DEFAULT_CONFIG, user_config)
    config["_meta"] = {
        "config_path": str(config_path),
        "config_dir": str(config_path.parent),
        "project_root": str(project_root()),
    }
    validate_config(config)
    return config


def resolve_config_file(path: str | Path) -> Path:
    config_path = _expand_path(path)
    if config_path.is_absolute() or config_path.exists():
        return config_path.resolve()

    local_config = Path(__file__).resolve().parents[1] / config_path
    if local_config.exists():
        return local_config.resolve()

    project_config = project_root() / config_path
    return project_config.resolve()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def validate_config(config: dict[str, Any]) -> None:
    mode = config["runtime"]["mode"]
    if mode not in {"real", "dry_run"}:
        raise ValueError("runtime.mode must be 'real' or 'dry_run'")

    device = config["runtime"]["device"]
    if device not in {"auto", "cpu", "cuda", "gpu", "mps"}:
        raise ValueError("runtime.device must be one of auto, cpu, cuda, gpu, mps")

    for model_key, supported in SUPPORTED_BACKENDS.items():
        backend = config["models"][model_key]["backend"]
        if backend not in supported:
            choices = ", ".join(sorted(supported))
            raise ValueError(f"models.{model_key}.backend must be one of: {choices}")


def prepare_config(config: dict[str, Any]) -> dict[str, Any]:
    prepared = copy.deepcopy(config)
    meta = prepared.setdefault("_meta", {})
    meta.setdefault("config_dir", str(Path.cwd()))
    meta.setdefault("project_root", str(project_root()))

    prepared["input"]["path"] = str(
        resolve_runtime_path(prepared["input"]["path"], prepared, prefer_existing=True)
    )
    prepared["input"]["output_root"] = str(
        resolve_runtime_path(prepared["input"]["output_root"], prepared)
    )
    for model_cfg in prepared.get("models", {}).values():
        if model_cfg.get("path"):
            model_cfg["path"] = str(resolve_runtime_path(model_cfg["path"], prepared, prefer_existing=True))

    validate_config(prepared)
    return prepared


def resolve_runtime_path(value: str | Path, config: dict[str, Any], prefer_existing: bool = False) -> Path:
    path = _expand_path(value)
    if path.is_absolute():
        return path.resolve()

    candidates = _path_candidates(path, config)
    if prefer_existing:
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
    return candidates[0].resolve()


def describe_path_candidates(value: str | Path, config: dict[str, Any]) -> list[str]:
    path = _expand_path(value)
    if path.is_absolute():
        return [str(path)]
    return [str(candidate.resolve()) for candidate in _path_candidates(path, config)]


def _path_candidates(path: Path, config: dict[str, Any]) -> list[Path]:
    meta = config.get("_meta", {})
    bases = [
        Path(meta.get("config_dir", Path.cwd())),
        Path.cwd(),
        Path(meta.get("project_root", project_root())),
    ]
    candidates: list[Path] = []
    for base in bases:
        candidate = base / path
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def redact_model_config(config: dict[str, Any]) -> dict[str, Any]:
    redacted = copy.deepcopy(config)
    diar = redacted.get("models", {}).get("diarization", {})
    token_env = diar.get("token_env")
    diar["token_status"] = "present" if token_env and os.environ.get(token_env) else "missing"
    if "token" in diar:
        diar["token"] = "***redacted***"
    return redacted
