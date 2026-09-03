from __future__ import annotations

import json
import math
import os
import random
import sys
import warnings
from contextlib import redirect_stderr
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.config import MODEL_PHASES
from pipeline.contracts import PhaseResult, Segment


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_project_imports() -> None:
    root = str(project_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_device(name: str) -> str:
    requested = (name or "auto").strip().lower()
    if requested == "gpu":
        requested = "cuda"
    if requested == "auto":
        return "cuda" if _cuda_available() else "cpu"
    if requested == "cuda" and not _cuda_available():
        warnings.warn("CUDA requested but unavailable; falling back to CPU.", RuntimeWarning)
        return "cpu"
    return requested


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Segment):
        return value.to_json()
    if isinstance(value, PhaseResult):
        return value.to_json()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items() if not _is_audio_blob_key(k)}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _is_audio_blob_key(key: Any) -> bool:
    return str(key) in {"audio", "enhanced_audio", "waveform", "audio_segment"}


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def write_phase_result(phase_dir: Path, result: PhaseResult) -> Path:
    return write_json(phase_dir / "phase_result.json", result)


def write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf

        sf.write(path, np.asarray(waveform, dtype=np.float32), sample_rate)
    except Exception:
        import wave

        clipped = np.clip(waveform, -1.0, 1.0)
        pcm = (clipped * 32767).astype("<i2")
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())
    return path


def make_run_dir(output_root: str | Path, audio_path: str | Path) -> Path:
    root = Path(output_root)
    stem = Path(audio_path).stem
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{stem}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def collect_audio_paths(input_path: str | Path, extensions: list[str]) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"input path not found: {path}")
    allowed = {ext.lower() for ext in extensions}
    audio_paths = sorted(
        child for child in path.iterdir()
        if child.is_file() and child.suffix.lower() in allowed and ".temp" not in child.name
    )
    if not audio_paths:
        choices = ", ".join(sorted(allowed))
        raise FileNotFoundError(f"no audio files found in {path} with extensions: {choices}")
    return audio_paths


def print_model_table(config: dict[str, Any], device: str) -> None:
    mode = config["runtime"]["mode"]
    rows = []
    token_rows = []
    for spec in MODEL_PHASES:
        model_cfg = config["models"][spec["model_key"]]
        token_env = model_cfg.get("token_env")
        rows.append({
            "Phase": spec["phase"],
            "Enabled": "yes" if model_cfg.get("enabled", True) else "no",
            "Backend": model_cfg.get("backend", ""),
            "Model": model_cfg.get("model", ""),
            "Device": device,
            "Mode": mode,
            "Source": _model_source(model_cfg, config, token_env),
        })
        if token_env:
            token_rows.append({
                "Name": spec.get("token_label") or "Token",
                "Env": token_env,
                "Status": "present" if os.environ.get(token_env) else "missing",
            })

    _print_table(rows)
    if token_rows:
        print()
        _print_table(_unique_rows(token_rows))


def _print_table(rows: list[dict[str, Any]]) -> None:
    try:
        with redirect_stderr(StringIO()):
            import pandas as pd

        table = pd.DataFrame(rows)
        print(table.to_string(index=False))
    except Exception:
        headers = list(rows[0]) if rows else []
        widths = {
            header: max(len(header), *(len(str(row.get(header, ""))) for row in rows))
            for header in headers
        }
        print(" | ".join(header.ljust(widths[header]) for header in headers))
        print("-+-".join("-" * widths[header] for header in headers))
        for row in rows:
            print(" | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


def _unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = []
    seen = set()
    for row in rows:
        key = tuple(row.items())
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _model_source(model_cfg: dict[str, Any], config: dict[str, Any], token_env: str | None) -> str:
    if model_cfg.get("source"):
        return str(model_cfg["source"])
    if model_cfg.get("path"):
        return _display_path(Path(model_cfg["path"]), config)
    return token_env or ""


def _display_path(path: Path, config: dict[str, Any]) -> str:
    meta = config.get("_meta", {})
    base = Path(meta.get("config_dir", Path.cwd()))
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return str(path)
