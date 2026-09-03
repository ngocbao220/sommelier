from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pipeline_debug.contracts import Segment
from pipeline_debug.io_utils import ensure_project_imports


@dataclass
class SileroVadAdapter:
    config: dict[str, Any]
    device: str
    mode: str = "real"
    _model: Any = None

    @property
    def model_info(self) -> dict[str, Any]:
        return {
            "backend": self.config.get("backend", "silero"),
            "model": self.config.get("model", "silero_vad"),
            "device": self.device,
            "mode": self.mode,
            "threshold": self.config.get("threshold"),
        }

    def load(self) -> None:
        if self.mode == "dry_run" or self._model is not None:
            return
        ensure_project_imports()
        import torch
        from models import silero_vad

        self._model = silero_vad.SileroVAD(
            local=self.config.get("source") == "local",
            model=self.config.get("model", "silero_vad"),
            device=torch.device(self.device),
        )

    def run(self, audio: dict) -> list[Segment]:
        duration = len(audio["waveform"]) / int(audio["sample_rate"])
        if duration <= 0:
            return []
        if self.mode == "dry_run":
            return [
                Segment(
                    index="00000",
                    start=0.0,
                    end=round(duration, 3),
                    speaker=None,
                    source_phase="02_vad_silero",
                    flags={"vad": True, "dry_run": True},
                )
            ]

        self.load()
        sample_rate = int(audio["sample_rate"])
        waveform = np.asarray(audio["waveform"], dtype=np.float32)
        if sample_rate != 16000:
            import librosa

            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        kwargs = {
            "sampling_rate": sample_rate,
            "threshold": float(self.config.get("threshold", 0.35)),
            "min_speech_duration_ms": int(self.config.get("min_speech_duration_ms", 250)),
            "min_silence_duration_ms": int(self.config.get("min_silence_duration_ms", 100)),
        }
        timestamps = self._model.get_speech_timestamps(waveform, self._model.vad_model, **kwargs)
        segments = []
        for idx, timestamp in enumerate(timestamps):
            segments.append(
                Segment(
                    index=f"{idx:05d}",
                    start=float(timestamp["start"]) / sample_rate,
                    end=float(timestamp["end"]) / sample_rate,
                    source_phase="02_vad_silero",
                    flags={"vad": True},
                )
            )
        return segments
