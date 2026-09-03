from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pipeline_debug.contracts import Segment


@dataclass
class SileroVadAdapter:
    config: dict[str, Any]
    device: str
    mode: str = "real"
    _model: Any = None
    _get_speech_timestamps: Any = None

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
        import torch

        model, utils = torch.hub.load(
            repo_or_dir=self.config.get("repo", "snakers4/silero-vad"),
            model=self.config.get("model", "silero_vad"),
            source=self.config.get("source", "github"),
            trust_repo=True,
        )
        self._model = model.to(torch.device(self.device))
        self._get_speech_timestamps = utils[0]

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
        import torch

        kwargs = {
            "sampling_rate": sample_rate,
            "threshold": float(self.config.get("threshold", 0.35)),
            "min_speech_duration_ms": int(self.config.get("min_speech_duration_ms", 250)),
            "min_silence_duration_ms": int(self.config.get("min_silence_duration_ms", 100)),
        }
        tensor = torch.from_numpy(waveform).to(torch.device(self.device))
        timestamps = self._get_speech_timestamps(tensor, self._model, **kwargs)
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
