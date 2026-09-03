from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline_debug.contracts import PipelineArtifact, artifact_for_path
from pipeline_debug.io_utils import write_wav


class AudioData(dict):
    @property
    def waveform(self) -> np.ndarray:
        return self["waveform"]

    @property
    def sample_rate(self) -> int:
        return int(self["sample_rate"])

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return len(self.waveform) / self.sample_rate


def load_audio(path: str | Path, target_sample_rate: int) -> AudioData:
    path = Path(path)
    try:
        import librosa

        waveform, sample_rate = librosa.load(path, sr=target_sample_rate, mono=True)
    except Exception:
        import soundfile as sf

        waveform, sample_rate = sf.read(path, always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if int(sample_rate) != int(target_sample_rate):
            import librosa

            waveform = librosa.resample(
                np.asarray(waveform, dtype=np.float32),
                orig_sr=int(sample_rate),
                target_sr=int(target_sample_rate),
            )
            sample_rate = target_sample_rate

    waveform = np.asarray(waveform, dtype=np.float32)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 1.0:
        waveform = waveform / peak
    return AudioData(
        waveform=waveform,
        sample_rate=int(target_sample_rate),
        name=path.stem,
        original_path=str(path),
    )


def slice_audio(audio: AudioData | dict, start: float, end: float) -> np.ndarray:
    sample_rate = int(audio["sample_rate"])
    start_frame = max(0, int(round(start * sample_rate)))
    end_frame = min(len(audio["waveform"]), int(round(end * sample_rate)))
    if end_frame <= start_frame:
        return np.asarray([], dtype=np.float32)
    return np.asarray(audio["waveform"][start_frame:end_frame], dtype=np.float32)


def write_audio_artifact(path: Path, audio: AudioData | dict,
                         waveform: np.ndarray | None = None) -> PipelineArtifact:
    data = np.asarray(audio["waveform"] if waveform is None else waveform, dtype=np.float32)
    write_wav(path, data, int(audio["sample_rate"]))
    duration = len(data) / int(audio["sample_rate"]) if int(audio["sample_rate"]) else 0.0
    return artifact_for_path(path, "audio/wav", int(audio["sample_rate"]), duration)

