from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.compat import ensure_torchvision_import_safe
from pipeline.contracts import Segment


@dataclass
class PyannoteDiarizationAdapter:
    config: dict[str, Any]
    device: str
    mode: str = "real"
    _pipeline: Any = None

    @property
    def model_info(self) -> dict[str, Any]:
        token_env = self.config.get("token_env", "HUGGINGFACE_TOKEN")
        return {
            "backend": self.config.get("backend", "pyannote"),
            "model": self.config.get("model"),
            "device": self.device,
            "mode": self.mode,
            "token_env": token_env,
            "token_status": "present" if os.environ.get(token_env) else "missing",
        }

    def load(self) -> None:
        if self.mode == "dry_run" or self._pipeline is not None:
            return
        token_env = self.config.get("token_env", "HUGGINGFACE_TOKEN")
        token = os.environ.get(token_env)
        if not token:
            raise RuntimeError(f"missing Hugging Face token: set {token_env}")

        import torch
        ensure_torchvision_import_safe()
        from pyannote.audio import Pipeline

        model_name = self.config.get("model", "pyannote/speaker-diarization-community-1")
        try:
            self._pipeline = Pipeline.from_pretrained(model_name, token=token)
        except TypeError:
            self._pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
        if hasattr(self._pipeline, "to"):
            self._pipeline.to(torch.device(self.device))

    def run(self, audio_path: str | Path, vad_segments: list[Segment]) -> list[Segment]:
        if self.mode == "dry_run":
            source = vad_segments or [Segment("00000", 0.0, 0.0)]
            output = []
            for idx, segment in enumerate(source):
                speaker = f"SPEAKER_{idx % 2:02d}"
                output.append(
                    Segment(
                        index=f"{idx:05d}",
                        start=segment.start,
                        end=segment.end,
                        speaker=speaker,
                        source_phase="03_diarization_pyannote",
                        flags={"diarization": True, "dry_run": True},
                    )
                )
            return output

        self.load()
        kwargs = {}
        if self.config.get("max_speakers"):
            kwargs["max_speakers"] = int(self.config["max_speakers"])
        annotation = self._pipeline(str(audio_path), **kwargs)
        return annotation_to_segments(annotation)


def annotation_to_segments(annotation: Any) -> list[Segment]:
    annotation = _speaker_annotation(annotation)
    rows = list(annotation.itertracks(yield_label=True))
    segments = []
    for idx, (turn, _track, speaker) in enumerate(rows):
        segments.append(
            Segment(
                index=f"{idx:05d}",
                start=float(turn.start),
                end=float(turn.end),
                speaker=str(speaker),
                source_phase="03_diarization_pyannote",
                flags={"diarization": True},
            )
        )
    return segments


def _speaker_annotation(output: Any) -> Any:
    if hasattr(output, "itertracks"):
        return output
    for attr in ("speaker_diarization", "exclusive_speaker_diarization"):
        annotation = getattr(output, attr, None)
        if annotation is not None and hasattr(annotation, "itertracks"):
            return annotation
    output_type = type(output).__name__
    available = ", ".join(
        name
        for name in dir(output)
        if not name.startswith("_") and not callable(getattr(output, name, None))
    )
    raise TypeError(
        f"unsupported diarization output {output_type}: expected Annotation-like object "
        f"with itertracks() or speaker_diarization; available attributes: {available or 'none'}"
    )
