from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.audio import slice_audio
from pipeline.compat import ensure_torchvision_import_safe
from pipeline.contracts import Segment
from pipeline.io_utils import ensure_project_imports
from pipeline.overlap import detect_overlaps


class _NullLogger:
    def debug(self, *_args, **_kwargs):
        pass

    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


@dataclass
class SepReformerAdapter:
    config: dict[str, Any]
    device: str
    mode: str = "real"
    _separator: Any = None
    _embedding_model: Any = None

    @property
    def model_info(self) -> dict[str, Any]:
        return {
            "backend": self.config.get("backend", "sepreformer"),
            "model": self.config.get("model"),
            "device": self.device,
            "mode": self.mode,
            "path": self.config.get("path"),
            "embedding_model": self.config.get("embedding_model"),
        }

    def load(self) -> None:
        if self.mode == "dry_run" or self._separator is not None:
            return
        ensure_project_imports()
        import torch
        ensure_torchvision_import_safe()
        from pyannote.audio import Model as PyannoteModel
        from utils.separation import SepReformerSeparator

        sepreformer_path = Path(self.config["path"])
        if not sepreformer_path.exists():
            raise FileNotFoundError(f"SepReformer path not found: {sepreformer_path}")

        token_env = self.config.get("token_env", "HUGGINGFACE_TOKEN")
        token = os.environ.get(token_env)
        self._separator = SepReformerSeparator(str(sepreformer_path), torch.device(self.device))
        embedding_name = self.config.get("embedding_model", "pyannote/embedding")
        kwargs = {"token": token} if token else {}
        try:
            self._embedding_model = PyannoteModel.from_pretrained(embedding_name, **kwargs)
        except TypeError:
            self._embedding_model = PyannoteModel.from_pretrained(embedding_name, use_auth_token=token)
        self._embedding_model = self._embedding_model.to(torch.device(self.device))

    def run(self, audio: dict, segments: list[Segment]) -> tuple[list[Segment], dict[str, np.ndarray], list[dict]]:
        threshold = float(self.config.get("overlap_threshold_seconds", 1.0))
        overlaps = detect_overlaps(segments, threshold)
        enhanced: dict[str, np.ndarray] = {}
        if not overlaps:
            return segments, enhanced, overlaps

        if self.mode == "dry_run":
            for overlap in overlaps:
                for index in (overlap["seg1"], overlap["seg2"]):
                    segment = next(item for item in segments if item.index == index)
                    enhanced[index] = slice_audio(audio, segment.start, segment.end)
                    segment.flags["sepreformer"] = True
                    segment.flags["dry_run"] = True
            return segments, enhanced, overlaps

        self.load()
        ensure_project_imports()
        from utils import separation as old_separation

        dict_segments = [
            {"index": seg.index, "start": seg.start, "end": seg.end, "speaker": seg.speaker}
            for seg in segments
        ]
        old_separation.set_logger(_NullLogger())
        _updated_audio, updated_segments = old_separation.process_overlapping_segments_with_separation(
            dict_segments,
            audio,
            overlap_threshold=threshold,
            separator=self._separator,
            embedding_model=self._embedding_model,
            device=self.device,
        )
        by_index = {seg.index: seg for seg in segments}
        for updated in updated_segments:
            index = updated["index"]
            if updated.get("sepreformer") and index in by_index:
                by_index[index].flags["sepreformer"] = True
            if "enhanced_audio" in updated:
                enhanced[index] = np.asarray(updated["enhanced_audio"], dtype=np.float32)
        return segments, enhanced, overlaps
