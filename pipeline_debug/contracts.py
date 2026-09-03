from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineArtifact:
    path: str
    media_type: str
    sample_rate: int | None = None
    duration_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "media_type": self.media_type,
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


@dataclass
class Segment:
    index: str
    start: float
    end: float
    speaker: str | None = None
    text: str | None = None
    source_phase: str | None = None
    flags: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end - self.start)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "duration_seconds": round(self.duration_seconds, 3),
        }
        if self.speaker is not None:
            payload["speaker"] = self.speaker
        if self.text is not None:
            payload["text"] = self.text
        if self.source_phase is not None:
            payload["source_phase"] = self.source_phase
        if self.flags:
            payload["flags"] = self.flags
        return payload


@dataclass
class PhaseResult:
    phase: str
    enabled: bool
    model_info: dict[str, Any]
    started_at: str
    ended_at: str
    duration_seconds: float
    inputs: list[PipelineArtifact] = field(default_factory=list)
    outputs: list[PipelineArtifact] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "enabled": self.enabled,
            "model_info": self.model_info,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": round(float(self.duration_seconds), 3),
            "inputs": [artifact.to_json() for artifact in self.inputs],
            "outputs": [artifact.to_json() for artifact in self.outputs],
            "metrics": self.metrics,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def artifact_for_path(path: Path, media_type: str, sample_rate: int | None = None,
                      duration_seconds: float | None = None,
                      metadata: dict[str, Any] | None = None) -> PipelineArtifact:
    return PipelineArtifact(
        path=str(path),
        media_type=media_type,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        metadata=metadata or {},
    )

