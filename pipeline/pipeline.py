from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from pipeline.audio import load_audio, slice_audio, write_audio_artifact
from pipeline.contracts import PhaseResult, artifact_for_path
from pipeline.diarization import PyannoteDiarizationAdapter
from pipeline.io_utils import (
    collect_audio_paths,
    make_run_dir,
    print_model_table,
    set_seed,
    utc_now,
    write_json,
    write_phase_result,
    write_wav,
)
from pipeline.overlap import detect_overlaps
from pipeline.sepreformer import SepReformerAdapter
from pipeline.vad import SileroVadAdapter


PHASE_ORDER = [
    "00_input",
    "01_preprocess",
    "02_vad_silero",
    "03_diarization_pyannote",
    "04_overlap_detection",
    "05_sepreformer",
    "06_export",
]


@contextmanager
def phase_timer(name: str, enabled: bool, model_info: dict[str, Any],
                inputs: list[Any] | None = None) -> Iterator[dict[str, Any]]:
    started = utc_now()
    start_time = time.time()
    state = {
        "phase": name,
        "enabled": enabled,
        "model_info": model_info,
        "inputs": inputs or [],
        "outputs": [],
        "metrics": {},
        "warnings": [],
        "errors": [],
    }
    try:
        yield state
    except Exception as exc:
        state["errors"].append(str(exc))
        raise
    finally:
        state["result"] = PhaseResult(
            phase=name,
            enabled=enabled,
            model_info=model_info,
            started_at=started,
            ended_at=utc_now(),
            duration_seconds=time.time() - start_time,
            inputs=state["inputs"],
            outputs=state["outputs"],
            metrics=state["metrics"],
            warnings=state["warnings"],
            errors=state["errors"],
        )


class DebugPipeline:
    def __init__(self, config: dict[str, Any], device: str) -> None:
        self.config = config
        self.device = device
        self.mode = config["runtime"]["mode"]
        self.debug = config["debug"]
        self.vad = SileroVadAdapter(config["models"]["vad"], device, self.mode)
        self.diar = PyannoteDiarizationAdapter(config["models"]["diarization"], device, self.mode)
        self.sep = SepReformerAdapter(config["models"]["overlap_separation"], device, self.mode)

    def run_all(self) -> list[Path]:
        set_seed(int(self.config["runtime"].get("seed", 42)))
        print_model_table(self.config, self.device)
        audio_paths = collect_audio_paths(
            self.config["input"]["path"],
            self.config["input"]["extensions"],
        )
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = lambda x, **_: x
        outputs = []
        for audio_path in tqdm(audio_paths, desc="Files"):
            outputs.append(self.run_one(audio_path))
        return outputs

    def run_one(self, audio_path: str | Path) -> Path:
        audio_path = Path(audio_path)
        run_dir = make_run_dir(self.config["input"]["output_root"], audio_path)
        phase_results = []
        original_artifact = artifact_for_path(audio_path, "audio/input")

        with phase_timer("00_input", True, {"backend": "filesystem"}, [original_artifact]) as phase:
            phase["metrics"]["audio_path"] = str(audio_path)
            phase["outputs"].append(original_artifact)
        self._write_phase(run_dir, phase)
        phase_results.append(phase["result"])
        if self._should_stop("00_input"):
            return self._finish(run_dir, phase_results, [])

        with phase_timer("01_preprocess", True, {"backend": "librosa"}, [original_artifact]) as phase:
            audio = load_audio(audio_path, int(self.config["input"]["sample_rate"]))
            phase_dir = run_dir / "01_preprocess"
            normalized = write_audio_artifact(phase_dir / "normalized.wav", audio)
            phase["outputs"].append(normalized)
            phase["metrics"]["duration_seconds"] = normalized.duration_seconds
        self._write_phase(run_dir, phase)
        phase_results.append(phase["result"])
        if self._should_stop("01_preprocess"):
            return self._finish(run_dir, phase_results, [])

        preprocessed_path = Path(phase_results[-1].outputs[0].path)
        with phase_timer("02_vad_silero", self.config["models"]["vad"]["enabled"],
                         self.vad.model_info, [artifact_for_path(preprocessed_path, "audio/wav")]) as phase:
            vad_segments = self.vad.run(audio) if phase["enabled"] else []
            write_json(run_dir / "02_vad_silero" / "segments.json",
                       [segment.to_json() for segment in vad_segments])
            phase["outputs"].append(artifact_for_path(run_dir / "02_vad_silero" / "segments.json", "application/json"))
            if self.debug.get("save_wav"):
                for segment in vad_segments:
                    wav = slice_audio(audio, segment.start, segment.end)
                    path = run_dir / "02_vad_silero" / f"{segment.index}.wav"
                    write_wav(path, wav, int(audio["sample_rate"]))
            phase["metrics"]["segments"] = len(vad_segments)
        self._write_phase(run_dir, phase)
        phase_results.append(phase["result"])
        if self._should_stop("02_vad_silero"):
            return self._finish(run_dir, phase_results, vad_segments)

        with phase_timer("03_diarization_pyannote", self.config["models"]["diarization"]["enabled"],
                         self.diar.model_info, [artifact_for_path(preprocessed_path, "audio/wav")]) as phase:
            diar_segments = self.diar.run(preprocessed_path, vad_segments) if phase["enabled"] else vad_segments
            write_json(run_dir / "03_diarization_pyannote" / "segments.json",
                       [segment.to_json() for segment in diar_segments])
            phase["outputs"].append(artifact_for_path(run_dir / "03_diarization_pyannote" / "segments.json", "application/json"))
            phase["metrics"]["segments"] = len(diar_segments)
            phase["metrics"]["speakers"] = sorted({seg.speaker for seg in diar_segments if seg.speaker})
        self._write_phase(run_dir, phase)
        phase_results.append(phase["result"])
        if self._should_stop("03_diarization_pyannote"):
            return self._finish(run_dir, phase_results, diar_segments)

        with phase_timer("04_overlap_detection", True, {"backend": "interval_scan"},
                         [artifact_for_path(run_dir / "03_diarization_pyannote" / "segments.json", "application/json")]) as phase:
            overlaps = detect_overlaps(
                diar_segments,
                float(self.config["models"]["overlap_separation"].get("overlap_threshold_seconds", 1.0)),
            )
            write_json(run_dir / "04_overlap_detection" / "overlaps.json", overlaps)
            phase["outputs"].append(artifact_for_path(run_dir / "04_overlap_detection" / "overlaps.json", "application/json"))
            phase["metrics"]["overlap_pairs"] = len(overlaps)
        self._write_phase(run_dir, phase)
        phase_results.append(phase["result"])
        if self._should_stop("04_overlap_detection"):
            return self._finish(run_dir, phase_results, diar_segments)

        with phase_timer("05_sepreformer", self.config["models"]["overlap_separation"]["enabled"],
                         self.sep.model_info, [artifact_for_path(run_dir / "04_overlap_detection" / "overlaps.json", "application/json")]) as phase:
            enhanced = {}
            if phase["enabled"]:
                diar_segments, enhanced, overlaps = self.sep.run(audio, diar_segments)
            phase_dir = run_dir / "05_sepreformer"
            write_json(phase_dir / "enhanced_segments.json", {
                "overlaps": overlaps,
                "segments": [segment.to_json() for segment in diar_segments],
                "enhanced_indices": sorted(enhanced),
            })
            phase["outputs"].append(artifact_for_path(phase_dir / "enhanced_segments.json", "application/json"))
            for index, waveform in enhanced.items():
                if self.debug.get("save_wav"):
                    write_wav(phase_dir / f"{index}_enhanced.wav", waveform, int(audio["sample_rate"]))
            phase["metrics"]["enhanced_segments"] = len(enhanced)
        self._write_phase(run_dir, phase)
        phase_results.append(phase["result"])
        if self._should_stop("05_sepreformer"):
            return self._finish(run_dir, phase_results, diar_segments)

        with phase_timer("06_export", True, {"backend": "json_wav_export"},
                         [artifact_for_path(run_dir / "05_sepreformer" / "enhanced_segments.json", "application/json")]) as phase:
            export_dir = run_dir / "06_export" / "segments"
            for segment in diar_segments:
                waveform = (
                    enhanced[segment.index]
                    if segment.index in enhanced
                    else slice_audio(audio, segment.start, segment.end)
                )
                if self.debug.get("save_wav"):
                    write_wav(export_dir / f"{segment.index}_{segment.speaker or 'UNKNOWN'}.wav",
                              waveform, int(audio["sample_rate"]))
            final_json = self._finish(run_dir, phase_results, diar_segments, final_only=True)
            phase["outputs"].append(artifact_for_path(final_json, "application/json"))
            phase["metrics"]["segments"] = len(diar_segments)
        self._write_phase(run_dir, phase)
        phase_results.append(phase["result"])
        return self._finish(run_dir, phase_results, diar_segments)

    def _write_phase(self, run_dir: Path, phase: dict[str, Any]) -> None:
        if self.debug.get("save_json", True):
            write_phase_result(run_dir / phase["phase"], phase["result"])

    def _finish(self, run_dir: Path, phase_results: list[PhaseResult],
                segments: list[Any], final_only: bool = False) -> Path:
        payload = {
            "run_dir": str(run_dir),
            "mode": self.mode,
            "device": self.device,
            "phases": [result.to_json() for result in phase_results],
            "segments": [segment.to_json() for segment in segments],
        }
        final_path = run_dir / "final_report.json"
        write_json(final_path, payload)
        if not final_only:
            print(f"Saved debug report: {final_path}")
        return final_path

    def _should_stop(self, phase: str) -> bool:
        stop_after = self.config["runtime"].get("stop_after") or ""
        return stop_after == phase
