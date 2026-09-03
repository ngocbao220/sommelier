# Sommelier Refactor Pipeline

This refactor keeps the original `main_original_ASR_MoE.py` intact and adds a debuggable runner under `refactor/`.

## Run

Real model execution is the default:

```bash
HUGGINGFACE_TOKEN=hf_... bash refactor/run_pipeline.sh --input path/to/audio.wav --output outputs/refactor
```

Contract-only execution avoids model downloads:

```bash
bash refactor/run_pipeline.sh --dry-run --input path/to/audio.wav --output /tmp/sommelier_refactor_debug
```

Stop after one phase:

```bash
bash refactor/run_pipeline.sh --dry-run --input path/to/audio.wav --stop-after 02_vad_silero
```

## Phase Artifacts

Each run creates a timestamped output folder with:

- `00_input/phase_result.json`
- `01_preprocess/normalized.wav`
- `02_vad_silero/segments.json` and VAD region WAV files
- `03_diarization_pyannote/segments.json`
- `04_overlap_detection/overlaps.json`
- `05_sepreformer/enhanced_segments.json` and enhanced WAV files when overlaps are processed
- `06_export/segments/*.wav`
- `final_report.json`

The model table printed at startup shows phase, enabled state, backend, model, device, mode, and model source. Hugging Face token values are never printed; only `present` or `missing` is shown.

## Models

Models are configured in `refactor/config.json`:

- VAD: `Silero_VAD`
- Diarization: `pyannote/speaker-diarization-community-1`
- Overlap separation: local `SepReformer`

