# Sommelier Refactor Pipeline

This folder is a standalone debuggable runner. It can be copied or zipped by itself and run without the parent Sommelier checkout for the default Colab sample flow.

## Run

Real model execution is the default:

```bash
bash run_pipeline.sh --config config.colab.json --input samples/sample.wav --output outputs/colab
```

Contract-only execution avoids model downloads:

```bash
bash run_pipeline.sh --dry-run --input samples/sample.wav --output /tmp/sommelier_refactor_debug
```

Stop after one phase:

```bash
bash run_pipeline.sh --dry-run --input samples/sample.wav --stop-after 02_vad_silero
```

## Colab

Create the uploadable package:

```bash
bash package_colab.sh
```

Upload `sommelier_refactor_colab.zip` to Colab, extract it to `/content`, then open `colab.ipynb` from the extracted folder. The notebook installs the minimal dependencies, verifies `samples/sample.wav`, and runs:

```bash
PYTHON_BIN=python bash run_pipeline.sh --config config.colab.json --input samples/sample.wav --output outputs/colab
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

Models are configured in JSON files:

- `config.colab.json`: real Silero VAD on `samples/sample.wav`; diarization and SepReformer are disabled so the package runs standalone without a Hugging Face token or external model checkout.
- `config.json`: full debug pipeline config; enable diarization with `HUGGINGFACE_TOKEN` and provide a valid SepReformer path before full real execution.
