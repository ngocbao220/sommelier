from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_debug.config import load_config, prepare_config, validate_config
from pipeline_debug.io_utils import resolve_device
from pipeline_debug.pipeline import DebugPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run debuggable Sommelier podcast pipeline")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--input", default="", help="Audio file or folder override")
    parser.add_argument("--output", default="", help="Output root override")
    parser.add_argument("--stop-after", default="", help="Stop after a phase id, e.g. 02_vad_silero")
    parser.add_argument("--phase", default="", help="Alias for --stop-after")
    parser.add_argument("--device", default="", help="Device override: auto, cpu, cuda, gpu, mps")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--real", action="store_true", help="Run real models")
    mode.add_argument("--dry-run", action="store_true", help="Run deterministic contracts without model loads")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.input:
        config["input"]["path"] = args.input
    if args.output:
        config["input"]["output_root"] = args.output
    if args.stop_after or args.phase:
        config["runtime"]["stop_after"] = args.stop_after or args.phase
    if args.device:
        config["runtime"]["device"] = args.device
    if args.real:
        config["runtime"]["mode"] = "real"
    if args.dry_run:
        config["runtime"]["mode"] = "dry_run"

    validate_config(config)
    config = prepare_config(config)
    device = resolve_device(config["runtime"]["device"])
    pipeline = DebugPipeline(config, device)
    outputs = pipeline.run_all()
    print("Directory processing finished.")
    print(f" - Success: {len(outputs)}")
    print(" - Failed: 0")
    for output in outputs:
        print(f" - Report: {Path(output)}")


if __name__ == "__main__":
    main()
