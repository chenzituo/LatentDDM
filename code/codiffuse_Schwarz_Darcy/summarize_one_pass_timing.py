"""Create hardware-matched one-pass versus iterative Darcy timing tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-pass-cpu", type=Path, required=True)
    parser.add_argument("--one-pass-gpu", type=Path, required=True)
    parser.add_argument("--huang-gpu", type=Path, required=True)
    parser.add_argument("--robin-cpu", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(args: argparse.Namespace) -> dict:
    cpu = _read(args.one_pass_cpu)
    gpu = _read(args.one_pass_gpu)
    huang = _read(args.huang_gpu)
    robin = _read(args.robin_cpu)
    one_cpu = cpu["summary"]["predicted/radius_global"]
    one_gpu = gpu["summary"]["predicted/radius_global"]
    rows = []

    def add_one(device: str, metric: dict, cases: int) -> None:
        rows.append({
            "method": "one-pass value+flux communicator", "device_class": device,
            "validation_cases": cases, "sweeps": 1,
            "batched_local_calls": metric["batched_local_calls"],
            "local_nfe": metric["local_nfe"],
            "communication_rounds": metric["communication_rounds"],
            "wall_seconds_per_case": metric["wall_seconds"],
            "speedup_over_one_pass_same_device": 1.0,
        })

    add_one("16-core CPU allocation", one_cpu, int(cpu["cases"]))
    for sweep in (8, 64):
        metric = robin["summary"][f"p_ratio_0.5000/checkpoint_{sweep:04d}"]
        rows.append({
            "method": "exact Robin OSM", "device_class": "16-core CPU allocation",
            "validation_cases": int(robin["cases"]), "sweeps": sweep,
            "batched_local_calls": metric["batched_local_calls"],
            "local_nfe": metric["local_solves"], "communication_rounds": sweep,
            "wall_seconds_per_case": metric["wall_seconds"],
            "speedup_over_one_pass_same_device":
                metric["wall_seconds"] / one_cpu["wall_seconds"],
        })
    add_one("NVIDIA H200", one_gpu, int(gpu["cases"]))
    for sweep in (8, 64):
        metric = huang["summary"][f"tau_0.1000/checkpoint_{sweep:04d}"]
        rows.append({
            "method": "literal Huang neural SNI", "device_class": "NVIDIA H200",
            "validation_cases": int(huang["cases_by_tau"]["0.1"]),
            "sweeps": sweep, "batched_local_calls": metric["batched_local_calls"],
            "local_nfe": metric["local_nfe"], "communication_rounds": sweep,
            "wall_seconds_per_case": metric["wall_seconds"],
            "speedup_over_one_pass_same_device":
                metric["wall_seconds"] / one_gpu["wall_seconds"],
        })
    result = {
        "contract": "darcy_hardware_matched_one_pass_timing_v1",
        "selection_split": "validation", "test_subsets_opened": False,
        "timing_definition": (
            "per-case synchronized inference; diagnostics excluded by each source evaluator"
        ),
        "comparison_rule": (
            "speedups are computed only within device class; validation manifests "
            "differ and are reported explicitly"
        ),
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "timing.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(summarize(parse_args()), indent=2))
