"""Matched evaluation of one-pass POU and overlap feature exchange.

The candidate is required to differ from the retained shared checkpoint only
by ``feature_schwarz_gate``.  This makes the comparison an ablation of hidden
overlap communication rather than a second backbone training run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from codiffuse_Schwarz_Airfoil.dataset import FieldStatistics
from codiffuse_Schwarz_Airfoil.train_codiffuse_schwarz_rollout import (
    evaluate,
    selection_score,
)
from codiffuse_Schwarz_Airfoil.train_multitrajectory_multiresolution import (
    PairedTrajectoryCorpus,
)
from codiffuse_Schwarz_Airfoil.train_one_step_multilevel import load_pretrained
from codiffuse_Schwarz_Airfoil.interface_operator import load_interface_trace_checkpoint


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--feature-candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--validation-horizon", type=int, default=20)
    parser.add_argument("--validation-starts-per-trajectory", type=int, default=2)
    parser.add_argument("--fine-fraction", type=float, default=.35)
    parser.add_argument("--fine-trust", type=float, default=.5)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--vorticity-amplitude-fraction", type=float, default=.15)
    parser.add_argument("--endpoint-relaxation", type=float, default=1.0)
    parser.add_argument("--history-momentum", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="validate checkpoint identity without running the GPU rollout",
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def verify_backbone_identity(
    baseline: dict[str, Any], candidate: dict[str, Any],
) -> dict[str, Any]:
    """Require bitwise-identical learned weights except for the scalar gate."""
    baseline_state: dict[str, Tensor] = baseline["model"]
    candidate_state: dict[str, Tensor] = candidate["model"]
    gate = "feature_schwarz_gate"
    baseline_keys = set(baseline_state)
    candidate_keys = set(candidate_state)
    extra = sorted(candidate_keys - baseline_keys)
    missing = sorted(baseline_keys - candidate_keys)
    if extra != [gate] or missing:
        raise ValueError(
            f"candidate state difference must be only {gate}: extra={extra}, missing={missing}"
        )
    changed = [
        key for key in sorted(baseline_keys)
        if not torch.equal(baseline_state[key], candidate_state[key])
    ]
    if changed:
        raise ValueError(f"candidate changed frozen shared weights: {changed[:8]}")
    if candidate.get("split") != baseline.get("split"):
        raise ValueError("baseline and candidate use different trajectory splits")
    spec = candidate.get("model_spec", {})
    if not bool(spec.get("endpoint_feature_schwarz", False)):
        raise ValueError("candidate model spec does not enable feature Schwarz")
    raw_gate = float(candidate_state[gate])
    return {
        "shared_tensor_count": len(baseline_keys),
        "shared_weights_bitwise_identical": True,
        "only_candidate_parameter": gate,
        "raw_gate": raw_gate,
        "effective_gate": math.tanh(raw_gate),
        "candidate_epoch": int(candidate.get("epoch", -1)),
        "candidate_contract": candidate.get("training_contract"),
    }


def evaluation_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Arguments consumed by the retained protected rollout evaluator."""
    return argparse.Namespace(
        validation_horizon=args.validation_horizon,
        validation_starts_per_trajectory=args.validation_starts_per_trajectory,
        fine_fraction=args.fine_fraction,
        fine_trust=args.fine_trust,
        halo=args.halo,
        vorticity_amplitude_fraction=args.vorticity_amplitude_fraction,
        placement_ensemble=1,
        endpoint_relaxation=args.endpoint_relaxation,
        history_momentum=args.history_momentum,
        force_weight=0.0,
        interface_trace_weight=0.0,
        interface_trace_stations=64,
        balanced_force_loss=False,
        seed=args.seed,
    )


@torch.no_grad()
def evaluate_checkpoint(
    payload: dict[str, Any],
    corpus: PairedTrajectoryCorpus,
    statistics: FieldStatistics,
    runtime_args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, Any]]:
    if bool(payload.get("model_spec", {}).get("interface_trace_operator", False)):
        model, _ = load_interface_trace_checkpoint(payload, device)
    else:
        model, _ = load_pretrained(payload, device, history_steps=2)
    model.set_endpoint_statistics(
        statistics.state_mean, statistics.state_std, statistics.delta_std,
        statistics.delta_mean,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    metrics = evaluate(model, corpus, statistics, runtime_args, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime = {
        "wall_seconds": time.perf_counter() - start,
        "peak_gpu_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "effective_feature_gate": (
            float(torch.tanh(model.feature_schwarz_gate.detach()))
            if getattr(model, "feature_schwarz_gate", None) is not None else 0.0
        ),
    }
    return metrics, runtime


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.validation_horizon <= 0 or args.validation_starts_per_trajectory <= 0:
        raise ValueError("rollout horizon and starts must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline = _load(args.baseline)
    candidate = _load(args.feature_candidate)
    identity = verify_backbone_identity(baseline, candidate)
    result: dict[str, Any] = {
        "contract": "airfoil_feature_schwarz_ablation_v1",
        "baseline": str(args.baseline.resolve()),
        "feature_candidate": str(args.feature_candidate.resolve()),
        "split": args.split,
        "checkpoint_identity": identity,
        "evaluation": {
            "validation_horizon": args.validation_horizon,
            "starts_per_trajectory": args.validation_starts_per_trajectory,
            "fine_fraction": args.fine_fraction,
            "fine_trust": args.fine_trust,
            "halo": args.halo,
            "seed": args.seed,
        },
    }
    if not args.metadata_only:
        if not torch.cuda.is_available():
            raise RuntimeError("full Airfoil ablation evaluation requires CUDA")
        device = torch.device("cuda")
        statistics = FieldStatistics.from_json(baseline["statistics"])
        if candidate["statistics"] != baseline["statistics"]:
            raise ValueError("candidate and baseline use different normalization")
        split = baseline["split"][args.split]
        corpus = PairedTrajectoryCorpus(args.data_root, split, "all")
        runtime_args = evaluation_namespace(args)
        try:
            baseline_metrics, baseline_runtime = evaluate_checkpoint(
                baseline, corpus, statistics, runtime_args, device,
            )
            candidate_metrics, candidate_runtime = evaluate_checkpoint(
                candidate, corpus, statistics, runtime_args, device,
            )
        finally:
            corpus.close()
        horizon = args.validation_horizon
        baseline_score = selection_score(baseline_metrics, horizon)
        candidate_score = selection_score(candidate_metrics, horizon)
        result["results"] = {
            "one_pass_pou": {
                "selection_score": baseline_score,
                **baseline_metrics,
                **baseline_runtime,
            },
            "overlap_feature_exchange": {
                "selection_score": candidate_score,
                **candidate_metrics,
                **candidate_runtime,
            },
            "relative_change": {
                key: candidate_metrics[key] / max(baseline_metrics[key], 1.0e-20) - 1.0
                for key in (
                    "step1_amr_vorticity_l2_mean",
                    f"step{horizon}_amr_vorticity_l2_mean",
                    "all_step_amr_vorticity_l2_mean",
                    "c_drag_relative_l2",
                    "c_lift_relative_l2",
                )
            } | {
                "selection_score": candidate_score / max(baseline_score, 1.0e-20) - 1.0,
                "wall_seconds": candidate_runtime["wall_seconds"]
                / max(baseline_runtime["wall_seconds"], 1.0e-20) - 1.0,
            },
        }
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
