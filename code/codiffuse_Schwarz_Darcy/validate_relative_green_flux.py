"""Validation-only Pareto gates for the analytic relative Green communicator."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .adapter import H5DarcyDataset
from .finetune_remote_pair_response import RemotePairDataset
from .train_incremental_flux_remote_schur import (
    ORDINARY_NONREGRESSION_FACTOR, PAIR_IMPROVEMENT_FACTOR,
    _load_communicator, ordinary_metrics, validate_pairs,
)
from .train_residual_port_chart import load_frozen_local, sha256
from .train_residual_port_communicator import load_frozen_chart


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ordinary-reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ordinary-validation-cases", type=int, default=60)
    parser.add_argument("--pair-validation-batches", type=int, default=32)
    parser.add_argument(
        "--trust-scales", type=float, nargs="+",
        default=(0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0),
    )
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def validate(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, _ = load_frozen_chart(args.chart_checkpoint, device)
    candidate, checkpoint = _load_communicator(args.checkpoint, spec, device)
    reference, reference_checkpoint = _load_communicator(
        args.ordinary_reference_checkpoint, spec, device,
    )
    if getattr(candidate, "analyzer_mode", None) != "value_flux_relative_green":
        raise ValueError("candidate is not a relative Green communicator")
    for payload in (checkpoint, reference_checkpoint):
        if payload["source_local_sha256"] != sha256(args.local_checkpoint) or \
                payload["source_chart_sha256"] != sha256(args.chart_checkpoint):
            raise ValueError("communicator/local/chart lineage differs")
    base = copy.deepcopy(candidate)
    base.flux_green.kernel.zero_()
    for model in (candidate, base, reference, chart, local):
        model.eval()
    pair_set = RemotePairDataset(args.data, args.pairs, holdout=True)
    ordinary_set = H5DarcyDataset(args.data, "validation")
    pair_loader = DataLoader(
        pair_set, batch_size=args.batch_size, shuffle=False,
        num_workers=min(args.workers, 2), pin_memory=device.type == "cuda",
    )
    ordinary_loader = DataLoader(
        ordinary_set, batch_size=args.batch_size, shuffle=False,
        num_workers=min(args.workers, 2), pin_memory=device.type == "cuda",
    )
    model_tail = (reference, chart, local, adapter, device, spec, scale, args)
    baseline_pair = validate_pairs(
        pair_loader, base, *model_tail,
        maximum_batches=args.pair_validation_batches,
    )
    immutable_ordinary = ordinary_metrics(
        reference, chart, local, adapter, ordinary_loader, device, spec, scale,
        args, incremental=False,
    )
    base_ordinary = ordinary_metrics(
        base, chart, local, adapter, ordinary_loader, device, spec, scale,
        args, incremental=True,
    )
    warm_delta = max(
        abs(base_ordinary[name] - immutable_ordinary[name])
        for name in immutable_ordinary
    )
    fitted_kernel = candidate.flux_green.kernel.detach().clone()
    rows = []; selected = None; best_score = float("inf")
    for trust in sorted(set(args.trust_scales)):
        if not 0 < trust <= 1:
            raise ValueError("trust scales must lie in (0,1]")
        candidate.flux_green.kernel.copy_(trust * fitted_kernel)
        candidate_pair = validate_pairs(
            pair_loader, candidate, *model_tail,
            maximum_batches=args.pair_validation_batches,
        )
        candidate_ordinary = ordinary_metrics(
            candidate, chart, local, adapter, ordinary_loader, device, spec,
            scale, args, incremental=True,
        )
        ordinary_gates = {
            name: candidate_ordinary[name]
            <= ORDINARY_NONREGRESSION_FACTOR * immutable_ordinary[name]
            for name in immutable_ordinary
        }
        response_gates = {
            name: candidate_pair[name]
            <= PAIR_IMPROVEMENT_FACTOR * baseline_pair[name]
            for name in (
                "full_response_relative_square",
                "receiver_response_relative_square",
                "receiver_balanced_log_relative_square",
                "receiver_tail_log_relative_square",
            )
        }
        response_gates.update({
            "receiver_gain_physical":
                0.05 <= candidate_pair["receiver_gain_mean"] <= 1.95,
            "schur_nonexpansive":
                candidate_pair["schur_cycle_relative_square"] <= 1.0,
        })
        clears = all(ordinary_gates.values()) and all(response_gates.values())
        score = sum(
            candidate_pair[name] / baseline_pair[name]
            for name in (
                "full_response_relative_square",
                "receiver_response_relative_square",
                "receiver_balanced_log_relative_square",
                "receiver_tail_log_relative_square",
            )
        )
        row = {
            "trust_scale": trust, "score": score, "clears": clears,
            "candidate_ordinary": candidate_ordinary,
            "candidate_pair": candidate_pair,
            "ordinary_gates": ordinary_gates,
            "response_gates": response_gates,
        }
        rows.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        if clears and score < best_score:
            selected, best_score = row, score
    result = {
        "contract": "darcy_relative_green_flux_validation_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "selection_data": "ordinary validation plus held-out train pairs",
        "test_subsets_opened": False,
        "base_warm_start_max_abs_metric_delta": warm_delta,
        "immutable_ordinary": immutable_ordinary,
        "base_ordinary": base_ordinary,
        "baseline_pair": baseline_pair,
        "trust_scale_rows": rows,
        "selected_trust_scale": None if selected is None else selected["trust_scale"],
        "selected": selected,
        "clears": warm_delta <= 1e-6 and selected is not None,
    }
    pair_set.close(); ordinary_set.close()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(validate(parse_args()), indent=2))
