"""Audit a frozen dynamic Darcy candidate on unused train-pair holdout rows.

The remote-pair trainer selects on the first ``selection_batches`` batches of
the train-only, base-index-disjoint holdout partition.  This program skips
that prefix and evaluates the frozen checkpoint only on the remaining suffix.
It does not open the Darcy validation/test solution subsets for remote-pair
selection and never changes the checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .adapter import H5DarcyDataset
from .finetune_remote_pair_response import RemotePairDataset
from .train_incremental_flux_remote_schur import (
    _load_communicator, ordinary_metrics, pair_terms,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ordinary-validation-cases", type=int, default=60)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def suffix_metrics(
    loader, skip: int, *model_args, max_graph_radius: int | None = None,
) -> tuple[dict[str, float], int]:
    totals: dict[str, float] = {}
    cases = 0
    for batch_index, host in enumerate(loader):
        if batch_index < skip:
            continue
        count = int(host["solution"].shape[0])
        values = pair_terms(
            *model_args, host=host, max_graph_radius=max_graph_radius,
        )
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + count * float(value)
        cases += count
    if cases == 0:
        raise RuntimeError("remote suffix is empty; reduce selection-batches")
    return ({name: value / cases for name, value in totals.items()}, cases)


@torch.no_grad()
def validate(args: argparse.Namespace) -> dict[str, Any]:
    if args.selection_batches < 0 or args.batch_size <= 0:
        raise ValueError("invalid held-out suffix extent")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, _ = load_frozen_chart(args.chart_checkpoint, device)
    candidate, payload = _load_communicator(args.checkpoint, spec, device)
    reference, reference_payload = _load_communicator(
        args.ordinary_reference_checkpoint, spec, device,
    )
    if getattr(candidate, "analyzer_mode", None) != \
            "value_flux_dynamic_relative_green":
        raise ValueError("suffix audit requires a dynamic relative Green checkpoint")
    for checkpoint in (payload, reference_payload):
        if checkpoint["source_local_sha256"] != sha256(args.local_checkpoint) or \
                checkpoint["source_chart_sha256"] != sha256(args.chart_checkpoint):
            raise ValueError("communicator/local/chart lineage differs")
    base = copy.deepcopy(candidate)
    with torch.no_grad():
        for parameter in base.flux_green.trust_path_parameters():
            parameter.zero_()
    for module in (candidate, base, reference, chart, local):
        module.eval()

    pair_set = RemotePairDataset(args.data, args.pairs, holdout=True)
    ordinary_set = H5DarcyDataset(args.data, "validation")
    pair_loader = DataLoader(
        pair_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    ordinary_loader = DataLoader(
        ordinary_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    tail = (reference, chart, local, adapter, device, spec, scale, args)
    baseline, baseline_cases = suffix_metrics(
        pair_loader, args.selection_batches, base, *tail,
    )
    candidate_pair, candidate_cases = suffix_metrics(
        pair_loader, args.selection_batches, candidate, *tail,
    )
    radius_zero_pair, radius_zero_cases = suffix_metrics(
        pair_loader, args.selection_batches, candidate, *tail,
        max_graph_radius=0,
    )
    radius_one_pair, radius_one_cases = suffix_metrics(
        pair_loader, args.selection_batches, candidate, *tail,
        max_graph_radius=1,
    )
    immutable = ordinary_metrics(
        reference, chart, local, adapter, ordinary_loader, device, spec, scale,
        args, incremental=False,
    )
    ordinary = ordinary_metrics(
        candidate, chart, local, adapter, ordinary_loader, device, spec, scale,
        args, incremental=True,
    )
    response_names = (
        "full_response_relative_square",
        "receiver_response_relative_square",
        "receiver_balanced_log_relative_square",
        "receiver_tail_log_relative_square",
    )
    gains = {
        name: 1.0 - candidate_pair[name] / baseline[name]
        for name in response_names
    }
    global_over_radius_zero = {
        name: 1.0 - candidate_pair[name] / radius_zero_pair[name]
        for name in response_names
    }
    global_over_radius_one = {
        name: 1.0 - candidate_pair[name] / radius_one_pair[name]
        for name in response_names
    }
    result = {
        "contract": "darcy_dynamic_remote_unused_suffix_audit_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "pair_partition": "train-only holdout suffix not used for selection",
        "selection_batches_skipped": args.selection_batches,
        "suffix_cases": candidate_cases,
        "baseline_suffix_cases": baseline_cases,
        "baseline_pair": baseline,
        "candidate_pair": candidate_pair,
        "candidate_radius_zero_pair": radius_zero_pair,
        "radius_zero_suffix_cases": radius_zero_cases,
        "candidate_radius_one_pair": radius_one_pair,
        "radius_one_suffix_cases": radius_one_cases,
        "relative_gains": gains,
        "global_over_radius_zero_gains": global_over_radius_zero,
        "global_over_radius_one_gains": global_over_radius_one,
        "ordinary_reference": immutable,
        "candidate_ordinary": ordinary,
        "ordinary_nonregression_2pct": {
            name: ordinary[name] <= 1.02 * immutable[name]
            for name in immutable
        },
        "mean_remote_improves_5pct": (
            gains["receiver_response_relative_square"] >= 0.05
        ),
        "all_remote_statistics_nonregress": all(
            value >= 0.0 for value in gains.values()
        ),
        "global_receiver_beats_radius_zero_5pct": (
            global_over_radius_zero["receiver_response_relative_square"] >= 0.05
        ),
        "global_full_response_beats_radius_zero_5pct": (
            global_over_radius_zero["full_response_relative_square"] >= 0.05
        ),
        "global_full_response_beats_radius_one_5pct": (
            global_over_radius_one["full_response_relative_square"] >= 0.05
        ),
        "validation_or_test_used_for_remote_selection": False,
        "test_subsets_opened": False,
    }
    pair_set.close(); ordinary_set.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(validate(parse_args()), indent=2))
