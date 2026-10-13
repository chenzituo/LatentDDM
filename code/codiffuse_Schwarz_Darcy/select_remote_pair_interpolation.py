"""Validation-select one parameter interpolation toward a remote-pair fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .adapter import H5DarcyDataset
from .finetune_remote_pair_response import (
    ORDINARY_NONREGRESSION_FACTOR, RemotePairDataset, validate_pairs,
)
from .residual_port_composer import SourceLinearResidualPortCommunicator
from .train_residual_port_chart import load_frozen_local, sha256
from .train_residual_port_communicator import (
    CANONICAL_ANALYZER_CONTRACT, CONTRACT, load_frozen_chart,
    source_linear_signal_width, validate as validate_ordinary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--remote-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alphas", type=float, nargs="+",
        default=(0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--pair-validation-batches", type=int, default=32)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def select(args: argparse.Namespace) -> dict[str, object]:
    if sorted(set(args.alphas)) != list(args.alphas) or args.alphas[0] != 0.0:
        raise ValueError("alphas must be unique, sorted, and start at zero")
    if min(args.alphas) < 0.0 or max(args.alphas) > 1.0:
        raise ValueError("interpolation alphas must lie in [0,1]")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    remote = torch.load(args.remote_checkpoint, map_location="cpu", weights_only=False)
    if initial.get("contract") != CONTRACT or remote.get("contract") != CONTRACT:
        raise ValueError("interpolation inputs must be phase-separated checkpoints")
    cfg = initial["communicator_spec"]
    if cfg.get("canonical_analyzer_contract") != CANONICAL_ANALYZER_CONTRACT:
        raise ValueError("interpolation requires homogeneous-q checkpoints")
    for key in ("source_local_sha256", "source_chart_sha256"):
        if initial[key] != remote[key]:
            raise ValueError(f"interpolation checkpoint lineage differs at {key}")
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, _ = load_frozen_chart(args.chart_checkpoint, device)
    communicator = SourceLinearResidualPortCommunicator(
        int(cfg["physical_channels"]), int(cfg["scalar_dim"]), spec.code_width,
        source_linear_signal_width(spec), token_width=int(cfg["token_width"]),
        heads=int(cfg["heads"]),
    ).to(device)
    initial_state = initial["communicator"]
    remote_state = remote["communicator"]
    if initial_state.keys() != remote_state.keys():
        raise ValueError("interpolation checkpoint parameter keys differ")
    heldout_pairs = RemotePairDataset(args.data, args.pairs, holdout=True)
    ordinary_val = H5DarcyDataset(args.data, "validation")
    pair_loader = DataLoader(
        heldout_pairs, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
    )
    ordinary_loader = DataLoader(
        ordinary_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
    )
    ordinary_keys = (
        "assembled", "energy", "residual", "value_interface", "flux_interface",
        "compatibility_cycle_relative_square",
    )
    rows = []
    selected: tuple[float, float, dict[str, torch.Tensor]] | None = None
    baseline_pair = None; baseline_ordinary = None
    for alpha in args.alphas:
        state = {
            name: (1.0 - alpha) * initial_state[name] + alpha * remote_state[name]
            if initial_state[name].is_floating_point() else initial_state[name]
            for name in initial_state
        }
        communicator.load_state_dict(state, strict=True); communicator.eval()
        pair = validate_pairs(
            pair_loader, communicator, chart, local, adapter, device, spec, scale, args,
        )
        ordinary = validate_ordinary(
            communicator, chart, local, adapter, ordinary_loader, device,
            spec, scale, args,
        )
        if baseline_pair is None:
            baseline_pair, baseline_ordinary = pair, ordinary
        assert baseline_ordinary is not None and baseline_pair is not None
        ordinary_gates = {
            key: ordinary[key] <= ORDINARY_NONREGRESSION_FACTOR * baseline_ordinary[key]
            for key in ordinary_keys
        }
        response_gates = {
            "full_strict_improvement": pair["full_response_relative_square"]
                < baseline_pair["full_response_relative_square"],
            "receiver_strict_improvement": pair["receiver_response_relative_square"]
                < baseline_pair["receiver_response_relative_square"],
            "receiver_balanced_strict_improvement":
                pair["receiver_balanced_log_relative_square"]
                < baseline_pair["receiver_balanced_log_relative_square"],
            "gain_physical": 0.05 <= pair["receiver_gain_mean"] <= 1.95,
        }
        clears = all(ordinary_gates.values()) and all(response_gates.values())
        score = (
            pair["receiver_response_relative_square"]
            + 0.25 * pair["receiver_balanced_log_relative_square"]
            + 0.5 * pair["full_response_relative_square"]
        )
        rows.append({
            "alpha": alpha, "score": score, "clears": clears,
            "ordinary_gates": ordinary_gates, "response_gates": response_gates,
            "pair": pair, "ordinary": ordinary,
        })
        if clears and (selected is None or score < selected[0]):
            selected = score, alpha, {key: value.detach().cpu() for key, value in state.items()}
    heldout_pairs.close(); ordinary_val.close()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "contract": "darcy_remote_pair_parameter_interpolation_gate_v1",
        "selection_data": ["ordinary validation", "heldout train-split pairs"],
        "untouched_validation_remote_pairs_opened": False,
        "test_subsets_opened": False,
        "initial_checkpoint_sha256": sha256(args.initial_checkpoint),
        "remote_checkpoint_sha256": sha256(args.remote_checkpoint),
        "selected_alpha": None if selected is None else selected[1],
        "rows": rows,
    }
    (args.output_dir / "interpolation_gate.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    if selected is None:
        raise RuntimeError("no interpolation alpha clears ordinary and response gates")
    _, alpha, state = selected
    payload = dict(initial)
    payload.update({
        "phase": "validation_selected_parameter_interpolation_single_kernel",
        "communicator": state,
        "interpolation_alpha": alpha,
        "interpolation_initial_checkpoint": str(args.initial_checkpoint.resolve()),
        "interpolation_remote_checkpoint": str(args.remote_checkpoint.resolve()),
        "interpolation_gate": result,
        "single_message_no_ensemble": True,
        "fixed_context_q_linearity_preserved": True,
        "untouched_validation_remote_pairs_opened": False,
        "test_subsets_opened": False,
    })
    torch.save(payload, args.output_dir / "checkpoint.pt")
    return result


if __name__ == "__main__":
    print(json.dumps(select(parse_args()), indent=2))
