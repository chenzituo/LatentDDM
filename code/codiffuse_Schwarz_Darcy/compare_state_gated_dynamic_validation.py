"""Paired validation audit for the identity-lifted Darcy communicator.

Training and trust-scale selection for the state-gated extension use only a
partition of the training set.  This entry point opens validation data once,
after selection, and compares the frozen candidate with its exact accepted
parent on identical cases.  It also repeats both models with radius-zero and
radius-one communication so an apparent gain cannot be credited to global
exchange when it survives removal of that exchange.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .adapter import H5DarcyDataset
from .evaluate import energy_relative_error, move_batch, relative_l2, residual_rms
from .latent_schur_teacher import inference_only_context
from .residual_port_composer import (
    DynamicRelativeGreenFluxResidualPortCommunicator,
    StateGatedDynamicRelativeGreenFluxResidualPortCommunicator,
)
from .train_incremental_flux_remote_schur import (
    _decomposed_fields,
    _load_communicator,
)
from .train_residual_port_chart import frozen_context, load_frozen_local, sha256
from .train_residual_port_communicator import load_frozen_chart
from .validate_latent_schur import measured_q, q_from_corrected


Tensor = torch.Tensor


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(values: np.ndarray, seed: int, draws: int) -> dict[str, float]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired bootstrap needs a nonempty vector")
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 1000):
        stop = min(start + 1000, draws)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "candidate_better_fraction": float(np.mean(values < 0.0)),
    }


def _per_case_metrics(
    fields: dict[str, Tensor], context, batch: dict[str, Tensor],
    spec, scale: Tensor, solution_std: float,
) -> dict[str, Tensor]:
    prediction = fields["assembled"] * solution_std * batch["mask"]
    raw_prediction = fields["raw_assembled"] * solution_std * batch["mask"]
    q0, support = measured_q(context, batch, spec, scale, "value_flux")
    q1 = q_from_corrected(
        fields["corrected"], context, batch, spec, scale, "value_flux",
    )
    axes = tuple(range(1, q0.ndim))
    cycle = (
        (q1.square() * support).sum(axes)
        / (q0.square() * support).sum(axes).clamp_min(1.0e-12)
    ).sqrt()
    return {
        "relative_l2": relative_l2(
            prediction, batch["solution"], batch["mask"],
        ),
        "raw_relative_l2": relative_l2(
            raw_prediction, batch["solution"], batch["mask"],
        ),
        "energy_relative_error": energy_relative_error(
            prediction, batch["solution"], batch["logk"], batch["mask"],
        ),
        "residual_rms": residual_rms(prediction, batch),
        "complete_q_cycle": cycle,
        "damping_alpha": fields["alpha"],
    }


@torch.no_grad()
def compare(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, _ = load_frozen_chart(args.chart_checkpoint, device)
    reference, reference_payload = _load_communicator(
        args.reference_checkpoint, spec, device,
    )
    candidate, candidate_payload = _load_communicator(
        args.candidate_checkpoint, spec, device,
    )
    if not isinstance(reference, DynamicRelativeGreenFluxResidualPortCommunicator) \
            or isinstance(reference, StateGatedDynamicRelativeGreenFluxResidualPortCommunicator):
        raise ValueError("reference is not the accepted pre-extension dynamic map")
    if not isinstance(candidate, StateGatedDynamicRelativeGreenFluxResidualPortCommunicator):
        raise ValueError("candidate is not the state-gated dynamic map")
    for payload in (reference_payload, candidate_payload):
        if payload.get("source_local_sha256") != sha256(args.local_checkpoint) or \
                payload.get("source_chart_sha256") != sha256(args.chart_checkpoint):
            raise ValueError("communicator/local/chart lineage differs")
    if candidate_payload.get("accepted_reference_sha256") != file_sha256(
        args.reference_checkpoint
    ):
        raise ValueError("candidate is not an audited lift of the reference")
    if not bool(candidate_payload.get("selected_metrics", {}).get("clears")):
        raise ValueError("candidate did not clear its train-only selection gate")
    if candidate_payload.get("validation_or_test_opened") or candidate_payload.get(
        "test_subsets_opened"
    ):
        raise ValueError("candidate selection metadata reports protected-data access")
    if max(candidate_payload.get("identity_lift_real_audit", {}).values()) != 0.0:
        raise ValueError("candidate checkpoint did not preserve the exact real-cover lift")

    for module in (local, chart, reference, candidate):
        module.eval()
    dataset = H5DarcyDataset(args.data, "validation")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    modes = {"global": None, "radius0": 0, "radius1": 1}
    records: list[dict[str, float | int]] = []
    processed = 0
    try:
        for host in loader:
            if processed >= args.maximum_cases:
                break
            batch = move_batch(host, device)
            original = int(batch["solution"].shape[0])
            keep = min(original, args.maximum_cases - processed)
            for name, value in list(batch.items()):
                if isinstance(value, Tensor) and value.shape[0] == original:
                    batch[name] = value[:keep]
            supervised = frozen_context(
                local, adapter, batch, spec, args.core, args.halo,
            )
            context = inference_only_context(supervised)
            values: dict[str, dict[str, dict[str, Tensor]]] = {}
            for mode, radius in modes.items():
                values[mode] = {}
                for name, communicator in (
                    ("reference", reference), ("candidate", candidate),
                ):
                    fields = _decomposed_fields(
                        communicator, chart, context, batch, spec, scale,
                        max_graph_radius=radius,
                    )
                    values[mode][name] = _per_case_metrics(
                        fields, context, batch, spec, scale,
                        adapter.statistics.solution_std,
                    )
            case_indices = batch["case_index"].detach().cpu().tolist()
            for item, case_index in enumerate(case_indices):
                row: dict[str, float | int] = {"case_index": int(case_index)}
                for mode in modes:
                    for name in ("reference", "candidate"):
                        for metric, tensor in values[mode][name].items():
                            row[f"{name}_{mode}_{metric}"] = float(tensor[item])
                records.append(row)
            processed += keep
    finally:
        dataset.close()
    if not records:
        raise RuntimeError("empty Darcy validation audit")

    metric_names = tuple(values["global"]["reference"])
    paired: dict[str, dict[str, float]] = {}
    deltas: dict[tuple[str, str], np.ndarray] = {}
    offset = 0
    for mode in modes:
        for metric in metric_names:
            difference = np.asarray([
                float(row[f"candidate_{mode}_{metric}"])
                - float(row[f"reference_{mode}_{metric}"])
                for row in records
            ])
            deltas[(mode, metric)] = difference
            paired[f"candidate_minus_reference_{mode}_{metric}"] = bootstrap(
                difference, args.seed + 17 * offset, args.bootstrap_draws,
            )
            offset += 1
    for mode in ("radius0", "radius1"):
        for metric in ("relative_l2", "complete_q_cycle"):
            loss = deltas[(mode, metric)] - deltas[("global", metric)]
            paired[f"{mode}_loss_of_incremental_{metric}_gain"] = bootstrap(
                loss, args.seed + 1009 + 31 * offset, args.bootstrap_draws,
            )
            offset += 1

    global_l2 = paired["candidate_minus_reference_global_relative_l2"]
    global_energy = paired[
        "candidate_minus_reference_global_energy_relative_error"
    ]
    global_residual = paired["candidate_minus_reference_global_residual_rms"]
    reference_cycle = np.mean([
        float(row["reference_global_complete_q_cycle"]) for row in records
    ])
    candidate_cycle = np.mean([
        float(row["candidate_global_complete_q_cycle"]) for row in records
    ])
    radius0_loss = paired["radius0_loss_of_incremental_relative_l2_gain"]
    promotion = {
        "train_only_selection_gate_passed": True,
        "validation_l2_mean_improves": bool(global_l2["mean"] < 0.0),
        "validation_l2_ci95_improves": bool(global_l2["ci95_high"] < 0.0),
        "energy_mean_nonregression_2pct_of_reference": bool(global_energy["mean"] <= (
            0.02 * np.mean([
                float(row["reference_global_energy_relative_error"])
                for row in records
            ])
        )),
        "residual_mean_nonregression_2pct_of_reference": bool(global_residual["mean"] <= (
            0.02 * np.mean([
                float(row["reference_global_residual_rms"])
                for row in records
            ])
        )),
        "complete_q_cycle_nonexpansive": bool(candidate_cycle <= 1.0),
        "complete_q_cycle_nonregression_1pct": bool(
            candidate_cycle <= 1.01 * reference_cycle
        ),
        "global_gain_exceeds_radius0_gain": bool(radius0_loss["mean"] > 0.0),
        "global_gain_exceeds_radius0_gain_ci95": bool(
            radius0_loss["ci95_low"] > 0.0
        ),
    }
    promotion["promoted"] = bool(all(promotion.values()))
    result = {
        "contract": "darcy_state_gated_dynamic_paired_validation_v1",
        "candidate_checkpoint": str(args.candidate_checkpoint.resolve()),
        "candidate_checkpoint_sha256": file_sha256(args.candidate_checkpoint),
        "reference_checkpoint": str(args.reference_checkpoint.resolve()),
        "reference_checkpoint_sha256": file_sha256(args.reference_checkpoint),
        "split": "validation",
        "cases": len(records),
        "radii": modes,
        "paired_bootstrap": paired,
        "reference_global_complete_q_cycle_mean": float(reference_cycle),
        "candidate_global_complete_q_cycle_mean": float(candidate_cycle),
        "promotion_gate": promotion,
        "records": records,
        "target_used_only_for_validation_metrics": True,
        "validation_used_for_training_or_trust_selection": False,
        "test_subsets_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-cases", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    compare(parse_args())
