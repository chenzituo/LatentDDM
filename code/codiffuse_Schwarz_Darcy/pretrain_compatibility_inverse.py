"""Label-free frozen-chart probe--invert pretraining for Darcy communication."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from codiffuse_Schwarz.reduced_inverse import (
    build_reduced_inverse_probe, inverse_cycle_relative_square,
)
from codiffuse_Schwarz.tile_communication import ResidualKernelAttention

from .adapter import H5DarcyDataset
from .evaluate import move_batch
from .residual_port_composer import SourceLinearResidualPortCommunicator
from .train import make_loader, seed_everything
from .train_residual_port_chart import frozen_context, load_frozen_local, sha256
from .train_residual_port_communicator import (
    CANONICAL_ANALYZER_CONTRACT, load_frozen_chart,
    source_linear_features, source_linear_signal_width,
)


Tensor = torch.Tensor
CONTRACT = "darcy_compatibility_inverse_probe_v1"


def coordinate_support(context, spec) -> Tensor:
    """Artificial-face code coordinates; structural physical/empty are zero."""
    artificial_face = (
        context.physical[:, 24:28] * context.valid
    ).flatten(2).sum(2) > 0
    repeat = spec.field_channels * spec.modes_per_face
    return artificial_face.repeat_interleave(repeat, dim=1).reshape(
        context.local.shape[0] // context.cover.tile_count, context.cover.tile_count,
        spec.code_width,
    ).to(context.local)


def probe_terms(
    communicator, chart, context, spec, scale, bound: float,
    real_code_rms_cap: float, stability_weight: float,
) -> dict[str, Tensor]:
    if bound <= 0.0:
        raise ValueError("probe bound must be positive")
    support = coordinate_support(context, spec)
    excitation = (
        torch.empty_like(support).uniform_(-bound, bound) * support
    )
    baseline_q = source_linear_features(context, spec, scale)
    response = chart(
        context.local, context.physical, context.scalar, context.valid,
        excitation.reshape(-1, spec.code_width),
    )
    perturbed_context = replace(context, local=response.corrected)
    perturbed_q = source_linear_features(perturbed_context, spec, scale)
    probe = build_reduced_inverse_probe(
        baseline_q, perturbed_q, excitation,
        supported_coordinates=support,
    )
    predicted, _ = communicator(
        context.local, context.physical, context.scalar, context.valid,
        context.cover.grid_shape, aligned_features=probe.inverse_rhs,
    )
    cycle = inverse_cycle_relative_square(
        predicted, probe, supported_coordinates=support,
    ).mean()
    predicted_real, _ = communicator(
        context.local, context.physical, context.scalar, context.valid,
        context.cover.grid_shape, aligned_features=baseline_q,
    )
    # ``code_head`` is deliberately zero initialized, hence ``predicted_real``
    # is exactly zero in the first optimization step.  Clamp the mean square
    # *before* sqrt: sqrt(0) has an infinite derivative and the inactive ReLU
    # hinge otherwise creates the indeterminate backward product 0 * inf,
    # poisoning the optimizer after the first batch.
    real_rms = predicted_real.square().mean((-1, -2)).clamp_min(1.0e-12).sqrt()
    stability = torch.relu(real_rms - real_code_rms_cap).square().mean()
    return {
        "total": cycle + stability_weight * stability,
        "inverse_cycle_relative_square": cycle,
        "real_q_code_rms": real_rms.mean(),
        "real_q_stability_hinge": stability,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--token-width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--probe-bound", type=float, default=1.0)
    parser.add_argument("--real-code-rms-cap", type=float, default=2.0)
    parser.add_argument("--stability-weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda",
    )
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, chart_checkpoint = load_frozen_chart(
        args.chart_checkpoint, device,
    )
    if chart_checkpoint["source_local_sha256"] != sha256(args.local_checkpoint):
        raise ValueError("probe chart/local checkpoint identities differ")
    signal_width = source_linear_signal_width(spec)
    communicator = SourceLinearResidualPortCommunicator(
        adapter.spatial_condition_channels + 1,
        adapter.scalar_condition_dim, spec.code_width, signal_width,
        token_width=args.token_width, heads=args.heads,
    ).to(device)
    if sum(isinstance(m, ResidualKernelAttention) for m in communicator.modules()) != 1:
        raise RuntimeError("probe communicator must contain one residual kernel")
    train_set = H5DarcyDataset(args.data, "train")
    val_set = H5DarcyDataset(args.data, "validation")
    train_loader = make_loader(
        train_set, args.batch_size, True, args.workers, device,
        args.samples_per_epoch,
    )
    val_loader = make_loader(
        val_set, args.batch_size, False, min(args.workers, 2), device,
    )
    optimizer = torch.optim.AdamW(communicator.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs + 1):
        communicator.train(); total = 0.0; batches = 0
        for host in train_loader:
            batch = move_batch(host, device)
            context = frozen_context(
                local, adapter, batch, spec, args.core, args.halo,
            )
            optimizer.zero_grad(set_to_none=True)
            terms = probe_terms(
                communicator, chart, context, spec, scale, args.probe_bound,
                args.real_code_rms_cap, args.stability_weight,
            )
            loss = terms["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite probe inverse loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(communicator.parameters(), 1.0)
            optimizer.step(); total += float(loss.detach()); batches += 1
        communicator.eval(); validation: dict[str, float] = {}; validation_batches = 0
        with torch.no_grad():
            for host in val_loader:
                if validation_batches >= args.validation_batches:
                    break
                batch = move_batch(host, device)
                context = frozen_context(
                    local, adapter, batch, spec, args.core, args.halo,
                )
                terms = probe_terms(
                    communicator, chart, context, spec, scale,
                    args.probe_bound, args.real_code_rms_cap,
                    args.stability_weight,
                )
                for name, value in terms.items():
                    validation[name] = validation.get(name, 0.0) + float(value)
                validation_batches += 1
        row = {
            "epoch": epoch,
            "train_total": total / max(batches, 1),
            **{
                f"validation_{name}": value / max(validation_batches, 1)
                for name, value in validation.items()
            },
        }
        history.append(row); print(json.dumps(row), flush=True)
    payload = {
        "contract": CONTRACT,
        "phase": "label_free_frozen_chart_probe_invert",
        "communicator": communicator.state_dict(),
        "source_local_sha256": sha256(args.local_checkpoint),
        "source_chart_sha256": sha256(args.chart_checkpoint),
        "port_spec": spec.to_json(),
        "communicator_spec": {
            "canonical_analyzer_contract": CANONICAL_ANALYZER_CONTRACT,
            "legacy_internal_alias": "source_linear",
            "compatibility_residual_sign": "q=d=P_chi(v)-v",
            "probe_training_pair": "-Delta_q -> xi",
            "production_input": "measured q (not -q)",
            "code_width": spec.code_width,
            "signal_width": signal_width,
            "token_width": args.token_width,
            "heads": args.heads,
            "q_d_coordinates": [],
            "supported_coordinates_only": True,
            "real_q_code_rms_cap": args.real_code_rms_cap,
            "stability_weight": args.stability_weight,
            "end_to_end_source_linearity_claimed": False,
        },
        "training_splits": ["train", "validation"],
        "test_subsets_opened": False,
        "target_or_pde_labels_used_in_probe_loss": False,
        "history": history,
        "args": vars(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    return {
        "contract": CONTRACT, "history": history,
        "test_subsets_opened": False,
    }


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2, default=str))
