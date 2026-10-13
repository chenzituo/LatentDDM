"""Train a Pareto-safe flux/remote correction about the accepted value map.

The accepted value-only communicator is lifted into the value+flux analyzer by
appending eight signal columns whose weights are exactly zero.  All original
parameters and value-channel columns remain frozen.  Training can therefore
change only the incremental flux contribution of the *same* shared global
kernel.  At zero increment the method is bitwise identical to the accepted
value-only field.

Pair supervision uses only train-split cached operator responses.  The exact
global field is never an inference input.  The receiver loss is evaluated on
the corrected receiver tile, exactly matching the frozen remote-pair gate (and
not on a re-tiled assembled surrogate).  A target-free affine compatibility
retraction selects a scalar on the segment from the accepted field to the raw
incremental field, so the accepted field is always an admissible fallback.
The exact zero-suffix corner is trained for a short raw-direction warmup;
subsequent epochs optimize the same affine-retracted field used at deployment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from .adapter import FACE_DIRICHLET_FIELDS, REQUIRED_FIELDS, H5DarcyDataset
from .evaluate import move_batch, relative_l2
from .finetune_remote_pair_response import RemotePairDataset, _relative_square
from .latent_schur_teacher import inference_only_context
from .residual_port_composer import (
    ContextualRelativeGreenFluxResidualPortCommunicator,
    DynamicRelativeGreenFluxResidualPortCommunicator,
    RelativeGreenFluxResidualPortCommunicator,
    SourceLinearResidualPortCommunicator,
    StateGatedDynamicRelativeGreenFluxResidualPortCommunicator,
)
from .train import seed_everything
from .train_residual_port_chart import (
    frozen_context, interface_losses, load_frozen_local, masked_mse,
    physics_losses, sha256,
)
from .train_residual_port_communicator import (
    CANONICAL_ANALYZER_CONTRACT, CONTRACT, forward_composer,
    load_frozen_chart, source_linear_features, source_linear_signal_width,
)
from .validate_latent_schur import measured_q, q_from_corrected
from .validate_residual_port_composer import (
    damp_affine_compatibility_direction,
)


Tensor = torch.Tensor
PAIR_IMPROVEMENT_FACTOR = 0.95
ORDINARY_NONREGRESSION_FACTOR = 1.02
RUN_CONTRACT = "darcy_incremental_flux_remote_schur_staged_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--ordinary-reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--raw-warmup-epochs", type=int, default=2)
    parser.add_argument("--samples-per-epoch", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ordinary-validation-cases", type=int, default=60)
    parser.add_argument("--pair-validation-batches", type=int, default=32)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--min-lr", type=float, default=2.0e-6)
    parser.add_argument("--l2sp-weight", type=float, default=1.0e-3)
    parser.add_argument("--full-weight", type=float, default=50.0)
    parser.add_argument("--receiver-weight", type=float, default=2.0)
    parser.add_argument("--balanced-weight", type=float, default=2.0)
    parser.add_argument("--tail-weight", type=float, default=2.0)
    parser.add_argument("--gain-weight", type=float, default=0.05)
    parser.add_argument("--field-weight", type=float, default=2.0)
    parser.add_argument("--reference-weight", type=float, default=10.0)
    parser.add_argument("--schur-weight", type=float, default=0.25)
    parser.add_argument(
        "--trust-scales", type=float, nargs="+",
        default=(0.0625, 0.125, 0.25, 0.5, 1.0),
    )
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _load_communicator(
    path: Path, spec, device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != CONTRACT:
        raise ValueError("incremental training requires a residual-port checkpoint")
    cfg = checkpoint["communicator_spec"]
    if cfg.get("canonical_analyzer_contract") != CANONICAL_ANALYZER_CONTRACT:
        raise ValueError("checkpoint lacks the homogeneous compatibility contract")
    mode = cfg.get("analyzer_mode", "source_linear")
    if mode in (
        "value_flux_relative_green", "value_flux_contextual_relative_green",
        "value_flux_dynamic_relative_green",
        "value_flux_state_gated_dynamic_relative_green",
    ):
        prefix_width = source_linear_signal_width(spec)
        if mode == "value_flux_contextual_relative_green":
            communicator_type = ContextualRelativeGreenFluxResidualPortCommunicator
        elif mode == "value_flux_state_gated_dynamic_relative_green":
            communicator_type = StateGatedDynamicRelativeGreenFluxResidualPortCommunicator
        elif mode == "value_flux_dynamic_relative_green":
            communicator_type = DynamicRelativeGreenFluxResidualPortCommunicator
        else:
            communicator_type = RelativeGreenFluxResidualPortCommunicator
        extra = {}
        if mode in (
            "value_flux_dynamic_relative_green",
            "value_flux_state_gated_dynamic_relative_green",
        ):
            extra = {
                "hidden_width": int(cfg["dynamic_hidden_width"]),
                "residual_rank": int(cfg["dynamic_residual_rank"]),
            }
        if mode == "value_flux_state_gated_dynamic_relative_green":
            extra["value_rank"] = int(cfg["state_value_rank"])
        communicator = communicator_type(
            int(cfg["physical_channels"]), int(cfg["scalar_dim"]),
            spec.code_width, prefix_width,
            token_width=int(cfg["token_width"]), heads=int(cfg["heads"]),
            max_offset=int(cfg.get("relative_green_max_offset", 2)),
            **extra,
        ).to(device)
        communicator.load_state_dict(checkpoint["communicator"], strict=True)
        return communicator, checkpoint
    if cfg.get("kernel_type", "softmax") != "softmax":
        raise ValueError("incremental value-column route requires softmax lineage")
    width = source_linear_signal_width(spec) + (8 if mode == "value_flux" else 0)
    communicator = SourceLinearResidualPortCommunicator(
        int(cfg["physical_channels"]), int(cfg["scalar_dim"]), spec.code_width,
        width, token_width=int(cfg["token_width"]), heads=int(cfg["heads"]),
    ).to(device)
    communicator.analyzer_mode = mode
    communicator.load_state_dict(checkpoint["communicator"], strict=True)
    return communicator, checkpoint


def _aligned(communicator, context, batch, spec, scale) -> Tensor:
    if communicator.analyzer_mode in (
        "value_flux", "value_flux_relative_green",
        "value_flux_contextual_relative_green",
        "value_flux_dynamic_relative_green",
        "value_flux_state_gated_dynamic_relative_green",
    ):
        return measured_q(context, batch, spec, scale, "value_flux")[0]
    if communicator.analyzer_mode == "source_linear":
        return source_linear_features(context, spec, scale)
    raise ValueError("incremental route requires source_linear/value_flux analysis")


def _paired_batch(host: dict[str, Any], device: torch.device):
    moved = move_batch(host, device)
    response = moved.pop("pair_solution_response")[:, None]
    delta = moved.pop("pair_delta_bc")
    target_tile = moved.pop("pair_target_tile")
    moved.pop("pair_amplitude", None)
    pair_count = response.shape[0]
    paired: dict[str, Tensor] = {}
    for name in (*REQUIRED_FIELDS, *FACE_DIRICHLET_FIELDS, "case_index"):
        value = moved[name]
        paired[name] = torch.cat((value, value), 0)
    paired["solution"][pair_count:] += response
    paired["bc_value"][pair_count:] += delta[:, None]
    paired["dirichlet_face_value"][pair_count:] += (
        delta[:, None] * paired["dirichlet_face_mask"][pair_count:]
    )
    return paired, response, target_tile


def _decomposed_fields(
    communicator, chart, context, batch, spec, scale,
    *, max_graph_radius: int | None = None,
) -> dict[str, Tensor]:
    aligned = _aligned(communicator, context, batch, spec, scale)
    prefix_width = source_linear_signal_width(spec)
    total_code, base_code, token = communicator.forward_decomposed(
        context.local, context.physical, context.scalar, context.valid,
        context.cover.grid_shape, prefix_width,
        max_graph_radius=max_graph_radius, aligned_features=aligned,
    )
    if getattr(chart, "chart_type", None) != "geometry_tied_linear_green":
        raise ValueError("incremental route requires the frozen linear geometry chart")
    modes = chart._response_modes(
        context.local, context.physical, context.scalar, context.valid,
    )
    total_correction = torch.einsum(
        "bk,bckhw->bchw", total_code.reshape(-1, spec.code_width), modes,
    ) * context.valid
    base_correction = torch.einsum(
        "bk,bckhw->bchw", base_code.reshape(-1, spec.code_width), modes,
    ) * context.valid
    raw = (context.local + total_correction) * context.valid
    base = (context.local + base_correction) * context.valid
    raw_assembled = context.cover.assemble_weighted(
        raw, context.target_normalized.shape[0],
    ) * batch["mask"]
    corrected, assembled, alpha, compatibility_ratio = (
        damp_affine_compatibility_direction(base, raw, context, batch)
    )
    return {
        "aligned": aligned, "total_code": total_code, "base_code": base_code,
        "token": token, "raw": raw, "raw_assembled": raw_assembled,
        "base": base, "corrected": corrected, "assembled": assembled, "alpha": alpha,
        "compatibility_ratio": compatibility_ratio,
    }


def _per_case_log_relative_square(
    error: Tensor, reference: Tensor, weight: Tensor,
) -> Tensor:
    axes = tuple(range(1, error.ndim))
    numerator = (error.square() * weight).sum(axes)
    denominator = (reference.square() * weight).sum(axes).clamp_min(1.0e-12)
    return torch.log1p(numerator / denominator)


def _reference_fields(reference, chart, context, batch, spec, scale):
    aligned = _aligned(reference, context, batch, spec, scale)
    assembled, code, _, detail = forward_composer(
        reference, chart, context, scale, batch, aligned_features=aligned,
    )
    return assembled, code, detail["corrected"]


def _field_view(fields: dict[str, Tensor], *, deploy: bool) -> tuple[Tensor, Tensor]:
    """Choose raw training proposal or safeguarded deployed realization."""
    if deploy:
        return fields["assembled"], fields["corrected"]
    return fields["raw_assembled"], fields["raw"]


def pair_terms(
    communicator, reference, chart, local, adapter, device, spec, scale, args,
    *, host, deploy: bool = True, max_graph_radius: int | None = None,
) -> dict[str, Tensor]:
    batch, exact_response, target_tile = _paired_batch(host, device)
    pair_count = exact_response.shape[0]
    supervised_context = frozen_context(
        local, adapter, batch, spec, args.core, args.halo,
    )
    context = inference_only_context(supervised_context)
    fields = _decomposed_fields(
        communicator, chart, context, batch, spec, scale,
        max_graph_radius=max_graph_radius,
    )
    # At the exact zero-suffix warm start the constrained affine minimizer has
    # a null direction and selects alpha=0, which correctly preserves the
    # accepted deployment field but cannot seed learning.  Optimize the raw
    # proposed direction; use the safeguarded field for every validation and
    # deployed metric.  The raw Schur-cycle term teaches a direction for which
    # the target-free retraction can subsequently choose alpha>0.
    assembled, corrected = _field_view(fields, deploy=deploy)
    predicted_full = adapter.statistics.solution_std * (
        assembled[pair_count:] - assembled[:pair_count]
    )
    full_weight = batch["mask"][:pair_count]
    exact_tiles = context.cover.tile(exact_response)
    weight_tiles = context.cover.tile(full_weight)
    tile_count = context.cover.tile_count
    target_index = target_tile[:, 0] * context.cover.grid_shape[1] + target_tile[:, 1]
    flat_base = torch.arange(pair_count, device=device) * tile_count + target_index
    flat_changed = flat_base + pair_count * tile_count
    exact_receiver = exact_tiles[flat_base]
    receiver_weight = weight_tiles[flat_base]
    # Match the frozen gate exactly: receiver response comes from the corrected
    # local tile, never from re-tiling the assembled field.
    predicted_receiver = adapter.statistics.solution_std * (
        corrected[flat_changed] - corrected[flat_base]
    )
    receiver_error = predicted_receiver - exact_receiver
    receiver_log = _per_case_log_relative_square(
        receiver_error, exact_receiver, receiver_weight,
    )
    tail_count = max(1, math.ceil(receiver_log.numel() / 4))
    truth_flat = (exact_receiver * receiver_weight).flatten(1)
    pred_flat = (predicted_receiver * receiver_weight).flatten(1)
    gain = (truth_flat * pred_flat).sum(1) / truth_flat.square().sum(1).clamp_min(1e-12)
    gain_loss = F.smooth_l1_loss(gain, torch.ones_like(gain))
    gain_violation = torch.relu(0.05 - gain) + torch.relu(gain - 1.95)
    # Linear tails prevent a single near-zero-response pair from dominating
    # the field and response objectives by orders of magnitude.
    gain_hinge = F.smooth_l1_loss(
        gain_violation, torch.zeros_like(gain_violation), beta=0.25,
    )
    reference_assembled, reference_code, _ = _reference_fields(
        reference, chart, context, batch, spec, scale,
    )
    absolute_field = masked_mse(
        assembled, supervised_context.target_normalized, batch["mask"],
    )
    reference_field = masked_mse(
        assembled[:pair_count], reference_assembled[:pair_count],
        batch["mask"][:pair_count],
    )
    reference_code_loss = F.smooth_l1_loss(
        fields["total_code"][:pair_count], reference_code[:pair_count],
    )
    q0, support = measured_q(context, batch, spec, scale, "value_flux")
    q1 = q_from_corrected(
        corrected, context, batch, spec, scale, "value_flux",
    )
    axes = tuple(range(1, q0.ndim))
    schur_cycle = (
        (q1.square() * support).sum(axes)
        / (q0.square() * support).sum(axes).clamp_min(1e-12)
    ).mean()
    return {
        "full_response_relative_square": _relative_square(
            predicted_full - exact_response, exact_response, full_weight,
        ),
        "receiver_response_relative_square": _relative_square(
            receiver_error, exact_receiver, receiver_weight,
        ),
        "receiver_balanced_log_relative_square": receiver_log.mean(),
        "receiver_tail_log_relative_square": receiver_log.topk(tail_count).values.mean(),
        "receiver_gain_loss": gain_loss,
        "receiver_gain_hinge": gain_hinge,
        "receiver_gain_mean": gain.mean(),
        "absolute_field": absolute_field,
        "reference_field": reference_field,
        "reference_code": reference_code_loss,
        "schur_cycle_relative_square": schur_cycle,
        "affine_alpha_mean": fields["alpha"].mean(),
        "affine_compatibility_ratio": fields["compatibility_ratio"].mean(),
        "increment_code_rms": (
            fields["total_code"] - fields["base_code"]
        ).square().mean().sqrt(),
    }


def objective(terms: dict[str, Tensor], args, suffix_penalty: Tensor) -> Tensor:
    return (
        args.full_weight * terms["full_response_relative_square"]
        + args.receiver_weight * terms["receiver_response_relative_square"]
        + args.balanced_weight * terms["receiver_balanced_log_relative_square"]
        + args.tail_weight * terms["receiver_tail_log_relative_square"]
        + args.gain_weight * (
            terms["receiver_gain_loss"] + terms["receiver_gain_hinge"]
        )
        + args.field_weight * terms["absolute_field"]
        + args.reference_weight * (
            terms["reference_field"] + terms["reference_code"]
        )
        + args.schur_weight * terms["schur_cycle_relative_square"]
        + args.l2sp_weight * suffix_penalty
    )


@torch.no_grad()
def validate_pairs(loader, *model_args, maximum_batches: int):
    communicator = model_args[0]
    communicator.eval(); totals: dict[str, float] = {}; batches = 0
    for host in loader:
        if batches >= maximum_batches:
            break
        values = pair_terms(*model_args, host=host)
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        batches += 1
    if batches == 0:
        raise RuntimeError("empty held-out train-pair validation")
    return {name: value / batches for name, value in totals.items()}


@torch.no_grad()
def ordinary_metrics(
    communicator, chart, local, adapter, loader, device, spec, scale, args,
    *, incremental: bool,
) -> dict[str, float]:
    communicator.eval(); totals: dict[str, float] = {}; cases = 0
    for host in loader:
        if cases >= args.ordinary_validation_cases:
            break
        batch = move_batch(host, device)
        original = batch["solution"].shape[0]
        keep = min(original, args.ordinary_validation_cases - cases)
        for name, value in list(batch.items()):
            if isinstance(value, Tensor) and value.shape[0] == original:
                batch[name] = value[:keep]
        supervised = frozen_context(local, adapter, batch, spec, args.core, args.halo)
        context = inference_only_context(supervised)
        if incremental:
            fields = _decomposed_fields(
                communicator, chart, context, batch, spec, scale,
            )
            assembled, corrected = fields["assembled"], fields["corrected"]
        else:
            assembled, _, corrected = _reference_fields(
                communicator, chart, context, batch, spec, scale,
            )
        consensus = context.cover.tile(assembled)
        energy, residual = physics_losses(
            assembled, batch, adapter.statistics.solution_std,
        )
        value, flux = interface_losses(corrected, assembled, context, batch)
        q0, support = measured_q(context, batch, spec, scale, "value_flux")
        q1 = q_from_corrected(corrected, context, batch, spec, scale, "value_flux")
        axes = tuple(range(1, q0.ndim))
        cycle = (
            (q1.square() * support).sum(axes)
            / (q0.square() * support).sum(axes).clamp_min(1e-12)
        ).mean()
        values = {
            "relative_l2": relative_l2(
                assembled * adapter.statistics.solution_std,
                batch["solution"], batch["mask"],
            ).mean(),
            "assembled": masked_mse(
                assembled, supervised.target_normalized, batch["mask"],
            ),
            "local": masked_mse(
                corrected, supervised.target_tiles, context.valid,
            ),
            "consensus": masked_mse(corrected, consensus, context.valid),
            "energy": energy, "residual": residual,
            "value_interface": value, "flux_interface": flux,
            "complete_q_cycle_square": cycle,
        }
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value) * keep
        cases += keep
    if cases == 0:
        raise RuntimeError("empty ordinary validation")
    return {name: value / cases for name, value in totals.items()}


def _suffix_parameters(communicator, prefix_width: int):
    parameters = (
        communicator.code_map.local_value.weight,
        communicator.code_map.global_kernel.value.weight,
    )
    masks = []
    for parameter in parameters:
        mask = torch.zeros_like(parameter); mask[:, prefix_width:] = 1
        parameter.requires_grad_(True)
        parameter.register_hook(lambda gradient, mask=mask: gradient * mask)
        masks.append(mask)
    return parameters, masks


def _set_flux_scale(parameters, initial, trained, prefix_width: int, scale: float):
    with torch.no_grad():
        for parameter, origin, current in zip(parameters, initial, trained, strict=True):
            parameter.copy_(origin)
            parameter[:, prefix_width:].copy_(
                origin[:, prefix_width:] + scale * (
                    current[:, prefix_width:] - origin[:, prefix_width:]
                )
            )


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs <= 0 or args.samples_per_epoch <= 0:
        raise ValueError("training extent must be positive")
    if not 0 <= args.raw_warmup_epochs <= args.epochs:
        raise ValueError("raw warmup must lie between zero and total epochs")
    if any(not 0 < value <= 1 for value in args.trust_scales):
        raise ValueError("trust scales must lie in (0,1]")
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, chart_checkpoint = load_frozen_chart(args.chart_checkpoint, device)
    communicator, initial_checkpoint = _load_communicator(
        args.initial_checkpoint, spec, device,
    )
    reference, reference_checkpoint = _load_communicator(
        args.ordinary_reference_checkpoint, spec, device,
    )
    for checkpoint in (initial_checkpoint, reference_checkpoint):
        if checkpoint["source_local_sha256"] != sha256(args.local_checkpoint) or \
                checkpoint["source_chart_sha256"] != sha256(args.chart_checkpoint):
            raise ValueError("communicator/local/chart lineage differs")
    if communicator.analyzer_mode != "value_flux":
        raise ValueError("initial checkpoint must be the zero-flux value+flux lift")
    if reference.analyzer_mode != "source_linear":
        raise ValueError("ordinary reference must be the accepted value-only map")
    prefix_width = source_linear_signal_width(spec)
    flux_width = communicator.aligned_width - prefix_width
    if flux_width != 8:
        raise ValueError("incremental route expects exactly eight flux coordinates")
    for parameter in communicator.parameters():
        parameter.requires_grad_(False)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    parameters, _ = _suffix_parameters(communicator, prefix_width)
    initial_suffix_parameters = [parameter.detach().clone() for parameter in parameters]
    if any(torch.count_nonzero(value[:, prefix_width:]) for value in initial_suffix_parameters):
        raise ValueError("warm-start flux columns are not exactly zero")
    reference.eval(); chart.eval(); local.eval()

    train_pairs = RemotePairDataset(args.data, args.pairs, holdout=False)
    heldout_pairs = RemotePairDataset(args.data, args.pairs, holdout=True)
    ordinary_set = H5DarcyDataset(args.data, "validation")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_pairs, batch_size=args.batch_size,
        sampler=RandomSampler(
            train_pairs, replacement=True, num_samples=args.samples_per_epoch,
            generator=generator,
        ),
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    pair_loader = DataLoader(
        heldout_pairs, batch_size=args.batch_size, shuffle=False,
        num_workers=min(args.workers, 2), pin_memory=device.type == "cuda",
    )
    ordinary_loader = DataLoader(
        ordinary_set, batch_size=args.batch_size, shuffle=False,
        num_workers=min(args.workers, 2), pin_memory=device.type == "cuda",
    )
    model_args = (
        communicator, reference, chart, local, adapter, device, spec, scale, args,
    )
    baseline_pair = validate_pairs(
        pair_loader, *model_args, maximum_batches=args.pair_validation_batches,
    )
    immutable_ordinary = ordinary_metrics(
        reference, chart, local, adapter, ordinary_loader, device, spec, scale,
        args, incremental=False,
    )
    warm_start_ordinary = ordinary_metrics(
        communicator, chart, local, adapter, ordinary_loader, device, spec, scale,
        args, incremental=True,
    )
    warm_delta = max(
        abs(warm_start_ordinary[name] - immutable_ordinary[name])
        for name in immutable_ordinary
    )
    if warm_delta > 1e-6:
        raise RuntimeError(
            f"zero-flux warm start does not reproduce immutable reference: {warm_delta}"
        )
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.min_lr,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best = math.inf; selected_epoch = None; selected_trust = None
    for epoch in range(1, args.epochs + 1):
        communicator.train(); totals: dict[str, float] = {}; batches = 0
        deploy_during_training = epoch > args.raw_warmup_epochs
        for host in train_loader:
            optimizer.zero_grad(set_to_none=True)
            terms = pair_terms(
                *model_args, host=host, deploy=deploy_during_training,
            )
            suffix_penalty = sum(
                parameter[:, prefix_width:].square().mean()
                for parameter in parameters
            )
            loss = objective(terms, args, suffix_penalty)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite incremental flux loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step(); batches += 1
            # Prefix columns are an immutable copy of the accepted operator.
            with torch.no_grad():
                for parameter, initial in zip(
                    parameters, initial_suffix_parameters, strict=True,
                ):
                    parameter[:, :prefix_width].copy_(initial[:, :prefix_width])
            for name, value in {"objective": loss, **terms}.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
        scheduler.step()
        trained = [parameter.detach().clone() for parameter in parameters]
        for trust in sorted(set(args.trust_scales)):
            _set_flux_scale(
                parameters, initial_suffix_parameters, trained, prefix_width, trust,
            )
            pair_validation = validate_pairs(
                pair_loader, *model_args,
                maximum_batches=args.pair_validation_batches,
            )
            ordinary = ordinary_metrics(
                communicator, chart, local, adapter, ordinary_loader, device,
                spec, scale, args, incremental=True,
            )
            ordinary_gates = {
                name: ordinary[name]
                <= ORDINARY_NONREGRESSION_FACTOR * immutable_ordinary[name]
                for name in immutable_ordinary
            }
            response_gates = {
                "full": pair_validation["full_response_relative_square"]
                    <= PAIR_IMPROVEMENT_FACTOR
                    * baseline_pair["full_response_relative_square"],
                "receiver": pair_validation["receiver_response_relative_square"]
                    <= PAIR_IMPROVEMENT_FACTOR
                    * baseline_pair["receiver_response_relative_square"],
                "receiver_balanced":
                    pair_validation["receiver_balanced_log_relative_square"]
                    <= PAIR_IMPROVEMENT_FACTOR
                    * baseline_pair["receiver_balanced_log_relative_square"],
                "receiver_tail":
                    pair_validation["receiver_tail_log_relative_square"]
                    <= PAIR_IMPROVEMENT_FACTOR
                    * baseline_pair["receiver_tail_log_relative_square"],
                "gain": 0.05 <= pair_validation["receiver_gain_mean"] <= 1.95,
                "schur": pair_validation["schur_cycle_relative_square"] <= 1.0,
            }
            clears = all(ordinary_gates.values()) and all(response_gates.values())
            score = (
                pair_validation["receiver_response_relative_square"]
                + pair_validation["receiver_balanced_log_relative_square"]
                + 0.5 * pair_validation["receiver_tail_log_relative_square"]
                + 0.5 * pair_validation["full_response_relative_square"]
            )
            row = {
                "epoch": epoch, "trust_scale": trust,
                "training_field": (
                    "affine_retracted" if deploy_during_training else "raw_warmup"
                ),
                "lr": optimizer.param_groups[0]["lr"],
                "selection_score": score, "clears": clears,
                "ordinary_gates": ordinary_gates,
                "response_gates": response_gates,
                **{f"train_{name}": value / batches for name, value in totals.items()},
                **{f"pair_validation_{name}": value for name, value in pair_validation.items()},
                **{f"ordinary_validation_{name}": value for name, value in ordinary.items()},
            }
            history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
            if clears and score < best:
                best = score; selected_epoch = epoch; selected_trust = trust
                payload = dict(initial_checkpoint)
                cfg = dict(payload["communicator_spec"])
                cfg.update({
                    "analyzer_mode": "value_flux",
                    "incremental_flux_warm_start": True,
                    "value_prefix_width": prefix_width,
                    "flux_suffix_width": flux_width,
                    "frozen_value_prefix": True,
                    "deployment_affine_retraction": "accepted_value_to_flux_corrected",
                    "receiver_observable": "corrected_local_target_tile",
                    "relative_attention_count": 1,
                })
                payload.update({
                    "contract": CONTRACT, "phase": RUN_CONTRACT,
                    "epoch": epoch, "selected_trust_scale": trust,
                    "communicator": communicator.state_dict(),
                    "communicator_spec": cfg,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "initial_communicator_checkpoint": str(args.initial_checkpoint.resolve()),
                    "initial_communicator_sha256": sha256(args.initial_checkpoint),
                    "ordinary_reference_checkpoint": str(
                        args.ordinary_reference_checkpoint.resolve()
                    ),
                    "ordinary_reference_sha256": sha256(
                        args.ordinary_reference_checkpoint
                    ),
                    "remote_pair_corpus": str(args.pairs.resolve()),
                    "remote_pair_corpus_sha256": sha256(args.pairs),
                    "training_splits": ["train", "validation"],
                    "pair_supervision_split": "train only",
                    "exact_global_solution_used_at_inference": False,
                    "discrete_operator_used_at_inference": False,
                    "fixed_context_q_linearity_preserved_before_retraction": True,
                    "zero_flux_recovers_accepted_value_map": True,
                    "one_shared_global_kernel": True,
                    "ordinary_nonregression_factor": ORDINARY_NONREGRESSION_FACTOR,
                    "pair_improvement_factor": PAIR_IMPROVEMENT_FACTOR,
                    "immutable_ordinary": immutable_ordinary,
                    "baseline_pair": baseline_pair,
                    "selection_clears": True,
                    "test_subsets_opened": False,
                    "history": history, "args": vars(args),
                })
                torch.save(payload, args.output_dir / "checkpoint.pt")
        _set_flux_scale(
            parameters, initial_suffix_parameters, trained, prefix_width, 1.0,
        )
        last = dict(initial_checkpoint)
        last_cfg = dict(last["communicator_spec"])
        last_cfg.update({
            "analyzer_mode": "value_flux",
            "incremental_flux_warm_start": True,
            "value_prefix_width": prefix_width,
            "flux_suffix_width": flux_width,
            "frozen_value_prefix": True,
            "deployment_affine_retraction": "accepted_value_to_flux_corrected",
            "receiver_observable": "corrected_local_target_tile",
            "relative_attention_count": 1,
        })
        last.update({
            "contract": CONTRACT, "phase": RUN_CONTRACT, "epoch": epoch,
            "communicator": communicator.state_dict(),
            "communicator_spec": last_cfg, "history": history,
            "ordinary_reference_checkpoint": str(
                args.ordinary_reference_checkpoint.resolve()
            ),
            "ordinary_reference_sha256": sha256(
                args.ordinary_reference_checkpoint
            ),
            "one_shared_global_kernel": True,
            "exact_global_solution_used_at_inference": False,
            "discrete_operator_used_at_inference": False,
            "test_subsets_opened": False, "args": vars(args),
        })
        torch.save(last, args.output_dir / "last_checkpoint.pt")
    train_pairs.close(); heldout_pairs.close(); ordinary_set.close()
    result = {
        "contract": RUN_CONTRACT, "selected_epoch": selected_epoch,
        "selected_trust_scale": selected_trust,
        "best_selection_score": None if selected_epoch is None else best,
        "immutable_ordinary": immutable_ordinary,
        "warm_start_ordinary": warm_start_ordinary,
        "warm_start_max_abs_metric_delta": warm_delta,
        "baseline_pair": baseline_pair,
        "test_subsets_opened": False, "history": history,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8",
    )
    with (args.output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        flat = [{
            key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
            for key, value in row.items()
        } for row in history]
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    if selected_epoch is None:
        raise RuntimeError("no incremental flux correction cleared frozen gates")
    return result


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2, default=str))
