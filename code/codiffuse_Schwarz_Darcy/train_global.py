"""Train a matched full-domain Darcy baseline from scratch."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .adapter import (
    DarcyAdapter,
    DarcyStatistics,
    H5DarcyDataset,
    batch_to_transition,
    compute_statistics,
)
from .evaluate import (
    energy_relative_error,
    initial_state,
    move_batch,
    relative_l2,
    residual_rms,
)
from .global_operator import MaskedTransolverOperator
from .masked_anchor_operator import MaskedAnchorOperator
from .low_rank_adaptation import (
    adaptation_state_dict,
    install_low_rank_adapters,
    merged_state_dict,
)
from .train import arbitrary_current, make_loader, seed_everything
from codiffuse_Schwarz.global_vit import GlobalPatchTransformer
from .budget_protocol import apply_training_manifest, shared_field_loss, validate_resume_protocol


Tensor = torch.Tensor
LOCAL_PARAMETER_REFERENCE = 3_868_066
TRAINING_DATA_SCHEMAS = frozenset({
    "darcy_full_domain_cartesian_v1",
    "darcy_full_domain_cartesian_v2_face_dirichlet",
    "darcy_domain_extension_cartesian_v2_fixed_dx",
})
TRAINING_DATA_ROLE = "shared_full_domain_finetune"
LOCAL_PRETRAINING_SCHEMAS = frozenset({
    "darcy_complex_cartesian_v1",
    "darcy_complex_cartesian_v2_face_dirichlet",
})
CHECKPOINT_CONTRACT = "darcy_global_masked_transolver_v2"
VIT_CHECKPOINT_CONTRACT = "darcy_global_patch_transformer_v1"
ANCHOR_CHECKPOINT_CONTRACT = "darcy_global_masked_anchor_operator_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=Path(
            "codiffuse_Schwarz_Darcy/data/darcy_full_domain_96_v1.h5"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stats-cases", type=int, default=512)
    parser.add_argument(
        "--statistics-checkpoint", type=Path,
        help=(
            "pretrained Darcy checkpoint whose DarcyStatistics are reused for "
            "matched normalization; adaptation defaults to --initial-checkpoint"
        ),
    )
    parser.add_argument("--max-train-cases", type=int)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--field-loss", choices=("legacy", "shared_field_v1"), default="legacy")
    parser.add_argument("--position-encoding", default="normalized_linear_fourier_v1",
                        choices=("normalized_linear_fourier_v1", "fixed_tile_fourier_v1",
                                 "fixed_tile_fourier_locality_v1"))
    parser.add_argument("--max-val-cases", type=int)
    parser.add_argument("--validation-sweeps", type=int, default=4)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--local-weight", type=float, default=1.0)
    parser.add_argument(
        "--current-policy", choices=("arbitrary", "zero"), default="arbitrary",
        help=(
            "train an iterative arbitrary-current correction, or the direct "
            "steady solution from a fixed zero state"
        ),
    )
    parser.add_argument(
        "--use-discrete-residual", action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "expose the finite-volume residual as an input channel; disable "
            "this for the PDE-agnostic whole-domain comparison"
        ),
    )
    parser.add_argument(
        "--selection-residual-weight", type=float, default=0.1,
        help="validation residual weight added to relative L2 for selection",
    )
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--min-lr", type=float, default=2.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--slices", type=int, default=32)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument(
        "--architecture", choices=("transolver", "global_vit", "point_anchor"),
        default="transolver",
        help="global model family",
    )
    parser.add_argument(
        "--patch-size", type=int, default=8,
        help="non-overlapping square patch width for --architecture global_vit",
    )
    parser.add_argument(
        "--anchors", type=int, default=64,
        help="learned anchor count for --architecture point_anchor",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initial-checkpoint", type=Path,
        help="load model weights only, resetting optimizer/schedule for a new stage",
    )
    parser.add_argument(
        "--adaptation-mode", choices=("full", "low_rank"), default="full",
        help=(
            "parameters updated after loading --initial-checkpoint; low_rank "
            "freezes the source model and trains centered Linear-map deltas"
        ),
    )
    parser.add_argument(
        "--low-rank", type=int, default=4,
        help="rank used by --adaptation-mode low_rank",
    )
    parser.add_argument(
        "--low-rank-alpha", type=float,
        help="low-rank update scale numerator (defaults to the rank)",
    )
    parser.add_argument(
        "--low-rank-dropout", type=float, default=0.0,
        help="dropout on the low-rank input branch",
    )
    parser.add_argument(
        "--low-rank-placement",
        choices=("all_linear", "attention_ffn"),
        default="all_linear",
        help=(
            "eligible maps for low-rank adaptation; attention_ffn excludes "
            "embeddings, normalization, biases, and output decoders"
        ),
    )
    parser.add_argument(
        "--adaptation-provenance", type=Path,
        help="optional validation-selection artifact recorded in checkpoints",
    )
    parser.add_argument(
        "--training-stage", choices=("full_domain", "local_pretraining"),
        default="full_domain",
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def build_global_model(
    args: argparse.Namespace, adapter: DarcyAdapter,
) -> MaskedTransolverOperator | GlobalPatchTransformer | MaskedAnchorOperator:
    architecture = getattr(args, "architecture", "transolver")
    if architecture == "global_vit":
        return GlobalPatchTransformer(
            input_channels=adapter.spatial_condition_channels,
            output_channels=1,
            dimensions=2,
            patch_size=args.patch_size,
            width=args.width,
            depth=args.depth,
            heads=args.heads,
            mlp_ratio=args.mlp_ratio,
            scalar_dim=adapter.scalar_condition_dim,
            dropout=args.dropout,
            position_encoding=getattr(args, "position_encoding", "normalized_linear_fourier_v1"),
        )
    if architecture == "point_anchor":
        return MaskedAnchorOperator(
            input_channels=adapter.spatial_condition_channels,
            scalar_dim=adapter.scalar_condition_dim,
            width=args.width,
            depth=args.depth,
            heads=args.heads,
            anchors=args.anchors,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
        )
    return MaskedTransolverOperator(
        input_channels=adapter.spatial_condition_channels,
        scalar_dim=adapter.scalar_condition_dim,
        width=args.width,
        depth=args.depth,
        heads=args.heads,
        slices=args.slices,
        mlp_ratio=args.mlp_ratio,
    )


def checkpoint_contract(args: argparse.Namespace) -> str:
    architecture = getattr(args, "architecture", "transolver")
    if architecture == "global_vit":
        return VIT_CHECKPOINT_CONTRACT
    if architecture == "point_anchor":
        return ANCHOR_CHECKPOINT_CONTRACT
    return CHECKPOINT_CONTRACT


def load_matched_statistics(path: Path | str) -> DarcyStatistics:
    """Load normalization from a recognized pretrained Darcy checkpoint."""

    path = Path(path).resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    supported = {
        "darcy_arbitrary_current_direct_endpoint_v1",
        "darcy_sni_face_boundary_to_solution_v1",
        CHECKPOINT_CONTRACT,
        VIT_CHECKPOINT_CONTRACT,
        ANCHOR_CHECKPOINT_CONTRACT,
    }
    if checkpoint.get("contract") not in supported:
        raise ValueError(
            "statistics checkpoint must be a recognized pretrained Darcy model"
        )
    values = checkpoint.get("statistics")
    if not isinstance(values, dict):
        raise ValueError("statistics checkpoint has no DarcyStatistics payload")
    return DarcyStatistics.from_json(values)


def global_conditions(
    adapter: DarcyAdapter,
    batch: dict[str, Any],
    current: Tensor,
    previous: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    transition = batch_to_transition(
        batch, current, previous=current if previous is None else previous,
        target=batch.get("solution"),
    )
    phase = torch.ones(
        current.shape[0], device=current.device, dtype=current.dtype,
    )
    condition = adapter.conditions(transition, phase, 1.0)
    return condition.spatial, condition.scalars, adapter.normalized_target(transition)


def global_step(
    model: MaskedTransolverOperator | GlobalPatchTransformer | MaskedAnchorOperator,
    adapter: DarcyAdapter,
    batch: dict[str, Any],
    current: Tensor,
    previous: Tensor,
    relaxation: float = 1.0,
) -> Tensor:
    transition = batch_to_transition(
        batch, current, previous=previous, target=batch.get("solution"),
    )
    phase = torch.ones(
        current.shape[0], device=current.device, dtype=current.dtype,
    )
    condition = adapter.conditions(transition, phase, 1.0)
    endpoint = model(condition.spatial, condition.scalars, batch["mask"])
    proposal = adapter.apply_endpoint(transition, endpoint)
    return current + relaxation * (proposal - current)


def matched_endpoint_loss(
    prediction: Tensor, target: Tensor, local_weight: float,
) -> tuple[Tensor, Tensor]:
    """Match one-tile local training: endpoint + local-endpoint MSE.

    Each record is a full physical domain. For the global control, endpoint
    and local-endpoint are the same tensor and overlap loss is structurally
    absent. The communicated-local comparator consumes these same records but
    exposes its tile overlap terms.
    """
    if local_weight < 0.0:
        raise ValueError("local_weight must be nonnegative")
    endpoint = (prediction - target).square().mean()
    return (1.0 + local_weight) * endpoint, endpoint


@torch.no_grad()
def validate_global(
    model: MaskedTransolverOperator | GlobalPatchTransformer | MaskedAnchorOperator,
    adapter: DarcyAdapter,
    loader,
    device: torch.device,
    statistics: DarcyStatistics,
    sweeps: int,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    values: dict[str, list[Tensor]] = {
        "sweep1_relative_l2": [], "final_relative_l2": [],
        "final_energy_relative_error": [], "final_residual_rms": [],
    }
    for batch_index, host_batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = move_batch(host_batch, device)
        current = initial_state(batch, statistics, "zero")
        previous = current.clone()
        for sweep in range(sweeps):
            next_state = global_step(model, adapter, batch, current, previous)
            if sweep == 0:
                values["sweep1_relative_l2"].append(
                    relative_l2(next_state, batch["solution"], batch["mask"]).cpu()
                )
            previous, current = current, next_state
        values["final_relative_l2"].append(
            relative_l2(current, batch["solution"], batch["mask"]).cpu()
        )
        values["final_energy_relative_error"].append(
            energy_relative_error(
                current, batch["solution"], batch["logk"], batch["mask"],
            ).cpu()
        )
        values["final_residual_rms"].append(residual_rms(current, batch).cpu())
    if not values["final_relative_l2"]:
        raise RuntimeError("global validation produced no batches")
    return {name: float(torch.cat(items).mean()) for name, items in values.items()}


def train_global(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.epochs, args.samples_per_epoch, args.batch_size) <= 0:
        raise ValueError("training sizes must be positive")
    current_policy = str(getattr(args, "current_policy", "arbitrary"))
    if current_policy not in ("arbitrary", "zero"):
        raise ValueError("current_policy must be arbitrary or zero")
    use_discrete_residual = bool(
        getattr(args, "use_discrete_residual", True)
    )
    selection_residual_weight = float(
        getattr(args, "selection_residual_weight", 0.1)
    )
    if selection_residual_weight < 0.0:
        raise ValueError("selection_residual_weight must be nonnegative")
    if current_policy == "zero" and args.validation_sweeps != 1:
        raise ValueError(
            "a direct-zero whole-domain operator must be selected in one pass"
        )
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    train_dataset = H5DarcyDataset(args.data, "train")
    val_dataset = H5DarcyDataset(args.data, "validation")
    training_stage = getattr(args, "training_stage", "full_domain")
    initial_checkpoint = getattr(args, "initial_checkpoint", None)
    if not args.resume and any(
        (args.output_dir / name).exists()
        for name in ("checkpoint.pt", "last_checkpoint.pt", "history.csv")
    ):
        raise FileExistsError(
            f"output directory {args.output_dir} already contains a training run"
        )
    expected_schemas = (
        TRAINING_DATA_SCHEMAS
        if training_stage == "full_domain" else LOCAL_PRETRAINING_SCHEMAS
    )
    expected_role = (
        TRAINING_DATA_ROLE if training_stage == "full_domain" else None
    )
    observed_schemas: set[str] = set()
    for name, dataset in (("train", train_dataset), ("validation", val_dataset)):
        schema = str(dataset.attrs.get("schema_version", ""))
        observed_schemas.add(schema)
        role_value = dataset.attrs.get("dataset_role")
        role = None if role_value is None else str(role_value)
        if schema not in expected_schemas or role != expected_role:
            raise ValueError(
                f"{name} does not match training stage {training_stage!r}; "
                f"expected ({sorted(expected_schemas)!r}, {expected_role!r}), "
                f"got ({schema!r}, {role!r})"
            )
    if len(observed_schemas) != 1:
        raise ValueError(
            "train and validation splits use different schema versions: "
            f"{sorted(observed_schemas)!r}"
        )
    training_data_schema = next(iter(observed_schemas))
    args.training_subset = apply_training_manifest(
        train_dataset, getattr(args, "train_manifest", None), args.max_train_cases,
    )
    if args.max_train_cases is not None and args.training_subset is None:
        train_dataset.length = min(train_dataset.length, args.max_train_cases)
    if args.max_val_cases is not None:
        val_dataset.length = min(val_dataset.length, args.max_val_cases)
    statistics_checkpoint = getattr(args, "statistics_checkpoint", None)
    if statistics_checkpoint is None and initial_checkpoint is not None:
        statistics_checkpoint = initial_checkpoint
    if statistics_checkpoint is None and args.resume is not None:
        statistics_checkpoint = args.resume
    if statistics_checkpoint is None:
        statistics = compute_statistics(
            train_dataset, args.stats_cases, args.seed,
        )
        statistics_provenance = "computed_from_current_train_split"
    else:
        statistics = load_matched_statistics(statistics_checkpoint)
        statistics_provenance = "loaded_from_matched_local_checkpoint"
    adapter = DarcyAdapter(
        statistics, use_discrete_residual=use_discrete_residual,
    )
    model = build_global_model(args, adapter).to(device)
    contract = checkpoint_contract(args)
    base_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    adaptation_mode = getattr(args, "adaptation_mode", "full")
    adaptation_provenance = getattr(args, "adaptation_provenance", None)
    if adaptation_mode == "low_rank" and initial_checkpoint is None and args.resume is None:
        raise ValueError("low_rank adaptation requires --initial-checkpoint or --resume")
    if adaptation_mode == "low_rank" and training_stage != "full_domain":
        raise ValueError("low_rank adaptation is defined only for full-domain transfer")
    history: list[dict[str, float]] = []
    best_score = math.inf
    start_epoch = 0
    low_rank_summary = None
    low_rank_zero_delta_max_abs: float | None = None
    if args.resume is not None and initial_checkpoint is not None:
        raise ValueError("resume and initial_checkpoint are mutually exclusive")
    if initial_checkpoint is not None:
        checkpoint = torch.load(
            initial_checkpoint, map_location="cpu", weights_only=False,
        )
        if checkpoint.get("contract") != contract:
            raise ValueError("initial checkpoint uses another contract")
        if checkpoint["statistics"] != statistics.to_json():
            raise ValueError("initial checkpoint statistics differ from current data")
        if checkpoint.get("model_spec") != model.model_spec():
            raise ValueError("initial checkpoint model specification differs")
        model.load_state_dict(checkpoint["model"])
    if adaptation_mode == "low_rank":
        source_state = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        low_rank_summary = install_low_rank_adapters(
            model,
            rank=args.low_rank,
            alpha=args.low_rank_alpha,
            dropout=args.low_rank_dropout,
            placement=args.low_rank_placement,
        )
        initial_merged = merged_state_dict(model)
        low_rank_zero_delta_max_abs = max(
            float((initial_merged[name].cpu() - source_state[name].cpu()).abs().max())
            for name in source_state
        )
        if low_rank_zero_delta_max_abs != 0.0:
            raise RuntimeError("new low-rank branch does not recover the source model")
    optimized_parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if not optimized_parameters:
        raise RuntimeError("training stage has no trainable parameters")
    optimizer = torch.optim.AdamW(
        optimized_parameters, lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr,
    )
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.training_subset is not None:
            validate_resume_protocol(checkpoint.get("args", {}), args)
        if checkpoint.get("model_spec") != model.model_spec():
            raise ValueError("resume model specification differs")
        if checkpoint.get("contract") != contract:
            raise ValueError("resume checkpoint uses another contract")
        if checkpoint["statistics"] != statistics.to_json():
            raise ValueError("resume statistics differ from current data")
        if checkpoint.get("adaptation_mode", "full") != adaptation_mode:
            raise ValueError("resume checkpoint uses another adaptation mode")
        if adaptation_mode == "low_rank":
            if checkpoint.get("low_rank", {}).get("rank") != args.low_rank:
                raise ValueError("resume checkpoint uses another low-rank rank")
            if checkpoint.get("low_rank", {}).get("placement", "all_linear") != args.low_rank_placement:
                raise ValueError("resume checkpoint uses another low-rank placement")
            model.load_state_dict(checkpoint["adaptation_model"])
            low_rank_zero_delta_max_abs = checkpoint.get(
                "low_rank_zero_delta_max_abs",
            )
        else:
            model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        history = list(checkpoint.get("history", []))
        best_score = min(
            (row.get("selection_score", math.inf) for row in history),
            default=math.inf,
        )
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.workers, device,
        args.samples_per_epoch,
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, False, args.workers, device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        totals: list[float] = []
        endpoints: list[float] = []
        generator = torch.Generator(device=device).manual_seed(args.seed + 10_007 * epoch)
        for host_batch in train_loader:
            batch = move_batch(host_batch, device)
            current = (
                torch.zeros_like(batch["solution"])
                if current_policy == "zero" else
                arbitrary_current(batch, statistics, generator)
            )
            spatial, scalars, target = global_conditions(adapter, batch, current)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(spatial, scalars, batch["mask"])
                if getattr(args, "field_loss", "legacy") == "shared_field_v1":
                    loss, endpoint = shared_field_loss(prediction, target, batch["mask"], batch["sdf"])
                else:
                    loss, endpoint = matched_endpoint_loss(
                        prediction, target, args.local_weight,
                    )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optimized_parameters, 1.0)
            optimizer.step()
            totals.append(float(loss.detach()))
            endpoints.append(float(endpoint.detach()))
        scheduler.step()
        validation = validate_global(
            model, adapter, val_loader, device, statistics,
            args.validation_sweeps, args.validation_batches,
        )
        selection_score = validation["final_relative_l2"] \
            + selection_residual_weight * validation["final_residual_rms"]
        record = {
            "epoch": float(epoch),
            "seconds": time.perf_counter() - epoch_started,
            "train_total": float(np.mean(totals)),
            "train_endpoint": float(np.mean(endpoints)),
            "train_local_endpoint": float(np.mean(endpoints)),
            "train_overlap": 0.0,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "selection_score": selection_score,
            **validation,
        }
        history.append(record)
        low_rank_metadata = None if low_rank_summary is None else {
            "rank": low_rank_summary.rank,
            "placement": low_rank_summary.placement,
            "alpha": float(
                args.low_rank if args.low_rank_alpha is None else args.low_rank_alpha
            ),
            "dropout": float(args.low_rank_dropout),
            "adapted_linear_maps": low_rank_summary.adapted_linear_maps,
            "adapted_module_paths": list(low_rank_summary.adapted_module_paths),
            "trainable_parameters": low_rank_summary.trainable_parameters,
            "frozen_parameters": low_rank_summary.frozen_parameters,
        }
        payload = {
            "contract": contract,
            "epoch": epoch,
            # The ordinary evaluator always receives a standard merged model.
            "model": (
                merged_state_dict(model)
                if adaptation_mode == "low_rank" else model.state_dict()
            ),
            # Resumable wrapped state and a compact audit trail are separate.
            "adaptation_model": (
                model.state_dict() if adaptation_mode == "low_rank" else None
            ),
            "adaptation_state": (
                adaptation_state_dict(model)
                if adaptation_mode == "low_rank" else None
            ),
            "adaptation_mode": adaptation_mode,
            "low_rank": low_rank_metadata,
            "low_rank_zero_delta_max_abs": low_rank_zero_delta_max_abs,
            "adaptation_provenance": (
                str(adaptation_provenance.resolve())
                if adaptation_provenance is not None else None
            ),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "statistics": statistics.to_json(),
            "statistics_checkpoint": (
                str(Path(statistics_checkpoint).resolve())
                if statistics_checkpoint is not None else None
            ),
            "statistics_provenance": statistics_provenance,
            "training_data_schema": training_data_schema,
            "training_data_role": expected_role,
            "training_stage": training_stage,
            "current_policy": current_policy,
            "use_discrete_residual": use_discrete_residual,
            "selection_residual_weight": selection_residual_weight,
            "inference_contract": (
                "pde_agnostic_direct_solution_v1"
                if current_policy == "zero" and not use_discrete_residual else
                "arbitrary_current_correction_v1"
            ),
            "initial_checkpoint": (
                str(initial_checkpoint.resolve())
                if initial_checkpoint is not None else None
            ),
            "model_spec": model.model_spec(),
            "args": vars(args),
            "history": history,
            "validation": validation,
        }
        torch.save(payload, args.output_dir / "last_checkpoint.pt")
        if selection_score < best_score:
            best_score = selection_score
            torch.save(payload, args.output_dir / "checkpoint.pt")
        print(json.dumps(record, sort_keys=True), flush=True)

    if history:
        with (args.output_dir / "history.csv").open(
            "w", newline="", encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    parameter_count = base_parameter_count
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    low_rank_metadata = None if low_rank_summary is None else {
        "rank": low_rank_summary.rank,
        "placement": low_rank_summary.placement,
        "alpha": float(
            args.low_rank if args.low_rank_alpha is None else args.low_rank_alpha
        ),
        "dropout": float(args.low_rank_dropout),
        "adapted_linear_maps": low_rank_summary.adapted_linear_maps,
        "adapted_module_paths": list(low_rank_summary.adapted_module_paths),
        "trainable_parameters": low_rank_summary.trainable_parameters,
        "frozen_parameters": low_rank_summary.frozen_parameters,
    }
    result = {
        "contract": contract,
        "data": str(args.data.resolve()),
        "training_data_schema": training_data_schema,
        "training_data_role": expected_role,
        "training_stage": training_stage,
        "current_policy": current_policy,
        "use_discrete_residual": use_discrete_residual,
        "selection_residual_weight": selection_residual_weight,
        "inference_contract": (
            "pde_agnostic_direct_solution_v1"
            if current_policy == "zero" and not use_discrete_residual else
            "arbitrary_current_correction_v1"
        ),
        "initial_checkpoint": (
            str(initial_checkpoint.resolve())
            if initial_checkpoint is not None else None
        ),
        "model_spec": model.model_spec(),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "adaptation_mode": adaptation_mode,
        "low_rank": low_rank_metadata,
        "low_rank_zero_delta_max_abs": low_rank_zero_delta_max_abs,
        "adaptation_provenance": (
            str(adaptation_provenance.resolve())
            if adaptation_provenance is not None else None
        ),
        "local_parameter_reference": LOCAL_PARAMETER_REFERENCE,
        "capacity_ratio_global_over_local": parameter_count / LOCAL_PARAMETER_REFERENCE,
        "train_cases": len(train_dataset),
        "validation_cases": len(val_dataset),
        "epochs_completed": len(history),
        "samples_per_epoch": args.samples_per_epoch,
        "sample_exposures": len(history) * args.samples_per_epoch,
        "statistics": statistics.to_json(),
        "statistics_checkpoint": (
            str(Path(statistics_checkpoint).resolve())
            if statistics_checkpoint is not None else None
        ),
        "statistics_provenance": statistics_provenance,
        "best_selection_score": best_score,
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "optimizer_state_bytes": sum(
            value.numel() * value.element_size()
            for state in optimizer.state.values()
            for value in state.values()
            if torch.is_tensor(value)
        ),
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8",
    )
    return result


def main() -> None:
    result = train_global(parse_args())
    print(json.dumps({
        "best_selection_score": result["best_selection_score"],
        "parameter_count": result["parameter_count"],
        "trainable_parameter_count": result["trainable_parameter_count"],
        "capacity_ratio_global_over_local": result["capacity_ratio_global_over_local"],
        "peak_gpu_bytes": result["peak_gpu_bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()
