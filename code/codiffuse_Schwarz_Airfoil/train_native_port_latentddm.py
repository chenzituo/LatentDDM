"""Train two-round LatentDDM on one frozen native Airfoil local endpoint."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

from codiffuse_Schwarz.boundary_trace_basis import (
    decode_trace_code_2d,
    project_trace_code_2d,
)

from .airfoil_trace_sni import assemble_subdomains_2d
from .dataset import FieldStatistics
from .native_port_latentddm import (
    CONTRACT,
    AirfoilNativePortLatentDDM,
)
from .train_absolute_robin_bvp import (
    active_robin_mask,
    defect_sni_sweep,
    local_solver_from_payload,
    make_example,
    masked_channel_mse,
    masked_relative_l2,
    physical_relative_square,
    sha256,
    target_port,
    uses_defect_coordinate,
    vorticity_relative_square,
)
from .train_multitrajectory_multiresolution import PairedTrajectoryCorpus


Tensor = torch.Tensor


def amp_context(device: torch.device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else nullcontext()
    )


def load_local(
    path: Path, device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object], SimpleNamespace]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    configuration = SimpleNamespace(**dict(payload["args"]))
    if not uses_defect_coordinate(configuration):
        raise ValueError("native LatentDDM requires a defect-coordinate endpoint")
    if configuration.robin_kind not in (
        "optimized_stokes", "dirichlet", "velocity_dirichlet",
    ):
        raise ValueError("native LatentDDM received a non-native transmission law")
    solver = local_solver_from_payload(payload, device=device)
    return solver.eval().requires_grad_(False), payload, configuration


def load_model(
    path: Path, device: torch.device,
) -> tuple[
    AirfoilNativePortLatentDDM,
    dict[str, object],
    dict[str, object],
    SimpleNamespace,
]:
    """Reconstruct a native communicator and its byte-identified local unit."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("contract") != CONTRACT:
        raise ValueError("checkpoint is not a native-port Airfoil LatentDDM")
    local_path = Path(payload["local_checkpoint"])
    if sha256(local_path) != payload["local_checkpoint_sha256"]:
        raise RuntimeError("the native LatentDDM local checkpoint has changed")
    solver, local_payload, configuration = load_local(local_path, device)
    args = SimpleNamespace(**dict(payload["args"]))
    composer = payload["composer"]
    model = AirfoilNativePortLatentDDM(
        solver, width=int(args.width), heads=int(args.heads),
        depth=int(args.depth), trace_modes=int(args.trace_modes),
        spatial_bins=int(args.spatial_bins),
        code_scale=composer["code_scale"],
        code_center=payload.get("code_center"),
        communication_mode=str(payload["communication_mode"]),
    ).to(device)
    model.load_composer_state_dict(composer)
    return model, payload, local_payload, configuration


def attach_configuration(
    example: dict[str, object], configuration: SimpleNamespace,
) -> dict[str, object]:
    example["configuration"] = configuration
    return example


def spatial_level(configuration: SimpleNamespace) -> str:
    """Return the resolution owned by the frozen native-port endpoint."""
    level = str(getattr(configuration, "spatial_level", "fine"))
    if level not in ("coarse", "fine"):
        raise ValueError(f"unsupported local spatial level: {level}")
    return level


def selected_levels(
    corpus: PairedTrajectoryCorpus, configuration: SimpleNamespace,
):
    return getattr(corpus, spatial_level(configuration))


def active_split(
    local_payload: dict[str, object], split_manifest: Path | None,
) -> dict[str, list[str]]:
    """Load train/validation names while retaining a sealed test group."""
    raw = (
        local_payload["split"]
        if split_manifest is None else
        json.loads(split_manifest.read_text(encoding="utf-8"))
    )
    required = ("train", "validation", "test")
    if not isinstance(raw, dict) or any(name not in raw for name in required):
        raise ValueError("split must contain train, validation, and test")
    split = {name: [str(value) for value in raw[name]] for name in required}
    if not split["train"] or not split["validation"] or not split["test"]:
        raise ValueError("train, validation, and protected test must be nonempty")
    groups = [set(split[name]) for name in required]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("composer split groups overlap")
    return split


def load_composer_initialization(
    path: Path,
    local_checkpoint: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, object], Tensor, Tensor]:
    """Validate and recover a source composer without changing its chart."""
    initialization = torch.load(path, map_location="cpu", weights_only=False)
    if initialization.get("contract") != CONTRACT:
        raise ValueError("initialization is not a native-port Airfoil LatentDDM")
    if initialization.get("local_checkpoint_sha256") != sha256(local_checkpoint):
        raise ValueError("initialization uses a different local endpoint")
    source_args = dict(initialization["args"])
    for name in (
        "width", "heads", "depth", "trace_modes", "spatial_bins",
        "communication_mode",
    ):
        if source_args.get(name) != getattr(args, name):
            raise ValueError(f"initialization and adaptation disagree on {name}")
    composer = initialization.get("composer")
    if not isinstance(composer, dict) or "code_scale" not in composer:
        raise ValueError("initialization has no source code chart")
    code_scale = torch.as_tensor(composer["code_scale"]).detach().cpu()
    code_center = initialization.get("code_center")
    if code_center is None:
        code_center = torch.zeros_like(code_scale)
    else:
        code_center = torch.as_tensor(code_center).detach().cpu()
    if code_center.shape != code_scale.shape:
        raise ValueError("initialization code center and scale disagree")
    return initialization, code_center, code_scale


def target_coordinate(
    example: dict[str, object],
    configuration: SimpleNamespace,
    reference_port: Tensor,
) -> tuple[Tensor, Tensor]:
    absolute, roles = target_port(example, configuration)
    active = active_robin_mask(roles, configuration)
    return (absolute - reference_port) * active, roles


def physical_code(
    port: Tensor, roles: Tensor, *, channels: int, modes: int,
) -> Tensor:
    return project_trace_code_2d(
        port, roles, channels=channels, modes=modes,
    )


@torch.inference_mode()
def estimate_code_chart(
    solver: torch.nn.Module,
    corpus: PairedTrajectoryCorpus,
    statistics: FieldStatistics,
    configuration: SimpleNamespace,
    *,
    samples: int,
    modes: int,
    communication_mode: str,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    generator = random.Random(seed)
    levels = selected_levels(corpus, configuration)
    sum_values = None
    sum_squares = None
    count = 0
    for sample_index in range(samples):
        trajectory = generator.randrange(len(corpus.paired))
        paired = corpus.paired[trajectory]
        index = generator.randrange(len(paired))
        example = make_example(
            paired[index], levels[trajectory], statistics, configuration,
            seed + sample_index * 10_000_019, device,
        )
        initial = defect_sni_sweep(
            solver, example, configuration, example["previous"],
            imposed_port=None, reference_port=None,
            pressure_coarse_correction=False,
        )
        reference = initial.reference_port
        if reference is None:
            raise RuntimeError("native defect endpoint produced no reference port")
        target, roles = target_coordinate(example, configuration, reference)
        desired = (
            target if communication_mode == "full_absolute"
            else target - initial.imposed_port
        )
        values = physical_code(
            desired, roles, channels=target.shape[1] // 4, modes=modes,
        ).float()
        total = values.sum(0)
        square = values.square().sum(0)
        sum_values = total if sum_values is None else sum_values + total
        sum_squares = square if sum_squares is None else sum_squares + square
        count += int(values.shape[0])
    if sum_values is None or sum_squares is None or count == 0:
        raise RuntimeError("no native face codes were sampled")
    center = torch.zeros_like(sum_values)
    if communication_mode == "full_absolute":
        center = sum_values / count
        variance = (sum_squares / count - center.square()).clamp_min(0.0)
    else:
        variance = sum_squares / count
    scale = variance.sqrt()
    floor = torch.quantile(scale, 0.25).clamp_min(1.0e-3) * 0.1
    return center.cpu(), scale.clamp_min(floor).cpu()


def code_supervision(
    model: AirfoilNativePortLatentDDM,
    output,
    example: dict[str, object],
    configuration: SimpleNamespace,
) -> tuple[Tensor, Tensor, Tensor]:
    if output.reference_port is None:
        raise RuntimeError("native LatentDDM output omitted its reference port")
    target, roles = target_coordinate(
        example, configuration, output.reference_port,
    )
    desired = (
        target if model.communication_mode == "full_absolute"
        else target - output.initial_port
    )
    code = physical_code(
        desired, roles, channels=model.channels_per_face,
        modes=model.trace_modes,
    )
    batch = int(example["previous"].shape[0])
    count = int(np.prod(example["cover"].grid_shape))
    standardized = (
        (code - model.code_center.to(code)) / model.code_scale.to(code)
    ).reshape(batch, count, model.code_width)
    return standardized, target, roles


@torch.inference_mode()
def validate(
    model: AirfoilNativePortLatentDDM,
    corpus: PairedTrajectoryCorpus,
    statistics: FieldStatistics,
    configuration: SimpleNamespace,
    projection_statistics: dict[str, object] | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    rows = []
    levels = selected_levels(corpus, configuration)
    for trajectory, paired in enumerate(corpus.paired):
        indices = np.linspace(
            0, len(paired) - 1,
            min(args.validation_samples_per_trajectory, len(paired)),
            dtype=int,
        )
        for index in indices:
            example = attach_configuration(make_example(
                paired[int(index)], levels[trajectory], statistics,
                configuration,
                args.validation_cover_seed + trajectory * 10_000_019 + int(index),
                device,
                projection_statistics=projection_statistics,
            ), configuration)
            output = model(example)
            target_code, target_boundary, roles = code_supervision(
                model, output, example, configuration,
            )
            valid = assemble_subdomains_2d(
                example["cover"], example["valid_tiles"],
                int(example["previous"].shape[0]), "normalized_overlap",
            )
            local_field = physical_relative_square(
                output.local_field, example["target"], valid,
                example["mean"], example["std"],
            ).sqrt()
            latent_field = physical_relative_square(
                output.field, example["target"], valid,
                example["mean"], example["std"],
            ).sqrt()
            local_vorticity = vorticity_relative_square(
                output.local_field, example["target"], valid,
                example["mean"], example["std"], example["dx"],
            ).sqrt()
            latent_vorticity = vorticity_relative_square(
                output.field, example["target"], valid,
                example["mean"], example["std"], example["dx"],
            ).sqrt()
            local_projection = None
            latent_projection = None
            if example["target_projection"] is not None:
                if output.auxiliary_field is None \
                        or output.local_auxiliary_field is None:
                    raise RuntimeError(
                        "projection-closed LatentDDM omitted its auxiliary state"
                    )
                local_projection = masked_relative_l2(
                    output.local_auxiliary_field,
                    example["target_projection"], valid,
                )
                latent_projection = masked_relative_l2(
                    output.auxiliary_field,
                    example["target_projection"], valid,
                )
            target_code_physical = target_code.reshape(
                output.correction_code.shape,
            ) * model.code_scale.to(output.correction_code) \
                + model.code_center.to(output.correction_code)
            decoded = decode_trace_code_2d(
                target_code_physical, roles,
                channels=model.channels_per_face, modes=model.trace_modes,
            )
            chart_port = (
                decoded if model.communication_mode == "full_absolute"
                else output.initial_port + decoded
            ) * active_robin_mask(roles, configuration)
            chart = defect_sni_sweep(
                model.local_solver, example, configuration,
                output.local_field, imposed_port=chart_port,
                reference_port=output.reference_port,
                pressure_coarse_correction=bool(getattr(
                    configuration, "pressure_coarse_correction", False,
                )),
            )
            chart_field = physical_relative_square(
                chart.field, example["target"], valid,
                example["mean"], example["std"],
            ).sqrt()
            active = active_robin_mask(roles, configuration)
            port_error = (
                ((output.corrected_port - target_boundary).square() * active).sum()
                / active.sum().clamp_min(1.0)
            ).sqrt()
            rows.append({
                "trajectory": trajectory,
                "index": int(index),
                "local_field": float(local_field),
                "latent_field": float(latent_field),
                "chart_oracle_field": float(chart_field),
                "local_vorticity": float(local_vorticity),
                "latent_vorticity": float(latent_vorticity),
                "local_projection": (
                    float(local_projection) if local_projection is not None else None
                ),
                "latent_projection": (
                    float(latent_projection) if latent_projection is not None else None
                ),
                "code_rmse": float((
                    output.standardized_correction_code.reshape_as(target_code)
                    - target_code
                ).square().mean().sqrt()),
                "port_rmse": float(port_error),
            })
    result: dict[str, object] = {"examples": len(rows), "rows": rows}
    for name in (
        "local_field", "latent_field", "chart_oracle_field",
        "local_vorticity", "latent_vorticity", "code_rmse", "port_rmse",
    ):
        result[f"{name}_mean"] = float(np.mean([row[name] for row in rows]))
    if rows[0]["latent_projection"] is not None:
        for name in ("local_projection", "latent_projection"):
            result[f"{name}_mean"] = float(np.mean([
                row[name] for row in rows
            ]))
    result["latent_improvement_fraction"] = float(np.mean([
        row["latent_field"] < row["local_field"] for row in rows
    ]))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split-manifest", type=Path,
        help=(
            "optional target split; only train and validation are opened, "
            "while the listed test trajectories remain sealed"
        ),
    )
    parser.add_argument(
        "--initialize", type=Path,
        help=(
            "source native-port LatentDDM checkpoint for composer-only target "
            "adaptation; its frozen local lineage and code chart are retained"
        ),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--samples-per-epoch", type=int, default=256)
    parser.add_argument("--validation-samples-per-trajectory", type=int, default=4)
    parser.add_argument("--validation-cover-seed", type=int, default=20260813)
    parser.add_argument("--code-scale-samples", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--trace-modes", type=int, default=8)
    parser.add_argument("--spatial-bins", type=int, default=4)
    parser.add_argument(
        "--communication-mode",
        choices=("centered_delta", "full_delta", "full_absolute"),
        default="centered_delta",
    )
    parser.add_argument("--code-weight", type=float, default=1.0)
    parser.add_argument("--physical-field-weight", type=float, default=50.0)
    parser.add_argument("--vorticity-weight", type=float, default=0.1)
    parser.add_argument("--projection-weight", type=float, default=1.0)
    parser.add_argument("--selection-projection-weight", type=float, default=0.05)
    parser.add_argument("--port-weight", type=float, default=0.1)
    parser.add_argument("--nondegradation-weight", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if min(
        args.epochs, args.samples_per_epoch,
        args.validation_samples_per_trajectory, args.code_scale_samples,
        args.width, args.heads, args.depth, args.trace_modes,
        args.spatial_bins,
    ) <= 0 or min(
        args.code_weight, args.physical_field_weight, args.vorticity_weight,
        args.projection_weight, args.selection_projection_weight,
        args.port_weight, args.nondegradation_weight, args.lr,
    ) < 0.0:
        raise ValueError("invalid native Airfoil LatentDDM configuration")
    args.output_dir.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver, local_payload, configuration = load_local(
        args.local_checkpoint, device,
    )
    statistics = FieldStatistics.from_json(local_payload["statistics"])
    projection_statistics = local_payload.get("projection_statistics")
    if int(local_payload.get("projection_channels", 0)) == 2 \
            and not isinstance(projection_statistics, dict):
        raise ValueError("projection-closed local checkpoint has no statistics")
    history = int(configuration.history_steps)
    split = active_split(local_payload, args.split_manifest)
    training = PairedTrajectoryCorpus(
        args.data_root, split["train"], "all", history,
    )
    validation = PairedTrajectoryCorpus(
        args.data_root, split["validation"], "all", history,
    )
    try:
        initialization = None
        if args.initialize is None:
            code_center, code_scale = estimate_code_chart(
                solver, training, statistics, configuration,
                samples=args.code_scale_samples, modes=args.trace_modes,
                communication_mode=args.communication_mode,
                seed=args.seed + 991, device=device,
            )
        else:
            initialization, code_center, code_scale = \
                load_composer_initialization(
                    args.initialize, args.local_checkpoint, args,
                )
        model = AirfoilNativePortLatentDDM(
            solver, width=args.width, heads=args.heads, depth=args.depth,
            trace_modes=args.trace_modes, spatial_bins=args.spatial_bins,
            code_scale=code_scale, code_center=code_center,
            communication_mode=args.communication_mode,
        ).to(device)
        if initialization is not None:
            model.load_composer_state_dict(initialization["composer"])
        optimizer = torch.optim.AdamW(
            model.trainable_parameters, lr=args.lr, weight_decay=1.0e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=args.lr * 0.05,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        generator = random.Random(args.seed)
        training_levels = selected_levels(training, configuration)
        best = float("inf")
        elapsed_training_seconds = 0.0
        log = args.output_dir / "metrics.jsonl"
        for epoch in range(args.epochs):
            model.train()
            started = time.perf_counter()
            totals = {
                "loss": 0.0, "code": 0.0, "physical_field": 0.0,
                "vorticity": 0.0, "projection": 0.0,
                "port": 0.0, "nondegradation": 0.0,
            }
            for sample_index in range(args.samples_per_epoch):
                trajectory = generator.randrange(len(training.paired))
                paired = training.paired[trajectory]
                index = generator.randrange(len(paired))
                example = attach_configuration(make_example(
                    paired[index], training_levels[trajectory], statistics,
                    configuration,
                    args.seed + epoch * 100_000_007 + sample_index, device,
                    projection_statistics=projection_statistics,
                ), configuration)
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device):
                    output = model(example)
                    target_code, target_boundary, roles = code_supervision(
                        model, output, example, configuration,
                    )
                    valid = assemble_subdomains_2d(
                        example["cover"], example["valid_tiles"],
                        int(example["previous"].shape[0]), "normalized_overlap",
                    )
                    field = physical_relative_square(
                        output.field, example["target"], valid,
                        example["mean"], example["std"],
                    )
                    local_field = physical_relative_square(
                        output.local_field, example["target"], valid,
                        example["mean"], example["std"],
                    ).detach()
                    vort = vorticity_relative_square(
                        output.field, example["target"], valid,
                        example["mean"], example["std"], example["dx"],
                    )
                    if example["target_projection"] is not None:
                        if output.auxiliary_field is None \
                                or output.local_auxiliary_field is None:
                            raise RuntimeError(
                                "projection-closed LatentDDM omitted auxiliary state"
                            )
                        projection = masked_channel_mse(
                            output.auxiliary_field,
                            example["target_projection"], valid,
                        )
                        local_projection = masked_channel_mse(
                            output.local_auxiliary_field,
                            example["target_projection"], valid,
                        ).detach()
                    else:
                        projection = field.new_zeros(())
                        local_projection = field.new_zeros(())
                    code = (
                        output.standardized_correction_code.reshape_as(target_code)
                        - target_code
                    ).square().mean()
                    active = active_robin_mask(roles, configuration)
                    port = (
                        (output.corrected_port - target_boundary).square() * active
                    ).sum() / active.sum().clamp_min(1.0)
                    nondegradation = torch.relu(field - local_field) \
                        + torch.relu(projection - local_projection)
                    loss = args.code_weight * code \
                        + args.physical_field_weight * field \
                        + args.vorticity_weight * vort \
                        + args.projection_weight * projection \
                        + args.port_weight * port \
                        + args.nondegradation_weight * nondegradation
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters, 1.0)
                scaler.step(optimizer)
                scaler.update()
                totals["loss"] += float(loss.detach())
                totals["code"] += float(code.detach())
                totals["physical_field"] += float(field.detach())
                totals["vorticity"] += float(vort.detach())
                totals["projection"] += float(projection.detach())
                totals["port"] += float(port.detach())
                totals["nondegradation"] += float(nondegradation.detach())
            metrics = validate(
                model, validation, statistics, configuration,
                projection_statistics, args, device,
            )
            scheduler.step()
            score = float(metrics["latent_field_mean"]) \
                + 0.1 * float(metrics["latent_vorticity_mean"]) \
                + args.selection_projection_weight * float(
                    metrics.get("latent_projection_mean", 0.0)
                ) \
                + 5.0 * max(
                    float(metrics["latent_field_mean"])
                    - float(metrics["local_field_mean"]), 0.0,
                )
            epoch_seconds = time.perf_counter() - started
            elapsed_training_seconds += epoch_seconds
            record = {
                "epoch": epoch,
                "seconds": epoch_seconds,
                "cumulative_training_seconds": elapsed_training_seconds,
                "lr": optimizer.param_groups[0]["lr"],
                **{
                    f"train_{name}": value / args.samples_per_epoch
                    for name, value in totals.items()
                },
                **{
                    f"validation_{name}": value
                    for name, value in metrics.items() if name != "rows"
                },
                "selection_score": score,
            }
            with log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            payload = {
                "contract": CONTRACT,
                "composer": model.composer_state_dict(),
                "code_center": model.code_center.detach().cpu(),
                "communication_mode": args.communication_mode,
                "epoch": epoch,
                "selection_score": score,
                "args": vars(args),
                "local_checkpoint": str(args.local_checkpoint.resolve()),
                "local_checkpoint_sha256": sha256(args.local_checkpoint),
                "local_contract": local_payload["contract"],
                "locality_contract": local_payload["locality_contract"],
                "statistics": local_payload["statistics"],
                "projection_statistics": projection_statistics,
                "projection_channels": int(local_payload.get(
                    "projection_channels", 0,
                )),
                "native_markov_state_closed": bool(local_payload.get(
                    "native_markov_state_closed", False,
                )),
                "split": split,
                "source_local_split": local_payload["split"],
                "split_manifest": (
                    str(args.split_manifest.resolve())
                    if args.split_manifest is not None else None
                ),
                "split_manifest_sha256": (
                    sha256(args.split_manifest)
                    if args.split_manifest is not None else None
                ),
                "initialization": (
                    str(args.initialize.resolve())
                    if args.initialize is not None else None
                ),
                "initialization_sha256": (
                    sha256(args.initialize)
                    if args.initialize is not None else None
                ),
                "composer_only_adaptation": args.initialize is not None,
                "training_seconds": elapsed_training_seconds,
                "adaptation_seconds": (
                    elapsed_training_seconds if args.initialize is not None else 0.0
                ),
                "source_training_seconds": (
                    float(initialization.get("training_seconds", 0.0))
                    if initialization is not None else elapsed_training_seconds
                ),
                "validation": metrics,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in model.trainable_parameters
                ),
                "local_parameters": sum(
                    parameter.numel() for parameter in solver.parameters()
                ),
                "local_operator_calls_per_step": 2,
                "communication_layers": args.depth,
                "port_channels": model.port_channels,
                "channels_per_face": model.channels_per_face,
                "target_field_used_by_inference": False,
                "absolute_coordinate_prompt": False,
                "protected_test_trajectories_opened": 0,
            }
            torch.save(payload, args.output_dir / "checkpoint_last.pt")
            if score < best:
                best = score
                torch.save(payload, args.output_dir / "checkpoint_best.pt")
            print(json.dumps(record), flush=True)
    finally:
        training.close()
        validation.close()
    summary = {
        "contract": "airfoil_latentddm_training_summary_v1",
        "composer_contract": CONTRACT,
        "composer_only_adaptation": args.initialize is not None,
        "epochs_completed": args.epochs,
        "samples_per_epoch": args.samples_per_epoch,
        "sample_exposures": args.epochs * args.samples_per_epoch,
        "training_seconds_total": elapsed_training_seconds,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.trainable_parameters
        ),
        "frozen_local_parameter_count": sum(
            parameter.numel() for parameter in solver.parameters()
        ),
        "best_validation_selection_score": best,
        "protected_test_trajectories_opened": 0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
