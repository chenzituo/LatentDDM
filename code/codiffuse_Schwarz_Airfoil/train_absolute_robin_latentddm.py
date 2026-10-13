"""Train two-round Airfoil LatentDDM on a frozen absolute-Robin local map."""

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

from .absolute_robin_latentddm import (
    CONTRACT,
    AirfoilRobinLatentDDM,
)
from .absolute_robin_sni import (
    CONTRACT as LOCAL_CONTRACT,
    absolute_robin_port_2d,
    absolute_robin_solver_from_payload,
)
from .airfoil_trace_sni import assemble_subdomains_2d
from .dataset import FieldStatistics
from .train_absolute_robin_bvp import (
    make_example,
    physical_relative_square,
    sha256,
    target_port,
    vorticity_relative_square,
)
from .train_multitrajectory_multiresolution import PairedTrajectoryCorpus


Tensor = torch.Tensor


def context(device: torch.device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else nullcontext()
    )


def spatial_level(configuration: SimpleNamespace) -> str:
    """Return and validate the resolution owned by the frozen endpoint."""
    level = str(getattr(configuration, "spatial_level", "fine"))
    if level not in ("coarse", "fine"):
        raise ValueError(f"unsupported local spatial level: {level}")
    return level


def selected_levels(
    corpus: PairedTrajectoryCorpus,
    configuration: SimpleNamespace,
):
    """Return the level used by the frozen local endpoint.

    Historical native-step checkpoints were trained on L6 and did not store a
    ``spatial_level`` field.  The fixed-cadence adaptation protocol stores
    ``coarse`` explicitly and must therefore remain on L5 throughout composer
    training and evaluation.
    """
    return getattr(corpus, spatial_level(configuration))


def active_split(
    local_payload: dict[str, object], split_manifest: Path | None,
) -> dict[str, list[str]]:
    """Load a source or target split without opening its protected test set."""
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


def load_local(
    checkpoint: Path, device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object], SimpleNamespace]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("contract") != LOCAL_CONTRACT:
        raise ValueError("checkpoint is not an absolute-Robin Airfoil local map")
    saved = dict(payload["args"])
    solver = absolute_robin_solver_from_payload(payload, device)
    solver.eval().requires_grad_(False)
    return solver, payload, SimpleNamespace(**saved)


def attach_configuration(
    example: dict[str, object], configuration: SimpleNamespace,
) -> dict[str, object]:
    example["configuration"] = configuration
    return example


def initial_and_target_code(
    example: dict[str, object],
    configuration: SimpleNamespace,
    modes: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    current_tiles = example["cover"].tile(
        example["previous"], padding_mode="replicate",
    )
    common = {
        "state_mean": example["mean"].flatten(),
        "state_std": example["std"].flatten(),
        "flux_scale": example["flux_scale"],
        "dx": example["dx"],
        "viscosity": example["viscosity"],
        "impedance": (
            float(configuration.mass_impedance),
            float(configuration.momentum_impedance),
        ),
        "face_width": int(configuration.face_width),
        "neighbor_faces": example["neighbor_faces"],
        "valid": example["valid_tiles"],
    }
    initial, roles = absolute_robin_port_2d(current_tiles, **common)
    target, target_roles = target_port(example, configuration)
    torch.testing.assert_close(roles, target_roles)
    correction = (target - initial) * roles.repeat_interleave(3, dim=1)
    code = project_trace_code_2d(
        correction, roles, channels=3, modes=modes,
    )
    return initial, target, roles, code


@torch.inference_mode()
def estimate_code_scale(
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
    sum_square = None
    sum_values = None
    count = 0
    for sample_index in range(samples):
        trajectory = generator.randrange(len(corpus.paired))
        paired = corpus.paired[trajectory]
        index = generator.randrange(len(paired))
        example = make_example(
            paired[index], levels[trajectory], statistics, configuration,
            seed + sample_index * 10_000_019, device,
        )
        _, target, roles, correction_code = initial_and_target_code(
            example, configuration, modes,
        )
        code = (
            project_trace_code_2d(
                target, roles, channels=3, modes=modes,
            )
            if communication_mode == "full_absolute" else correction_code
        )
        values = code.float()
        square = values.square().sum(0)
        sum_square = square if sum_square is None else sum_square + square
        total = values.sum(0)
        sum_values = total if sum_values is None else sum_values + total
        count += values.shape[0]
    if sum_square is None or sum_values is None or count == 0:
        raise RuntimeError("no Robin correction codes were sampled")
    # Absolute boundary decoding benefits from an affine code chart.  Delta
    # modes remain centered at zero so they retain their literal meaning.
    center = torch.zeros_like(sum_square)
    if communication_mode == "full_absolute":
        center = sum_values / count
        # Convert the accumulated second moment to a centered variance.
        scale = torch.sqrt((sum_square / count - center.square()).clamp_min(0.0))
    else:
        scale = torch.sqrt(sum_square / count)
    # A tiny coefficient should not amplify numerical projection noise.
    floor = torch.quantile(scale, 0.25).clamp_min(1.0e-3) * 0.1
    return center.cpu(), scale.clamp_min(floor).cpu()


def code_target(
    example: dict[str, object],
    configuration: SimpleNamespace,
    model: AirfoilRobinLatentDDM,
) -> tuple[Tensor, Tensor, Tensor]:
    _, target, roles, correction_code = initial_and_target_code(
        example, configuration, model.trace_modes,
    )
    code = (
        project_trace_code_2d(
            target, roles, channels=3, modes=model.trace_modes,
        )
        if model.communication_mode == "full_absolute" else correction_code
    )
    batch = example["previous"].shape[0]
    count = int(np.prod(example["cover"].grid_shape))
    standardized = (
        (code - model.code_center.to(code)) / model.code_scale.to(code)
    ).reshape(
        batch, count, model.code_width,
    )
    return standardized, target, roles


@torch.inference_mode()
def validate(
    model: AirfoilRobinLatentDDM,
    corpus: PairedTrajectoryCorpus,
    statistics: FieldStatistics,
    configuration: SimpleNamespace,
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
                args.seed + trajectory * 10_000_019 + int(index), device,
            ), configuration)
            output = model(example)
            target_code, target_boundary, roles = code_target(
                example, configuration, model,
            )
            valid = assemble_subdomains_2d(
                example["cover"], example["valid_tiles"],
                example["previous"].shape[0], "normalized_overlap",
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
            target_code_physical = target_code.reshape(
                output.correction_code.shape,
            ) * model.code_scale.to(output.correction_code) \
                + model.code_center.to(output.correction_code)
            chart_decoded = decode_trace_code_2d(
                target_code_physical, roles, channels=3,
                modes=model.trace_modes,
            )
            chart_port = (
                chart_decoded
                if model.communication_mode == "full_absolute" else
                output.initial_port + chart_decoded
            )
            chart_tiles = model.local_solver(
                example["physical"], example["scalar"], chart_port, roles,
            )
            chart_field = assemble_subdomains_2d(
                example["cover"], chart_tiles,
                example["previous"].shape[0], "normalized_overlap",
            )
            chart_error = physical_relative_square(
                chart_field, example["target"], valid,
                example["mean"], example["std"],
            ).sqrt()
            active = roles.repeat_interleave(3, dim=1)
            port_error = (
                ((output.corrected_port - target_boundary).square() * active).sum()
                / active.sum().clamp_min(1.0)
            ).sqrt()
            rows.append({
                "trajectory": trajectory,
                "index": int(index),
                "local_field": float(local_field),
                "latent_field": float(latent_field),
                "chart_oracle_field": float(chart_error),
                "local_vorticity": float(local_vorticity),
                "latent_vorticity": float(latent_vorticity),
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
            "optional target split; train and validation are opened while the "
            "listed test trajectories remain sealed"
        ),
    )
    parser.add_argument(
        "--initialize", type=Path,
        help=(
            "source LatentDDM checkpoint used to initialize composer-only "
            "adaptation; its frozen local lineage and code chart are retained"
        ),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--samples-per-epoch", type=int, default=256)
    parser.add_argument("--validation-samples-per-trajectory", type=int, default=8)
    parser.add_argument("--code-scale-samples", type=int, default=128)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--trace-modes", type=int, default=4)
    parser.add_argument("--spatial-bins", type=int, default=4)
    parser.add_argument(
        "--communication-mode",
        choices=("centered_delta", "full_delta", "full_absolute"),
        default="centered_delta",
        help="Eq. (3)--(4) composer ablation",
    )
    parser.add_argument("--code-weight", type=float, default=1.0)
    parser.add_argument("--physical-field-weight", type=float, default=50.0)
    parser.add_argument("--vorticity-weight", type=float, default=0.1)
    parser.add_argument("--port-weight", type=float, default=0.1)
    parser.add_argument("--nondegradation-weight", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--seed", type=int, default=20260823)
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
        args.code_weight, args.physical_field_weight,
        args.vorticity_weight, args.port_weight,
        args.nondegradation_weight, args.lr,
    ) < 0.0:
        raise ValueError("invalid Airfoil Robin LatentDDM configuration")
    args.output_dir.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver, local_payload, configuration = load_local(
        args.local_checkpoint, device,
    )
    statistics = FieldStatistics.from_json(local_payload["statistics"])
    split = active_split(local_payload, args.split_manifest)
    training = PairedTrajectoryCorpus(
        args.data_root, split["train"], "all",
    )
    validation = PairedTrajectoryCorpus(
        args.data_root, split["validation"], "all",
    )
    try:
        training_levels = selected_levels(training, configuration)
        initialization = None
        if args.initialize is None:
            code_center, code_scale = estimate_code_scale(
                training, statistics, configuration,
                samples=args.code_scale_samples, modes=args.trace_modes,
                communication_mode=args.communication_mode,
                seed=args.seed + 991, device=device,
            )
        else:
            initialization = torch.load(
                args.initialize, map_location="cpu", weights_only=False,
            )
            if initialization.get("contract") != CONTRACT:
                raise ValueError("initialization is not an Airfoil LatentDDM")
            if initialization.get("local_checkpoint_sha256") != sha256(
                args.local_checkpoint
            ):
                raise ValueError("initialization uses a different local endpoint")
            source_args = dict(initialization["args"])
            for name in (
                "width", "heads", "depth", "trace_modes", "spatial_bins",
                "communication_mode",
            ):
                if source_args.get(name) != getattr(args, name):
                    raise ValueError(
                        f"initialization and adaptation disagree on {name}"
                    )
            code_center = initialization.get("code_center")
            if code_center is None:
                code_center = torch.zeros_like(
                    initialization["composer"]["code_scale"]
                )
            code_scale = initialization["composer"]["code_scale"]
        model = AirfoilRobinLatentDDM(
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
        best = float("inf")
        elapsed_training_seconds = 0.0
        log = args.output_dir / "metrics.jsonl"
        for epoch in range(args.epochs):
            model.train()
            started = time.perf_counter()
            totals = {
                "loss": 0.0, "code": 0.0, "physical_field": 0.0,
                "vorticity": 0.0, "port": 0.0, "nondegradation": 0.0,
            }
            for sample_index in range(args.samples_per_epoch):
                trajectory = generator.randrange(len(training.paired))
                paired = training.paired[trajectory]
                index = generator.randrange(len(paired))
                example = attach_configuration(make_example(
                    paired[index], training_levels[trajectory], statistics,
                    configuration,
                    args.seed + epoch * 100_000_007 + sample_index,
                    device,
                ), configuration)
                optimizer.zero_grad(set_to_none=True)
                with context(device):
                    output = model(example)
                    target_code, target_boundary, roles = code_target(
                        example, configuration, model,
                    )
                    valid = assemble_subdomains_2d(
                        example["cover"], example["valid_tiles"],
                        example["previous"].shape[0], "normalized_overlap",
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
                    code = (
                        output.standardized_correction_code.reshape_as(target_code)
                        - target_code
                    ).square().mean()
                    active = roles.repeat_interleave(3, dim=1)
                    port = (
                        (output.corrected_port - target_boundary).square() * active
                    ).sum() / active.sum().clamp_min(1.0)
                    nondegradation = torch.relu(field - local_field)
                    loss = args.code_weight * code \
                        + args.physical_field_weight * field \
                        + args.vorticity_weight * vort \
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
                totals["port"] += float(port.detach())
                totals["nondegradation"] += float(nondegradation.detach())
            metrics = validate(
                model, validation, statistics, configuration, args, device,
            )
            scheduler.step()
            score = float(metrics["latent_field_mean"]) \
                + 0.1 * float(metrics["latent_vorticity_mean"]) \
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
                "local_contract": LOCAL_CONTRACT,
                "locality_contract": local_payload["locality_contract"],
                "statistics": local_payload["statistics"],
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
