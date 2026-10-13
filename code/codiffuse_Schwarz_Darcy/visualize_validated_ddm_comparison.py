"""Plot only contract-valid Darcy one-pass and iterative DDM predictions.

The one-pass residual-port composer and literal Huang SNI share one frozen
face-resolved Dirichlet local solver.  The Mao-style Robin OSM reimplementation
uses its separately trained mixed-Robin local solver.  Validation artifacts are
mandatory gates: this script refuses to produce a figure from an invalid or
non-convergent iterative recurrence.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import H5DarcyDataset, MODEL_SPEC
from .evaluate import energy_relative_error, relative_l2, residual_rms
from .physics import exposed_dirichlet_faces
from .residual_port_composer import (
    ResidualPortCommunicator, SourceLinearResidualPortCommunicator,
)
from .robin_osm import load_neural_robin_local, neural_robin_osm_sweep
from .train_residual_port_chart import (
    FrozenContext, load_frozen_local, local_model_call, sha256,
)
from .train_residual_port_communicator import (
    CANONICAL_ANALYZER_CONTRACT, CONTRACT as COMPOSER_CONTRACT,
    communicator_features, forward_composer, load_frozen_chart,
    source_linear_signal_width,
)
from .validate_face_sni import face_sni_step


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--composer-checkpoint", type=Path, required=True)
    parser.add_argument("--composer-validation", type=Path, required=True)
    parser.add_argument("--sni-validation", type=Path, required=True)
    parser.add_argument("--robin-checkpoint", type=Path, required=True)
    parser.add_argument("--robin-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("A", "B", "C", "D"), default="A")
    parser.add_argument("--case-index", type=int, default=55)
    parser.add_argument("--offset", default="8,8")
    parser.add_argument("--sni-sweeps", type=int, default=512)
    parser.add_argument("--robin-sweeps", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_batch(item: dict[str, object], device: torch.device) -> dict[str, Tensor]:
    return {
        key: value[None].to(device)
        for key, value in item.items() if isinstance(value, Tensor)
    }


def _metrics(field: Tensor, batch: dict[str, Tensor]) -> dict[str, float]:
    return {
        "relative_l2": float(relative_l2(
            field, batch["solution"], batch["mask"],
        )[0]),
        "energy_relative_error": float(energy_relative_error(
            field, batch["solution"], batch["logk"], batch["mask"],
        )[0]),
        "residual_rms": float(residual_rms(field, batch)[0]),
    }


def _load_communicator(
    checkpoint_path: Path, local_checkpoint: Path, device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, Any, Tensor, Any, dict]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False,
    )
    if checkpoint.get("contract") != COMPOSER_CONTRACT:
        raise ValueError("one-pass checkpoint has the wrong contract")
    if checkpoint.get("source_local_sha256") != sha256(local_checkpoint):
        raise ValueError("one-pass and SNI local checkpoint identities differ")
    local, adapter, _ = load_frozen_local(local_checkpoint, device)
    chart_path = Path(checkpoint["source_chart_checkpoint"])
    if checkpoint.get("source_chart_sha256") != sha256(chart_path):
        raise ValueError("one-pass response-chart identity differs")
    chart, spec, scale, _ = load_frozen_chart(chart_path, device)
    communicator_spec = checkpoint["communicator_spec"]
    if communicator_spec.get("analyzer_mode") == "source_linear":
        if communicator_spec.get("canonical_analyzer_contract") != \
                CANONICAL_ANALYZER_CONTRACT:
            raise ValueError("one-pass analyzer is not the homogeneous kernel")
        signal_width = source_linear_signal_width(spec)
        communicator = SourceLinearResidualPortCommunicator(
            int(communicator_spec["physical_channels"]),
            int(communicator_spec["scalar_dim"]), spec.code_width,
            signal_width, token_width=int(communicator_spec["token_width"]),
            heads=int(communicator_spec["heads"]),
        ).to(device)
    else:
        communicator = ResidualPortCommunicator(
            1, int(communicator_spec["physical_channels"]),
            int(communicator_spec["scalar_dim"]), spec.code_width,
            token_width=int(communicator_spec["token_width"]),
            heads=int(communicator_spec["heads"]),
            aligned_width=int(communicator_spec.get("aligned_width", 0)),
        ).to(device)
        communicator.analyzer_mode = communicator_spec.get(
            "analyzer_mode", "generic",
        )
    communicator.load_state_dict(checkpoint["communicator"], strict=True)
    communicator.eval(); chart.eval(); local.eval()
    return local, chart, communicator, adapter, scale, spec, checkpoint


@torch.no_grad()
def _target_free_context(
    local: torch.nn.Module, adapter: Any, batch: dict[str, Tensor], spec: Any,
    cover: CartesianCover,
) -> FrozenContext:
    """Build the inference context without reading the reference solution."""
    from .adapter import tiled_sni_conditions

    zero = torch.zeros_like(batch["solution"])
    physical, scalar = tiled_sni_conditions(adapter, batch, zero, cover)
    local_tiles = local_model_call(local, physical, scalar, cover)
    valid = cover.tile(batch["mask"].to(local_tiles)) * cover.valid_tiles(
        batch["solution"].shape[0], local_tiles.device, local_tiles.dtype,
    )
    local_tiles = local_tiles * valid
    placeholder = torch.zeros_like(batch["solution"])
    placeholder_tiles = cover.tile(placeholder) * valid
    placeholder_code = placeholder.new_zeros(
        batch["solution"].shape[0], cover.tile_count, spec.code_width,
    )
    return FrozenContext(
        local_tiles, physical, scalar, valid, placeholder,
        placeholder_tiles, placeholder_code, cover,
    )


def _gate_validation(
    composer_path: Path, sni_path: Path, robin_path: Path,
    local_checkpoint: Path, composer_checkpoint: Path,
    robin_checkpoint: Path, robin_sweeps: int,
) -> dict[str, Any]:
    composer = _read_json(composer_path)
    if composer.get("contract") != "darcy_residual_port_validation_ladder_v1":
        raise ValueError("missing residual-port causal validation gate")
    summary = composer["summary"]
    local_l2 = summary["local_zero_code/radius_none"]["relative_l2"]
    predicted_l2 = summary["predicted/radius_global"]["relative_l2"]
    shuffled_l2 = summary["shuffled_code/radius_global_shuffled"]["relative_l2"]
    cycle = summary["predicted/radius_global"]["real_defect_cycle_ratio"]
    if not (predicted_l2 < local_l2 and predicted_l2 < shuffled_l2 and cycle < 1.0):
        raise RuntimeError("one-pass validation field/causality/cycle gate failed")
    if composer.get("source_local_sha256") != _sha256(local_checkpoint):
        raise ValueError("composer validation used a different local checkpoint")
    if Path(composer["checkpoint"]).resolve() != composer_checkpoint.resolve():
        raise ValueError("composer validation used a different composer checkpoint")

    sni = _read_json(sni_path)
    if sni.get("contract") != "darcy_huang_raw_additive_extended_validation_v1" \
            or not sni.get("exact_huang_equation_2", False):
        raise ValueError("missing literal Huang Eq.(2) validation gate")
    if sni.get("checkpoint_sha256") != _sha256(local_checkpoint):
        raise ValueError("SNI validation used a different local checkpoint")
    sni_summary = sni["summary"]
    if not (
        sni_summary["tau_0.1000/checkpoint_0512"]["relative_l2"]
        < sni_summary["tau_0.1000/checkpoint_0001"]["relative_l2"]
    ):
        raise RuntimeError("literal Huang SNI validation did not converge")

    robin = _read_json(robin_path)
    if robin.get("contract") != "darcy_neural_robin_osm_matched_frontier_v1":
        raise ValueError("missing neural Robin OSM validation gate")
    if Path(robin["checkpoint"]).resolve() != robin_checkpoint.resolve():
        raise ValueError("Robin validation used a different local checkpoint")
    frontier = robin["frontiers"]["neural_robin_osm"]
    first = frontier["checkpoint_0001"]
    selected = frontier[f"checkpoint_{robin_sweeps:04d}"]
    if selected["failed_cases"] != 0 or not np.isfinite(selected["relative_l2"]):
        raise RuntimeError("neural Robin recurrence failed on validation")
    if not selected["relative_l2"] < first["relative_l2"]:
        raise RuntimeError("neural Robin recurrence does not improve with sweeps")
    return {"composer": composer, "sni": sni, "robin": robin}


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.sni_sweeps <= 0 or args.robin_sweeps <= 0:
        raise ValueError("iterative sweep counts must be positive")
    gates = _gate_validation(
        args.composer_validation, args.sni_validation, args.robin_validation,
        args.local_checkpoint, args.composer_checkpoint,
        args.robin_checkpoint, args.robin_sweeps,
    )
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    local, chart, communicator, adapter, scale, spec, composer_checkpoint = (
        _load_communicator(args.composer_checkpoint, args.local_checkpoint, device)
    )
    dataset = H5DarcyDataset(args.data, args.split)
    if not 0 <= args.case_index < len(dataset):
        raise IndexError("case index lies outside the requested split")
    batch = _to_batch(dataset[args.case_index], device)
    dataset.close()
    oy, ox = (int(piece) for piece in args.offset.split(","))
    target = batch["solution"]
    cover = CartesianCover(
        target.shape[-2], target.shape[-1], MODEL_SPEC["core"],
        MODEL_SPEC["halo"], oy, ox,
    )

    context = _target_free_context(local, adapter, batch, spec, cover)
    aligned = communicator_features(
        communicator, context, batch, spec, scale, adapter,
    )
    one_pass_normalized, _, _, detail = forward_composer(
        communicator, chart, context, scale, batch,
        max_graph_radius=None, aligned_features=aligned,
    )
    local_field = cover.assemble_weighted(context.local, 1) \
        * adapter.statistics.solution_std * batch["mask"]
    one_pass_field = one_pass_normalized \
        * adapter.statistics.solution_std * batch["mask"]

    # Huang's raw sum requires tau < 1/K.  Validation selected tau=.1=.9/9;
    # apply the predeclared scale-free rule .9/K on a different tile count.
    sni_tau = 0.9 / cover.tile_count
    sni_field = torch.zeros_like(target)
    for _ in range(args.sni_sweeps):
        sni_field = face_sni_step(
            local, adapter, batch, sni_field, cover,
            "huang_raw_additive", sni_tau,
        )

    robin_model, robin_adapter, robin_checkpoint = load_neural_robin_local(
        args.robin_checkpoint, device,
    )
    mask_np = batch["mask"][0, 0].cpu().numpy() > 0.5
    kappa_np = 10.0 ** batch["logk"][0, 0].cpu().numpy().astype(np.float64)
    source_np = batch["source"][0, 0].cpu().numpy().astype(np.float64)
    sdf_np = batch["sdf"][0, 0].cpu().numpy().astype(np.float64)
    normals_np = batch["normals"][0].cpu().numpy().astype(np.float64)
    face_mask_np = exposed_dirichlet_faces(mask_np)
    bc_np = batch["bc_value"][0, 0].cpu().numpy().astype(np.float64)
    face_value_np = np.broadcast_to(bc_np[None], face_mask_np.shape).copy()
    robin_np = np.zeros_like(source_np)
    for _ in range(args.robin_sweeps):
        robin_np = neural_robin_osm_sweep(
            robin_model, robin_adapter, mask_np, kappa_np, source_np,
            sdf_np, normals_np, face_mask_np, face_value_np, robin_np, cover,
            impedance_ratio=float(gates["robin"]["impedance_ratio"]),
            relaxation=float(gates["robin"]["relaxation"]), device=device,
        ).field
    robin_field = torch.from_numpy(robin_np)[None, None].to(
        device=device, dtype=target.dtype,
    )

    fields = {
        "reference": target,
        "local": local_field,
        "one_pass": one_pass_field,
        "huang_sni": sni_field,
        "robin_osm": robin_field,
    }
    metrics = {
        name: _metrics(field, batch) for name, field in fields.items()
        if name != "reference"
    }
    result: dict[str, Any] = {
        "contract": "darcy_contract_valid_solution_ddm_comparison_v1",
        "split": args.split, "case_index": args.case_index,
        "case_selection": "predeclared legacy figure index; not reselected",
        "offset": [oy, ox], "tile_count": cover.tile_count,
        "shared_dirichlet_local_checkpoint": str(args.local_checkpoint.resolve()),
        "shared_dirichlet_local_sha256": _sha256(args.local_checkpoint),
        "composer_checkpoint": str(args.composer_checkpoint.resolve()),
        "robin_checkpoint": str(args.robin_checkpoint.resolve()),
        "robin_checkpoint_epoch": robin_checkpoint["epoch"],
        "target_used_by_inference": False,
        "sni": {
            "assembly": "literal Huang raw zero-extension additive Eq.(2)",
            "tau_rule": "0.9 / tile_count", "tau": sni_tau,
            "sweeps": args.sni_sweeps,
            "local_nfe": args.sni_sweeps * cover.tile_count,
            "batched_local_calls": args.sni_sweeps,
        },
        "robin": {
            "label": "Mao-style Robin OSM (reimplementation)",
            "sweeps": args.robin_sweeps,
            "local_nfe": args.robin_sweeps * cover.tile_count,
            "batched_local_calls": args.robin_sweeps,
        },
        "one_pass": {
            "local_nfe": cover.tile_count,
            "batched_local_calls": 1,
            "communication_rounds": 1,
        },
        "metrics": metrics,
        "validation_gates": {
            "composer": str(args.composer_validation.resolve()),
            "sni": str(args.sni_validation.resolve()),
            "robin": str(args.robin_validation.resolve()),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "darcy_validated_ddm_comparison"
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )

    method_order = ["reference", "local", "one_pass", "huang_sni", "robin_osm"]
    titles = [
        "Reference", "Local, zero trace\n1 call",
        "One-pass port\n1 call + 1 message",
        f"Huang SNI\n{args.sni_sweeps} calls",
        f"Robin OSM\n{args.robin_sweeps} calls",
    ]
    mask = mask_np
    target_np = target[0, 0].cpu().numpy()
    vmin, vmax = np.quantile(target_np[mask], (0.01, 0.99))
    error_max = max(float(np.quantile(
        np.abs(fields[name][0, 0].cpu().numpy() - target_np)[mask], 0.98,
    )) for name in method_order[1:])
    figure, axes = plt.subplots(
        2, len(method_order), figsize=(13.6, 4.25), constrained_layout=True,
    )
    field_image = None
    for column, (name, title) in enumerate(zip(method_order, titles)):
        array = fields[name][0, 0].detach().cpu().numpy()
        field_image = axes[0, column].imshow(
            np.where(mask, array, np.nan), origin="lower", cmap="viridis",
            vmin=vmin, vmax=vmax,
        )
        suffix = "" if name == "reference" else \
            f"\n$L^2$={metrics[name]['relative_l2']:.3f}"
        axes[0, column].set_title(title + suffix, fontsize=8.5)
        if name == "reference":
            axes[1, column].imshow(
                np.where(mask, batch["logk"][0, 0].cpu().numpy(), np.nan),
                origin="lower", cmap="magma",
            )
            axes[1, column].set_title(r"Input $\log_{10}\kappa$", fontsize=9)
        else:
            axes[1, column].imshow(
                np.where(mask, np.abs(array - target_np), np.nan),
                origin="lower", cmap="inferno", vmin=0.0, vmax=error_max,
            )
            axes[1, column].set_title("Absolute error", fontsize=9)
        for row in range(2):
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
    assert field_image is not None
    figure.colorbar(
        field_image, ax=axes[0].tolist(), shrink=0.78, pad=0.012,
        label="solution $u$",
    )
    figure.colorbar(
        axes[1, 1].images[0], ax=axes[1, 1:].tolist(), shrink=0.78,
        pad=0.012, label=r"$|\hat u-u|$",
    )
    for suffix in ("pdf", "png"):
        figure.savefig(args.output_dir / f"{stem}.{suffix}", dpi=240)
    plt.close(figure)
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2))
