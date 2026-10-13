"""Validation-only target-free latent-Schur/Gauss--Newton reference."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from codiffuse_Schwarz.residual_port import scale_port_code

from .adapter import H5DarcyDataset
from .evaluate import move_batch, relative_l2
from .physics import active_vector, assemble_darcy_system
from .pretrain_compatibility_inverse import coordinate_support
from .train import make_loader
from .train_residual_port_chart import frozen_context, load_frozen_local, sha256
from .train_residual_port_communicator import (
    compatibility_cycle_metrics, interface_losses, load_frozen_chart,
    physics_losses, source_linear_features,
)


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-cases", type=int, default=64)
    parser.add_argument(
        "--start-case", type=int, default=0,
        help="skip this many leading cases (for disjoint calibration/confirmation)",
    )
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument(
        "--ridges", type=float, nargs="+",
        default=(1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1),
    )
    parser.add_argument(
        "--damping", type=float, nargs="+", default=(0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument(
        "--analysis", choices=("value", "value_flux"), default="value",
        help=(
            "observable homogeneous interface analysis. value_flux appends "
            "permeability-weighted normal-flux moments on artificial faces; "
            "it does not evaluate a discrete PDE operator"
        ),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def chart_corrected(chart, context, code: Tensor) -> Tensor:
    return chart(
        context.local, context.physical, context.scalar, context.valid,
        code.reshape(-1, code.shape[-1]),
    ).corrected


def artificial_face_flux_features(
    context, batch: dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    """Return eight homogeneous normal-flux moments and their support.

    The four artificial tile faces contribute constant and tangential-linear
    moments of ``k d_n(P_chi(v)-v)``.  These are interface observations, not a
    finite-volume/PDE residual: they require only the provisional local fields,
    the normalized POU assembly, and the supplied permeability.  For fixed
    geometry/coefficient they are linear in the broken field and vanish for a
    compatible restriction.
    """
    batch_size = context.local.shape[0] // context.cover.tile_count
    consensus = context.cover.tile(
        context.cover.assemble_weighted(context.local, batch_size),
    )
    defect = (consensus - context.local) * context.valid
    kappa = context.cover.tile(
        torch.pow(10.0, batch["logk"].to(defect).clamp(-6.0, 6.0)),
    )
    dx = 2.0 / max(context.cover.height, context.cover.width)
    # FaceDarcyAdapter v2 channels 25:29 are artificial-role masks in N/S/W/E
    # order.  Each role lives only on its corresponding outer patch edge.
    artificial = context.physical[:, 25:29] * context.valid
    tangent = torch.linspace(
        -1.0, 1.0, context.cover.tile_size,
        device=defect.device, dtype=defect.dtype,
    )
    face_values = (
        -kappa[:, 0, 0, :] * (defect[:, 0, 1, :] - defect[:, 0, 0, :]) / dx,
        kappa[:, 0, -1, :] * (defect[:, 0, -1, :] - defect[:, 0, -2, :]) / dx,
        -kappa[:, 0, :, 0] * (defect[:, 0, :, 1] - defect[:, 0, :, 0]) / dx,
        kappa[:, 0, :, -1] * (defect[:, 0, :, -1] - defect[:, 0, :, -2]) / dx,
    )
    face_masks = (
        artificial[:, 0, 0, :], artificial[:, 1, -1, :],
        artificial[:, 2, :, 0], artificial[:, 3, :, -1],
    )
    active = (context.valid > 0.5).to(defect)
    mean_kappa = (kappa * active).flatten(1).sum(1) \
        / active.flatten(1).sum(1).clamp_min(1.0)
    flux_scale = (mean_kappa / dx).clamp_min(1.0e-8)
    moments = []
    supports = []
    for value, weight in zip(face_values, face_masks, strict=True):
        denominator = weight.sum(1).clamp_min(1.0)
        normalized = value / flux_scale[:, None]
        moments.extend((
            (normalized * weight).sum(1) / denominator,
            (normalized * tangent[None] * weight).sum(1) / denominator,
        ))
        supported = (weight.sum(1) > 0.5).to(defect)
        supports.extend((supported, supported))
    feature = torch.stack(moments, 1).reshape(
        batch_size, context.cover.tile_count, 8,
    )
    support = torch.stack(supports, 1).reshape_as(feature)
    return feature, support


def measured_q(
    context, batch: dict[str, Tensor], spec, scale, analysis: str,
) -> tuple[Tensor, Tensor]:
    value = source_linear_features(context, spec, scale)
    support = coordinate_support(context, spec)
    tile_support = (support.sum(-1, keepdim=True) > 0).to(support)
    value_support = torch.cat((support, tile_support.expand(-1, -1, 8)), -1)
    if analysis == "value":
        return value, value_support
    if analysis != "value_flux":
        raise ValueError(f"unknown interface analysis {analysis!r}")
    flux, flux_support = artificial_face_flux_features(context, batch)
    return torch.cat((value, flux), -1), torch.cat((value_support, flux_support), -1)


def q_from_corrected(
    corrected, context, batch, spec, scale, analysis: str,
) -> Tensor:
    return measured_q(
        replace(context, local=corrected), batch, spec, scale, analysis,
    )[0]


def linearized_q(
    chart, context, batch, spec, scale, analysis: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Exact basis Jacobian of the frozen linear chart's measured q map."""
    q0, q_weight = measured_q(context, batch, spec, scale, analysis)
    columns = []
    width = context.cover.tile_count * spec.code_width
    for coordinate in range(width):
        code = context.local.new_zeros(
            1, context.cover.tile_count, spec.code_width,
        )
        code.reshape(-1)[coordinate] = 1.0
        q1 = q_from_corrected(
            chart_corrected(chart, context, code), context, batch, spec, scale,
            analysis,
        )
        columns.append((q1 - q0).reshape(-1))
    return q0.reshape(-1), torch.stack(columns, 1), q_weight.reshape(-1)


def weak_conservation_residual(
    physical_prediction: np.ndarray, system,
) -> float:
    """Net FV flux-minus-source balance, separately normalized as q_phys."""
    residual = system.matrix @ active_vector(physical_prediction, system) - system.rhs
    return float(abs(residual.sum()) / max(np.abs(system.rhs).sum(), 1.0e-12))


def diagnostic_metrics(normalized, corrected, context, batch, solution_std) -> dict:
    physical = normalized * solution_std * batch["mask"]
    energy, residual = physics_losses(normalized, batch, solution_std)
    value, flux = interface_losses(corrected, normalized, context, batch)
    return {
        "relative_l2": float(relative_l2(
            physical, batch["solution"], batch["mask"],
        ).mean()),
        "energy_relative_square": float(energy),
        "residual_relative_square": float(residual),
        "interface_value_square": float(value),
        "interface_flux_square": float(flux),
    }


@torch.no_grad()
def validate(args: argparse.Namespace) -> dict[str, object]:
    if args.maximum_cases <= 0 or args.start_case < 0 \
            or any(value <= 0 for value in args.ridges):
        raise ValueError("invalid case count or ridge")
    if any(not 0.0 <= value <= 1.0 for value in args.damping):
        raise ValueError("damping candidates must lie in [0,1]")
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda",
    )
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, chart_checkpoint = load_frozen_chart(
        args.chart_checkpoint, device,
    )
    if chart_checkpoint["source_local_sha256"] != sha256(args.local_checkpoint):
        raise ValueError("latent-Schur chart/local identities differ")
    dataset = H5DarcyDataset(args.data, "validation")
    loader = make_loader(dataset, 1, False, args.workers, device)
    rows = []; processed = 0; seen = 0
    for host in loader:
        if processed >= args.maximum_cases:
            break
        if seen < args.start_case:
            seen += 1
            continue
        batch = move_batch(host, device)
        if device.type == "cuda": torch.cuda.synchronize()
        started = time.perf_counter()
        context = frozen_context(
            local, adapter, batch, spec, args.core, args.halo,
        )
        if device.type == "cuda": torch.cuda.synchronize()
        local_seconds = time.perf_counter() - started
        q0, jacobian, weight = linearized_q(
            chart, context, batch, spec, scale, args.analysis,
        )
        active = weight > 0.5
        if not active.any():
            raise RuntimeError("latent-Schur case has no supported q coordinates")
        j = jacobian[active].double(); q = q0[active].double()
        identity = torch.eye(j.shape[1], device=device, dtype=torch.float64)
        candidates = [("zero", 0.0, 0.0, torch.zeros(j.shape[1], device=device))]
        for ridge in args.ridges:
            z = torch.linalg.solve(
                j.T @ j + ridge * identity, -(j.T @ q),
            ).to(context.local.dtype)
            for damping in args.damping:
                candidate = damping * z
                corrected = chart_corrected(
                    chart, context,
                    candidate.reshape(1, context.cover.tile_count, spec.code_width),
                )
                measured = q_from_corrected(
                    corrected, context, batch, spec, scale, args.analysis,
                ).reshape(-1)[active]
                ratio_square = float(
                    measured.square().sum() / q0.reshape(-1)[active].square().sum().clamp_min(1.0e-12)
                )
                candidates.append(("gn", ridge, damping, candidate, ratio_square))
        # The explicit zero candidate guarantees target-free non-regression.
        selected = min(
            candidates[1:], key=lambda item: item[4],
            default=("zero", 0.0, 0.0, candidates[0][3], 1.0),
        )
        if selected[4] > 1.0:
            selected = ("zero", 0.0, 0.0, candidates[0][3], 1.0)
        code = selected[3].reshape(
            1, context.cover.tile_count, spec.code_width,
        )
        corrected = chart_corrected(chart, context, code)
        assembled = context.cover.assemble_weighted(corrected, 1) * batch["mask"]
        if device.type == "cuda": torch.cuda.synchronize()
        total_seconds = time.perf_counter() - started
        base_assembled = context.cover.assemble_weighted(context.local, 1) * batch["mask"]
        base_cycle, base_cycle_square = compatibility_cycle_metrics(
            context.local, context, spec, scale,
        )
        cycle, cycle_square = compatibility_cycle_metrics(
            corrected, context, spec, scale,
        )
        mask = batch["mask"][0, 0].cpu().numpy() > 0
        kappa = 10.0 ** batch["logk"][0, 0].cpu().numpy().astype(np.float64)
        source = batch["source"][0, 0].cpu().numpy().astype(np.float64)
        face_mask = batch["dirichlet_face_mask"][0].cpu().numpy() > 0
        face_value = batch["dirichlet_face_value"][0].cpu().numpy().astype(np.float64)
        system = assemble_darcy_system(
            mask, kappa, source, np.zeros_like(source),
            dirichlet_face_mask=face_mask, dirichlet_face_value=face_value,
        )
        for method, normalized, tile_field, elapsed, q_ratio, q_square in (
            ("local", base_assembled, context.local, local_seconds, base_cycle, base_cycle_square),
            ("latent_schur", assembled, corrected, total_seconds, cycle, cycle_square),
        ):
            analysis_square = 1.0 if method == "local" else float(selected[4])
            physical_np = (
                normalized[0, 0].cpu().numpy()
                * adapter.statistics.solution_std * mask
            )
            rows.append({
                "case_index": int(batch["case_index"].reshape(-1)[0]),
                "method": method,
                **diagnostic_metrics(
                    normalized, tile_field, context, batch,
                    adapter.statistics.solution_std,
                ),
                "real_defect_cycle_ratio": float(q_ratio),
                "real_defect_cycle_relative_square": float(q_square),
                "complete_analysis_cycle_ratio": float(analysis_square**0.5),
                "complete_analysis_cycle_relative_square": analysis_square,
                "q_phys_weak_conservation": weak_conservation_residual(
                    physical_np, system,
                ),
                "selected_ridge": selected[1] if method == "latent_schur" else 0.0,
                "selected_damping": selected[2] if method == "latent_schur" else 0.0,
                "wall_seconds": elapsed,
                "local_nfe": context.cover.tile_count,
                "batched_local_calls": 1,
            })
        processed += 1
        seen += 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "cases.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for method in ("local", "latent_schur"):
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            key: float(np.mean([float(row[key]) for row in selected]))
            for key in rows[0]
            if key not in ("case_index", "method")
        }
    gates = {
        name: summary["latent_schur"][name] <= summary["local"][name]
        for name in (
            "complete_analysis_cycle_relative_square",
            "real_defect_cycle_relative_square", "relative_l2",
            "energy_relative_square", "residual_relative_square",
            "interface_value_square", "interface_flux_square",
            "q_phys_weak_conservation",
        )
    }
    result = {
        "contract": "darcy_target_free_latent_schur_validation_v1",
        "selection_split": "validation",
        "start_case": args.start_case,
        "test_subsets_opened": False,
        "target_used_for_selection": False,
        "pde_solve_used_for_selection": False,
        "q": "homogeneous q=d=P_chi(v)-v; q_d empty",
        "interface_analysis": (
            "value moments only" if args.analysis == "value" else
            "value plus k*d_n(P_chi(v)-v) artificial-face moments; no discrete operator"
        ),
        "q_phys": "separate weak FV flux-minus-known-source balance",
        "cases": processed,
        "summary": summary,
        "independent_nonregression_gates": gates,
        "distillation_authorized": all(gates.values()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(validate(parse_args()), indent=2))
