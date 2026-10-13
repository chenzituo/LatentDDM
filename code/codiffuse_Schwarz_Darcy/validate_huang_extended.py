"""Extended full-validation frontier for literal Huang raw-additive SNI."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import DarcyStatistics, FaceDarcyAdapter, H5DarcyDataset, MODEL_SPEC, build_model
from .evaluate import energy_relative_error, initial_state, move_batch, relative_l2, residual_rms
from .train_face_local import CONTRACT
from .validate_face_sni import _offset, _sha256, face_sni_step


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sweeps", type=int, default=512)
    parser.add_argument(
        "--checkpoints", type=int, nargs="+",
        default=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-validation-cases", type=int, default=250)
    parser.add_argument("--convergence-patience", type=int, default=3)
    parser.add_argument("--relative-update-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--strict-residual-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--offset", default="0,0")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def selected_and_adjacent(pilot: dict) -> tuple[list[float], float]:
    if pilot.get("contract") != "darcy_huang_raw_additive_sni_validation_v1" \
            or not pilot.get("exact_huang_equation_2"):
        raise ValueError("extended Huang frontier requires the exact raw-additive pilot")
    selected = float(pilot["selected_control"]["tau"])
    convergence = float(pilot["selected_control"]["residual_tolerance"])
    by_tau: dict[float, float] = {}
    for row in pilot["candidate_controls"]:
        tau, score = float(row["tau"]), float(row["selection_score"])
        if np.isfinite(score):
            by_tau[tau] = min(score, by_tau.get(tau, np.inf))
    grid = sorted(by_tau)
    if selected not in grid:
        raise ValueError("pilot-selected tau is absent from its finite candidate grid")
    index = grid.index(selected)
    neighbors = [grid[item] for item in (index - 1, index + 1) if 0 <= item < len(grid)]
    result = [selected]
    if neighbors:
        result.append(min(neighbors, key=lambda tau: by_tau[tau]))
    return result, convergence


def _case_rows(
    tau: float,
    batch: dict[str, Tensor],
    current: Tensor,
    sweep: int,
    requested: int,
    cumulative_seconds: float,
    cover: CartesianCover,
    *,
    carried_forward: bool,
    first_pilot_crossing: Tensor,
) -> list[dict[str, float | int | bool]]:
    l2 = relative_l2(current, batch["solution"], batch["mask"])
    residual = residual_rms(current, batch)
    energy = energy_relative_error(
        current, batch["solution"], batch["logk"], batch["mask"],
    )
    return [{
        "tau": tau,
        "case_index": int(batch["case_index"][item]),
        "requested_checkpoint": requested,
        "actual_sweep": sweep,
        "relative_l2": float(l2[item]),
        "energy_relative_error": float(energy[item]),
        "residual_rms": float(residual[item]),
        "local_nfe": sweep * cover.tile_count,
        "batched_local_calls": sweep,
        "wall_seconds": cumulative_seconds,
        "carried_forward_after_stop": carried_forward,
        "first_pilot_residual_crossing_sweep": int(first_pilot_crossing[item]),
    } for item in range(current.shape[0])]


def strict_converged(
    residual: Tensor,
    relative_update: Tensor,
    strict_residual_tolerance: float,
    relative_update_tolerance: float,
) -> Tensor:
    """Strict convergence; pilot operating tolerance never stops the run."""
    return (
        (residual <= strict_residual_tolerance)
        & (relative_update <= relative_update_tolerance)
    )


@torch.no_grad()
def validate(args: argparse.Namespace) -> dict[str, object]:
    if args.max_sweeps < 512:
        raise ValueError("extended Huang frontier must permit at least 512 sweeps")
    checkpoints = sorted({value for value in args.checkpoints if 0 < value <= args.max_sweeps})
    if args.max_sweeps not in checkpoints:
        checkpoints.append(args.max_sweeps)
    if args.convergence_patience <= 0 or min(
        args.relative_update_tolerance, args.strict_residual_tolerance,
    ) <= 0:
        raise ValueError("convergence controls must be positive")
    pilot = json.loads(args.pilot.read_text())
    taus, convergence_residual = selected_and_adjacent(pilot)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != CONTRACT:
        raise ValueError("extended Huang frontier requires frozen face-local checkpoint")
    statistics = DarcyStatistics.from_json(checkpoint["statistics"])
    adapter = FaceDarcyAdapter(statistics)
    model = build_model(adapter, communication_stage="none").to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = H5DarcyDataset(args.data, "validation")
    if not dataset.has_face_dirichlet:
        raise ValueError("extended Huang frontier requires face-resolved validation data")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    oy, ox = _offset(args.offset)
    rows: list[dict[str, float | int | bool]] = []
    cases_by_tau: dict[str, int] = {}
    for tau in taus:
        processed = 0
        for host_batch in loader:
            if processed >= args.max_validation_cases:
                break
            batch = move_batch(host_batch, device)
            original = batch["solution"].shape[0]
            keep = min(original, args.max_validation_cases - processed)
            for key, value in list(batch.items()):
                if isinstance(value, Tensor) and value.shape[0] == original:
                    batch[key] = value[:keep]
            cover = CartesianCover(
                batch["solution"].shape[-2], batch["solution"].shape[-1],
                MODEL_SPEC["core"], MODEL_SPEC["halo"], oy, ox,
            )
            if not 0.0 < tau < 1.0 / cover.tile_count:
                raise ValueError(f"tau={tau} violates Huang 0<tau<1/K for K={cover.tile_count}")
            current = initial_state(batch, statistics, "zero")
            cumulative = 0.0
            consecutive = 0
            stopped_at: int | None = None
            first_pilot_crossing = torch.full(
                (keep,), -1, device=device, dtype=torch.int64,
            )
            for sweep in range(1, args.max_sweeps + 1):
                previous = current
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                current = face_sni_step(
                    model, adapter, batch, current, cover,
                    "huang_raw_additive", tau,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                cumulative += (time.perf_counter() - started) / keep
                denominator = current.square().flatten(1).sum(1).sqrt().clamp_min(1.0e-12)
                update = (current - previous).square().flatten(1).sum(1).sqrt() / denominator
                residual = residual_rms(current, batch)
                newly_crossed = (
                    (first_pilot_crossing < 0)
                    & (residual <= convergence_residual)
                )
                first_pilot_crossing = torch.where(
                    newly_crossed,
                    torch.full_like(first_pilot_crossing, sweep),
                    first_pilot_crossing,
                )
                converged = bool(torch.all(strict_converged(
                    residual, update, args.strict_residual_tolerance,
                    args.relative_update_tolerance,
                )))
                consecutive = consecutive + 1 if converged else 0
                if sweep in checkpoints:
                    rows.extend(_case_rows(
                        tau, batch, current, sweep, sweep, cumulative, cover,
                        carried_forward=False,
                        first_pilot_crossing=first_pilot_crossing,
                    ))
                if consecutive >= args.convergence_patience:
                    stopped_at = sweep
                    if sweep not in checkpoints:
                        # Preserve the actual stopping state before carrying it
                        # onto later requested frontier checkpoints.
                        rows.extend(_case_rows(
                            tau, batch, current, sweep, sweep, cumulative, cover,
                            carried_forward=False,
                            first_pilot_crossing=first_pilot_crossing,
                        ))
                    break
            final_sweep = args.max_sweeps if stopped_at is None else stopped_at
            for requested in checkpoints:
                if requested > final_sweep:
                    rows.extend(_case_rows(
                        tau, batch, current, final_sweep, requested, cumulative,
                        cover, carried_forward=True,
                        first_pilot_crossing=first_pilot_crossing,
                    ))
            processed += keep
        cases_by_tau[str(tau)] = processed
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "extended_frontier.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary: dict[str, dict[str, float]] = {}
    for tau in taus:
        tau_rows = [row for row in rows if float(row["tau"]) == tau]
        for requested in checkpoints:
            selected_rows = [
                row for row in tau_rows
                if int(row["requested_checkpoint"]) == requested
            ]
            if len(selected_rows) != cases_by_tau[str(tau)]:
                raise RuntimeError(
                    f"frontier checkpoint {requested} does not represent every case"
                )
            summary[f"tau_{tau:.4f}/checkpoint_{requested:04d}"] = {
                key: float(np.mean([float(row[key]) for row in selected_rows]))
                for key in (
                    "relative_l2", "energy_relative_error", "residual_rms",
                    "local_nfe", "batched_local_calls", "wall_seconds",
                )
            }
    final_candidates = []
    for tau in taus:
        values = summary[f"tau_{tau:.4f}/checkpoint_{args.max_sweeps:04d}"]
        final_candidates.append({
            "tau": tau, **values,
            "selection_score": values["relative_l2"] + 0.1 * values["residual_rms"],
        })
    result = {
        "contract": "darcy_huang_raw_additive_extended_validation_v1",
        "selection_split": "validation",
        "test_subsets_opened": False,
        "exact_huang_equation_2": True,
        "assembly": "raw_zero_extension_sum",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "pilot": str(args.pilot.resolve()),
        "taus": taus,
        "pilot_convergence_residual": convergence_residual,
        "strict_residual_tolerance": args.strict_residual_tolerance,
        "relative_update_tolerance": args.relative_update_tolerance,
        "requested_checkpoints": checkpoints,
        "max_sweeps": args.max_sweeps,
        "cases_by_tau": cases_by_tau,
        "summary": summary,
        "selected_extended_control": min(
            final_candidates, key=lambda row: row["selection_score"],
        ),
        "local_nfe_definition": "actual sweeps times overlapping tile count",
        "wall_time_definition": "cumulative synchronized neural recurrence time per case",
        "carried_forward_semantics": "converged cases retain actual stopping NFE/time at later requested checkpoints",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(validate(parse_args()), indent=2))
