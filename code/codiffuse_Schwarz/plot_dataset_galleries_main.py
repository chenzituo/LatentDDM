"""Render the full-page dataset galleries used only by main.tex."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codiffuse_main_gallery_mpl")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "codiffuse_Schwarz/workshop/submission/Two_case/figures"
DARCY_ROOT = ROOT / "codiffuse_Schwarz_Darcy/data"
AIRFOIL_ROOT = Path(
    "/orcd/scratch/orcd/003/zituo/AMR_Diffusion_Foundation/codiffuse_airfoil"
)
LAYOUT = ROOT / "codiffuse_Schwarz_Airfoil/layouts/pitch_rate_ood_v1"


def _finish(figure: plt.Figure, stem: str, metadata: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT / f"{stem}.png", dpi=240, bbox_inches="tight")
    (OUTPUT / f"{stem}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plt.close(figure)


def _even_indices(count: int, samples: int) -> list[int]:
    """Return unique, evenly spaced interior indices without reading fields."""
    if count < samples:
        raise ValueError("the requested gallery is larger than the split")
    indices = [int(round((rank + 0.5) * count / samples - 0.5)) for rank in range(samples)]
    if len(set(indices)) != samples or min(indices) < 0 or max(indices) >= count:
        raise RuntimeError("deterministic index construction was not unique")
    return indices


def _airfoil_candidates(manifest: str, regime: str) -> list[dict]:
    names = json.loads((LAYOUT / manifest).read_text(encoding="utf-8"))["train"]
    rows = []
    for name in names:
        path = AIRFOIL_ROOT / name
        with h5py.File(path, "r") as handle:
            physical = json.loads(handle.attrs["source_manifest_json"])["physical_parameters"]
            pitch_rate = float(np.max(np.abs(handle["motion"][:, 1])))
        rows.append(
            {
                "regime": regime,
                "name": name,
                "path": path,
                "pitch_rate": pitch_rate,
                "reynolds": float(physical["reynolds"]),
                "thickness": float(physical["thickness"]),
                "cycle_amplitude_deg": float(physical["cycle_amplitude_deg"]),
                "cycle_period": float(physical["cycle_period"]),
                "pivot_fraction": float(physical["pivot_fraction"]),
            }
        )
    return sorted(rows, key=lambda row: (row["pitch_rate"], row["name"]))


def _thickness_coverage(candidates: list[dict], samples: int) -> list[dict]:
    """Cover the full Airfoil thickness range without inspecting fields."""
    ranked = sorted(candidates, key=lambda record: (record["thickness"], record["name"]))
    indices = np.rint(np.linspace(0, len(ranked) - 1, samples)).astype(int)
    return [dict(ranked[index]) for index in indices]


def airfoil_gallery() -> None:
    trajectory_rows: list[dict] = []
    for manifest, regime in (("source_split.json", "ID"), ("target_split.json", "OOD")):
        candidates = _airfoil_candidates(manifest, regime)
        trajectory_rows.extend(_thickness_coverage(candidates, 4))

    # The first four phases are on the pitch-down stroke and the final four
    # are on the pitch-up stroke.  Repeating this schedule for every row makes
    # geometry and wake-history differences visually separable.
    target_phases = np.pi * np.asarray((2.0, 2.3, 2.6, 2.9, 3.1, 3.4, 3.7, 4.0))
    selected: list[dict] = []
    vortices: list[np.ndarray] = []
    for row_index, trajectory in enumerate(trajectory_rows):
        with h5py.File(trajectory["path"], "r") as handle:
            phase = np.asarray(handle["motion"][:, 3])
            dx = float(handle.attrs["dx"])
            window = np.asarray(handle.attrs["window"], dtype=float)
            for column_index, target_phase in enumerate(target_phases):
                frame = int(np.argmin(np.abs(phase - target_phase)))
                state = np.asarray(handle["state"][frame], dtype=np.float32)
                sdf = np.asarray(handle["sdf"][frame, 0], dtype=np.float32)
                vorticity = (
                    np.gradient(state[1], dx, axis=-1)
                    - np.gradient(state[0], dx, axis=-2)
                ).astype(np.float32)
                vorticity[sdf < 0.0] = np.nan
                vorticity[:3] = np.nan
                vorticity[-3:] = np.nan
                vorticity[:, :3] = np.nan
                vorticity[:, -3:] = np.nan
                record = dict(trajectory)
                record.update(
                    {
                        "row": row_index,
                        "column": column_index,
                        "frame": frame,
                        "target_phase": float(target_phase),
                        "phase": float(phase[frame]),
                        "angle_deg": float(np.rad2deg(handle["motion"][frame, 0])),
                        "pitch_rate_at_frame": float(handle["motion"][frame, 1]),
                        "sdf": sdf,
                        "window": window,
                        "vorticity": vorticity,
                    }
                )
                selected.append(record)
                vortices.append(vorticity)

    scale = max(float(np.nanpercentile(np.abs(value), 99.5)) for value in vortices)
    zoom = (-0.90, 1.30, -0.90, 0.90)
    figure, axes = plt.subplots(8, 8, figsize=(5.40, 6.25))
    figure.subplots_adjust(left=0.090, right=0.925, top=0.955, bottom=0.045,
                           wspace=0.045, hspace=0.055)
    image = None
    for axis, record in zip(axes.flat, selected):
        xmin, xmax, ymin, ymax = record["window"]
        image = axis.imshow(
            record["vorticity"], origin="lower", extent=(xmin, xmax, ymin, ymax),
            aspect="equal", cmap="RdBu_r", vmin=-scale, vmax=scale,
            interpolation="bilinear", rasterized=True,
        )
        axis.contour(
            np.linspace(xmin, xmax, record["sdf"].shape[1]),
            np.linspace(ymin, ymax, record["sdf"].shape[0]),
            record["sdf"], levels=(0.0,), colors="black", linewidths=0.30,
        )
        axis.set_xlim(zoom[0], zoom[1])
        axis.set_ylim(zoom[2], zoom[3])
        axis.set_xticks([])
        axis.set_yticks([])
        if record["row"] == 0:
            axis.set_title(
                rf"${record['target_phase'] / np.pi:.1f}\pi$", fontsize=5.4, pad=1.5
            )
        for spine in axis.spines.values():
            spine.set_color("#888888")
            spine.set_linewidth(0.25)
    for row_index, trajectory in enumerate(trajectory_rows):
        y = 0.955 - (row_index + 0.5) * (0.955 - 0.045) / 8
        figure.text(
            0.064,
            y,
            f"$t/c$\n{trajectory['thickness']:.3f}",
            ha="center",
            va="center",
            fontsize=4.9,
            linespacing=0.9,
        )
    figure.text(0.014, 0.728, "ID source", rotation=90, ha="center", va="center",
                fontsize=7.0, fontweight="bold")
    figure.text(0.014, 0.272, "OOD target", rotation=90, ha="center", va="center",
                fontsize=7.0, fontweight="bold")
    figure.text(0.30, 0.992, "pitch-down stroke", ha="center", va="top", fontsize=5.8)
    figure.text(0.72, 0.992, "pitch-up stroke", ha="center", va="top", fontsize=5.8)
    color_axis = figure.add_axes((0.945, 0.20, 0.014, 0.60))
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label("vorticity $\\omega$", fontsize=7)
    colorbar.ax.tick_params(labelsize=6, length=2)
    metadata = {
        "contract": "airfoil_dataset_64_gallery_main_v2",
        "selection": (
            "four evenly spaced thickness ranks per regime, including the endpoints; "
            "eight fixed second-cycle phases per trajectory"
        ),
        "trajectory_selection_features": ["thickness"],
        "field_or_appearance_used_for_selection": False,
        "model_error_used_for_selection": False,
        "snapshot_phase_targets": target_phases.tolist(),
        "display_absolute_percentile": 99.5,
        "display_limit": scale,
        "display_zoom": list(zoom),
        "samples": [
            {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in record.items()
                if key not in {"sdf", "window", "vorticity", "path"}
            }
            for record in selected
        ],
    }
    _finish(figure, "airfoil_dataset_64_samples_main", metadata)


def darcy_gallery() -> None:
    specifications = (
        ("3x3 source", "96 x 96", DARCY_ROOT / "darcy_full_domain_face_96_v2.h5"),
        ("4x4 target", "128 x 128", DARCY_ROOT / "darcy_domain_extension_fixed_dx_4x4_v1.h5"),
        ("8x8 target", "256 x 256", DARCY_ROOT / "darcy_domain_extension_fixed_dx_8x8_v1.h5"),
        ("16x16 target", "512 x 512", DARCY_ROOT / "darcy_domain_extension_fixed_dx_16x16_v1.h5"),
    )
    groups: list[dict] = []
    active_values: list[np.ndarray] = []
    for label, resolution, path in specifications:
        samples = []
        with h5py.File(path, "r") as handle:
            group = handle["val"]
            for index in _even_indices(len(group["solution"]), 8):
                solution = np.asarray(group["solution"][index, 0], dtype=np.float32)
                mask = np.asarray(group["mask"][index, 0], dtype=np.float32) > 0.5
                active_values.append(solution[mask])
                samples.append(
                    {
                        "file": path.name,
                        "split": "val",
                        "index": index,
                        "seed": int(group["seed"][index]),
                        "family": group["family"][index].decode("utf-8"),
                        "solution": solution,
                        "mask": mask,
                    }
                )
        groups.append({"label": label, "resolution": resolution, "samples": samples})

    values = np.concatenate(active_values)
    vmin, vmax = np.quantile(values, (0.005, 0.995))
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("white")
    figure = plt.figure(figsize=(5.40, 6.35))
    outer = figure.add_gridspec(2, 2, left=0.035, right=0.91, top=0.975,
                                bottom=0.035, wspace=0.10, hspace=0.10)
    image = None
    for group_index, record_group in enumerate(groups):
        inner = outer[group_index // 2, group_index % 2].subgridspec(
            3, 3, wspace=0.035, hspace=0.035
        )
        sample_cursor = 0
        for cell in range(9):
            axis = figure.add_subplot(inner[cell // 3, cell % 3])
            if cell == 4:
                axis.axis("off")
                axis.text(
                    0.5, 0.55, record_group["label"], ha="center", va="center",
                    fontsize=7.5, fontweight="bold", transform=axis.transAxes,
                )
                axis.text(
                    0.5, 0.36, record_group["resolution"], ha="center", va="center",
                    fontsize=6.2, color="#555555", transform=axis.transAxes,
                )
                continue
            sample = record_group["samples"][sample_cursor]
            sample_cursor += 1
            field = np.ma.masked_where(~sample["mask"], sample["solution"])
            image = axis.imshow(
                field, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                interpolation="bilinear", rasterized=True,
            )
            axis.contour(sample["mask"].astype(float), levels=(0.5,), colors="#202020",
                         linewidths=0.22, origin="lower")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#888888")
                spine.set_linewidth(0.25)
    color_axis = figure.add_axes((0.935, 0.19, 0.014, 0.62))
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label("pressure $u$", fontsize=7)
    colorbar.ax.tick_params(labelsize=6, length=2)
    metadata = {
        "contract": "darcy_dataset_32_gallery_main_v1",
        "selection": "eight evenly spaced validation indices at each cover",
        "solution_or_prediction_used_for_selection": False,
        "display_quantiles": [0.005, 0.995],
        "display_limits": [float(vmin), float(vmax)],
        "groups": [
            {
                "label": group["label"],
                "resolution": group["resolution"],
                "samples": [
                    {key: value for key, value in sample.items() if key not in {"solution", "mask"}}
                    for sample in group["samples"]
                ],
            }
            for group in groups
        ],
    }
    _finish(figure, "darcy_dataset_32_samples_main", metadata)


if __name__ == "__main__":
    airfoil_gallery()
    darcy_gallery()
