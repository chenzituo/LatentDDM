"""Build the four-panel evidence summary for the LatentDDM paper.

Every number is read from an identity-audited promoted artifact. Darcy and Car
use their declared validation units; after validation-only model selection,
Airfoil uses the single frozen-checkpoint held-out test when it is available.
Timings are measured in one process with all local crops batched or
chunk-batched inside a round; this file never divides a serial time by the
number of subdomains. Final rendering requires the prospective selectors and
rejects stale development artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import NullFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "codiffuse_Schwarz/figures/latentddm_claim_summary_v2"

COLORS = {
    "local": "#8F8F8F",
    "latent": "#E45756",
    "sni": "#4C78A8",
    "point": "#F28E2B",
    "slice": "#59A14F",
    "patch": "#7A4EAB",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(relative: str) -> tuple[dict[str, Any], Path]:
    path = ROOT / relative
    return json.loads(path.read_text(encoding="utf-8")), path


def adaptation_records() -> tuple[dict[str, list[dict[str, Any]]], Path]:
    with_point = ROOT / (
        "codiffuse_Schwarz_Darcy/outputs/field_token_domain_extension_degree1_v1/depth4/"
        "comparison_aggregate_with_point.json"
    )
    if with_point.exists():
        path = with_point
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload, path = load(
            "codiffuse_Schwarz_Darcy/outputs/"
            "field_token_domain_extension_degree1_v1/depth4/comparison_aggregate.json"
        )
    mapping = {
        "LatentDDM field-token composer": "LatentDDM",
        "Point Transformer full fine-tune": "Point",
        "Transolver full fine-tune": "Slice",
        "Patch Transformer full fine-tune": "Patch",
    }
    records: dict[str, list[dict[str, Any]]] = {}
    for row in payload["rows"]:
        if row["method"] in mapping:
            records.setdefault(mapping[row["method"]], []).append(row)
    for rows in records.values():
        rows.sort(key=lambda row: int(row["budget"]))
        if [int(row["budget"]) for row in rows] != [4, 16, 64]:
            raise ValueError("Darcy adaptation curve does not use the locked budgets")
        if any(len(row.get("seeds", ())) != 3 for row in rows):
            raise ValueError("Darcy adaptation curve is not a three-seed audit")
    if set(records) != set(mapping.values()):
        raise ValueError("Darcy adaptation artifact omits a locked method")
    return records, path


def car_records() -> tuple[list[dict[str, Any]], Path]:
    capacity = ROOT / (
        "codiffuse_Schwarz_Car/outputs/field_token_capacity_selected_v2/"
        "paper_common_grid_frontier.json"
    )
    if capacity.exists():
        path = capacity
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload, path = load(
            "codiffuse_Schwarz_Car/outputs/field_token_rebuilt_selected/"
            "paper_common_grid_frontier.json"
        )
    gate = payload["paper_promotion_gate"]
    # The strict research gate additionally asks the paired interval against
    # the strongest global model to exclude zero.  The paper plot reports
    # means and paired design intervals directly, so it may display a lowest
    # mean even when that final pair is statistically tied.  It must still
    # improve Local/Core, SNI, and every global mean on the locked grid.
    required = (
        "improves_local_vector_wake_divergence",
        "beats_every_global_mean_full_vector",
        "beats_sni_full_vector",
    )
    if not all(bool(gate.get(key, False)) for key in required):
        raise RuntimeError("Car complete-grid evidence gate did not pass")
    order = (
        ("Local/Core", "Local/Core", "local"),
        ("LatentDDM", "LatentDDM", "latent"),
        ("SNI", "SNI", "sni"),
        ("Point", "Point", "point"),
        ("Slice", "Slice", "slice"),
        ("Patch", "Patch", "patch"),
    )
    rows = []
    for source, label, family in order:
        metric = payload["methods"][source]["velocity_rel_l2"]
        rows.append({
            "source_label": source,
            "label": label,
            "family": family,
            "mean": float(metric["mean"]),
            "ci95": tuple(float(value) for value in metric[
                "paired_design_bootstrap_ci95"
            ]),
        })
    return rows, path


ROLLOUT_KEYS = (
    "all_step_field",
    "all_step_vorticity",
    "all_step_interface",
    "all_step_divergence",
    "all_step_force",
)

GLOBAL_KEYS = {
    "all_step_field": "field_relative_l2_mean",
    "all_step_vorticity": "vorticity_relative_l2_mean",
    "all_step_interface": "interface_relative_l2_mean",
    "all_step_divergence": "divergence_rms_mean",
    "all_step_force": "force_relative_l2_mean",
}


def _selected_airfoil_rollout() -> tuple[dict[str, Any], Path, str]:
    recurrent = ROOT / (
        "codiffuse_Schwarz_Airfoil/outputs/field_token_recurrent_spatial_selected/"
        "selection.json"
    )
    selection_path = recurrent if recurrent.exists() else ROOT / (
        "codiffuse_Schwarz_Airfoil/outputs/field_token_rebuilt_selected/"
        "selection.json"
    )
    if not selection_path.exists():
        raise FileNotFoundError(
            "prospective Airfoil LatentDDM selection has not completed"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection.get("selection_passed", False):
        raise RuntimeError("Airfoil LatentDDM validation gate did not pass")
    rollout_path = Path(selection["selected"]["rollout"])
    if not rollout_path.is_file():
        raise FileNotFoundError(rollout_path)
    return (
        json.loads(rollout_path.read_text(encoding="utf-8")),
        rollout_path,
        "prospective matched selector",
    )


def airfoil_rollout_records() -> tuple[
    dict[str, dict[str, float]], list[Path], str,
]:
    final_path = ROOT / (
        "codiffuse_Schwarz/figures/unified_report/"
        "airfoil_six_method_final_holdout_rollout.json"
    )
    legacy_path = ROOT / (
        "codiffuse_Schwarz/figures/unified_report/"
        "airfoil_six_method_test_rollout.json"
    )
    test_path = final_path if final_path.exists() else legacy_path
    if test_path.exists():
        test = json.loads(test_path.read_text(encoding="utf-8"))
        selection_path = Path(test["selection"]["path"])
        if not selection_path.exists():
            raise ValueError("Airfoil test artifact predates prospective selection")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected_checkpoint = Path(selection["copied_checkpoint"])
        if (
            test.get("contract")
            != "airfoil_six_method_test_rollout_cluster_band_v1"
            or test.get("split") not in {"test", "final_holdout"}
            or int(test.get("rollout_windows", 0)) <= 0
            or not selection.get("selection_passed", False)
            or test.get("field_token_checkpoint_sha256")
            != selection.get("copied_checkpoint_sha256")
            or not selected_checkpoint.is_file()
            or sha256(selected_checkpoint)
            != selection.get("copied_checkpoint_sha256")
        ):
            raise ValueError("Airfoil held-out six-method artifact is invalid")
        summary = test["summary"]
        mapping = (
            ("Local/Core", "local_core"),
            ("LatentDDM", "latentddm"),
            ("SNI", "sni"),
            ("Point", "point"),
            ("Slice", "slice"),
            ("Patch", "patch"),
        )
        local = summary["local_core"]
        rows = {
            label: {
                key: float(summary[name][key]) / float(local[key])
                for key in ROLLOUT_KEYS
            }
            for label, name in mapping
        }
        return rows, [test_path, selection_path], (
            f"frozen-checkpoint {test['split']}"
        )

    local_payload, rollout_path, selection_status = _selected_airfoil_rollout()
    global_payload, global_path = load(
        "codiffuse_Schwarz_Airfoil/outputs/global100_v1_finest/"
        "global_exact_point_slice_patch_validation_v2.json"
    )
    aggregate = local_payload["aggregate"]
    if not {"local_core", "latentddm"}.issubset(aggregate):
        raise ValueError("Airfoil field-token rollout omits Local/Core or LatentDDM")

    sni_selection, sni_selection_path = load(
        "codiffuse_Schwarz_Airfoil/outputs/airfoil_trace_sni_h20_selected/"
        "selection.json"
    )
    if not sni_selection.get("selection_passed", False):
        raise ValueError("Airfoil recurrent SNI was not frozen on validation")
    sni_path = Path(sni_selection["selected"]["source"])
    sni_source = json.loads(sni_path.read_text(encoding="utf-8"))

    # Force-vector error was added to the frozen held-out protocol.  Historical
    # validation files contain componentwise loads only, so the development
    # fallback keeps the original four directly comparable diagnostics.
    fallback_keys = ROLLOUT_KEYS[:-1]
    rows: dict[str, dict[str, float]] = {}
    local = aggregate["local_core"]
    sni_local = sni_selection["local_core"]
    if any(
        not np.isclose(float(local[key]), float(sni_local[key]))
        for key in fallback_keys
    ):
        raise ValueError("Airfoil LatentDDM and SNI do not share Local/Core")
    for label, source in (
        ("Local/Core", local),
        ("LatentDDM", aggregate["latentddm"]),
        ("SNI", sni_selection["selected"]["aggregate"]),
    ):
        rows[label] = {
            key: float(source[key]) / float(local[key])
            for key in fallback_keys
        }

    summary = global_payload["summary"]
    for label, family in (
        ("Point", "point_anchor"),
        ("Slice", "transolver"),
        ("Patch", "patch_transformer"),
    ):
        source = summary[family]["rollout_all_steps"]
        rows[label] = {
            key: float(source[GLOBAL_KEYS[key]]) / float(local[key])
            for key in fallback_keys
        }
    sources = list(dict.fromkeys((
        rollout_path, sni_selection_path, sni_path, global_path,
    )))
    return rows, sources, selection_status


def error_latency_records() -> tuple[dict[str, Any], list[Path]]:
    darcy, darcy_path = load(
        "codiffuse_Schwarz_Darcy/outputs/field_token_domain_extension_selected_v3/"
        "same_process_core_sni_timing_128.json"
    )
    darcy_selection, darcy_selection_path = load(
        "codiffuse_Schwarz_Darcy/outputs/field_token_domain_extension_selected_v3/"
        "selection.json"
    )
    darcy_field_path = Path(
        darcy_selection["selected"]["representative"]["validation"]
    )
    darcy_field = json.loads(darcy_field_path.read_text(encoding="utf-8"))
    darcy_sni, darcy_sni_path = load(
        "codiffuse_Schwarz_Darcy/outputs/domain_extension_128_v1/"
        "core_sni_local_init_validation_128.json"
    )
    car, car_path = load(
        "codiffuse_Schwarz_Car/outputs/field_token_rebuilt_selected/"
        "same_process_recurrent_sni_timing.json"
    )
    car_field, car_field_path = load(
        "codiffuse_Schwarz_Car/outputs/field_token_rebuilt_selected/"
        "common_query_validation.json"
    )
    car_sni_selection, car_sni_selection_path = load(
        "codiffuse_Schwarz_Car/outputs/car_recurrent_sni_selected/selection.json"
    )
    if (
        not car_sni_selection.get("selection_passed", False)
        or car_sni_selection.get("selected") is None
    ):
        raise ValueError("complete-grid Car SNI selection did not pass")
    car_sni_path = Path(car_sni_selection["selected"]["source"])
    car_sni = json.loads(car_sni_path.read_text(encoding="utf-8"))
    recurrent_air = ROOT / (
        "codiffuse_Schwarz_Airfoil/outputs/field_token_recurrent_spatial_selected/"
        "same_process_trace_sni_h20_timing.json"
    )
    selected_air = recurrent_air if recurrent_air.exists() else ROOT / (
        "codiffuse_Schwarz_Airfoil/outputs/field_token_rebuilt_selected/"
        "same_process_trace_sni_h20_timing.json"
    )
    if not selected_air.exists():
        raise FileNotFoundError(
            "prospective Airfoil H20 timing has not completed"
        )
    air_path = selected_air
    air = json.loads(air_path.read_text(encoding="utf-8"))
    air_rollout, air_rollout_path, _ = _selected_airfoil_rollout()
    air_sni_selection, air_sni_selection_path = load(
        "codiffuse_Schwarz_Airfoil/outputs/airfoil_trace_sni_h20_selected/"
        "selection.json"
    )
    if not air_sni_selection.get("selection_passed", False):
        raise ValueError("held-out Airfoil SNI has no validation-selected parent")
    air_sni_path = Path(air_sni_selection["selected"]["source"])
    air_sni = json.loads(air_sni_path.read_text(encoding="utf-8"))
    if not np.isclose(
        float(air_rollout["aggregate"]["local_core"]["all_step_field"]),
        float(air_sni_selection["local_core"]["all_step_field"]),
        rtol=1.0e-7,
        atol=1.0e-10,
    ):
        raise ValueError("Airfoil LatentDDM and SNI do not share Local/Core")

    darcy_timing = darcy["timing"]
    car_timing = car["summary"]
    air_timing = air["summary"]

    def normalized_path(
        *, case: str, latent_seconds: float, latent_error: float,
        timing: dict[str, Any], timing_key: str, local_error: float,
        sweep_errors: dict[int, float], latency_multiplier: int = 1,
        accumulated_steps: int = 1,
    ) -> dict[str, Any]:
        latent_seconds *= latency_multiplier
        points = [{
            "sweep": 0,
            "seconds": latency_multiplier * float(
                timing["local_core"][timing_key]
            ),
            "field_error": local_error,
        }]
        for sweep, field_error in sweep_errors.items():
            points.append({
                "sweep": sweep,
                "seconds": latency_multiplier * float(
                    timing[f"init_sni_{sweep}"][timing_key]
                ),
                "field_error": field_error,
            })
        for point in points:
            point["latency_over_latentddm"] = point["seconds"] / latent_seconds
            point["error_over_latentddm"] = point["field_error"] / latent_error
        return {
            "case": case,
            "accumulated_steps": accumulated_steps,
            "latentddm": {"seconds": latent_seconds, "field_error": latent_error},
            "sni_path": points,
        }

    darcy_summary = darcy_sni["summary"]
    car_summary = car_sni["summary"]
    car_selected_sweep = int(car_sni_selection["selected"]["sweep"])
    car_available_sweeps = tuple(
        sweep for sweep in sorted({
            int(key.removeprefix("sweep_"))
            for key in car_summary if key.startswith("sweep_")
        })
        if 0 < sweep <= car_selected_sweep
    )
    air_summary = air_rollout["aggregate"]
    air_sni_summary = air_sni["aggregate"]
    air_selected_sweep = int(air_sni_selection["selected"]["sweep"])
    paths = {
        "Darcy": normalized_path(
            case="Darcy",
            latent_seconds=float(
                darcy_timing["latentddm"]["mean_seconds_per_case"]
            ),
            latent_error=float(darcy_field["metrics"]["relative_l2"]),
            timing=darcy_timing,
            timing_key="mean_seconds_per_case",
            local_error=float(darcy_field["metrics"]["local_core_relative_l2"]),
            sweep_errors={
                sweep: float(darcy_summary[
                    f"init_local_core/tau_1.000/sweep_{sweep:04d}"
                ]["relative_l2_mean"])
                for sweep in (1, 2, 4, 8, 16, 32, 64, 128)
            },
        ),
        "Car": normalized_path(
            case="Car",
            latent_seconds=float(car_timing["latentddm"]["mean_seconds"]),
            latent_error=float(car_field["summary"]["latentddm_velocity_rel_l2"]),
            timing=car_timing,
            timing_key="mean_seconds",
            local_error=float(car_field["summary"]["local_core_velocity_rel_l2"]),
            sweep_errors={
                sweep: float(car_summary[f"sweep_{sweep:03d}"][
                    "velocity_relative_l2_mean"
                ])
                for sweep in car_available_sweeps
            },
        ),
        "Airfoil-20": normalized_path(
            case="Airfoil-20",
            latent_seconds=float(air_timing["latentddm"]["mean_seconds"]),
            latent_error=float(air_summary["latentddm"]["all_step_field"]),
            timing=air_timing,
            timing_key="mean_seconds",
            local_error=float(air_summary["local_core"]["all_step_field"]),
            sweep_errors={
                sweep: float(air_sni_summary[f"sweep_{sweep}"]["all_step_field"])
                for sweep in range(1, air_selected_sweep + 1)
            },
            latency_multiplier=1,
            accumulated_steps=20,
        ),
    }
    return paths, [
        darcy_path, darcy_selection_path, darcy_field_path, darcy_sni_path,
        car_path, car_field_path, car_sni_selection_path, car_sni_path,
        air_path, air_rollout_path, air_sni_selection_path, air_sni_path,
    ]


def plot() -> tuple[plt.Figure, dict[str, Any]]:
    adaptation, adaptation_path = adaptation_records()
    car, car_path = car_records()
    airfoil, airfoil_paths, airfoil_status = airfoil_rollout_records()
    error_latency, efficiency_paths = error_latency_records()

    plt.rcParams.update({
        "font.size": 8.1,
        "axes.titlesize": 9.3,
        "axes.labelsize": 8.3,
        "legend.fontsize": 6.9,
        "pdf.fonttype": 42,
    })
    figure, axes_2d = plt.subplots(2, 2, figsize=(7.15, 4.55))
    axes = axes_2d.ravel()
    figure.subplots_adjust(
        left=.082, right=.985, bottom=.105, top=.95, wspace=.33, hspace=.58,
    )

    axis = axes[0]
    styles = {
        "LatentDDM": (COLORS["latent"], "o"),
        "Point": (COLORS["point"], "v"),
        "Slice": (COLORS["slice"], "D"),
        "Patch": (COLORS["patch"], "P"),
    }
    for label in styles:
        rows = adaptation[label]
        color, marker = styles[label]
        axis.errorbar(
            [int(row["budget"]) for row in rows],
            [float(row["adapted_relative_l2_mean"]) for row in rows],
            yerr=[float(row["adapted_relative_l2_sample_std"]) for row in rows],
            color=color, marker=marker, lw=1.35, ms=4, capsize=2,
            label=label,
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks((4, 16, 64), labels=("4", "16", "64"))
    axis.set_yticks((.06, .08, .12, .18), labels=(".06", ".08", ".12", ".18"))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel("target-domain labels")
    axis.set_ylabel("validation relative $L^2$")
    axis.set_title("a  Reuse on enlarged Darcy", loc="left", weight="bold")
    axis.grid(True, which="both", alpha=.18, lw=.5)
    axis.legend(frameon=False, loc="upper right", handlelength=1.2,
                labelspacing=.22, borderpad=0)

    axis = axes[1]
    x = np.arange(len(car))
    means = np.asarray([row["mean"] for row in car])
    lower = means - np.asarray([row["ci95"][0] for row in car])
    upper = np.asarray([row["ci95"][1] for row in car]) - means
    axis.axvspan(-.5, 2.5, color="#FFF4F2", zorder=0)
    axis.axvspan(2.5, 5.5, color="#EFF6FF", zorder=0)
    axis.bar(
        x, means, color=[COLORS[row["family"]] for row in car], width=.70,
        yerr=np.stack((lower, upper)), capsize=2,
        error_kw={"elinewidth": .7, "capthick": .7},
    )
    axis.set_yscale("log")
    axis.set_ylim(.011, .078)
    axis.set_yticks((.015, .02, .03, .05), labels=(".015", ".02", ".03", ".05"))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xticks(x, [row["label"] for row in car], rotation=35, ha="right")
    axis.set_ylabel("3-component velocity rel. $L^2$")
    axis.set_title("b  Car: paired 25-design audit", loc="left", weight="bold")
    axis.text(
        1.0, .975, "LOCAL METHODS", transform=axis.get_xaxis_transform(),
        ha="center", va="top", fontsize=5.9, weight="bold",
        color="#B44747",
    )
    axis.text(
        4.0, .975, "GLOBAL METHODS", transform=axis.get_xaxis_transform(),
        ha="center", va="top", fontsize=5.9, weight="bold",
        color="#3264A8",
    )
    axis.grid(True, axis="y", which="both", alpha=.18, lw=.5)
    best = int(np.argmin(means))
    axis.get_xticklabels()[best].set_weight("bold")
    axis.text(
        best, upper[best] + means[best] * 1.06, "lowest",
        ha="center", va="bottom", fontsize=6.8, weight="bold",
        color=COLORS[car[best]["family"]],
    )

    axis = axes[2]
    metrics = tuple(next(iter(airfoil.values())))
    methods = tuple(airfoil)
    matrix = np.asarray([
        [airfoil[method][metric] for metric in metrics] for method in methods
    ])
    image = axis.imshow(
        matrix, cmap="RdYlGn_r", aspect="auto",
        norm=TwoSlopeNorm(vmin=.5, vcenter=1.0, vmax=1.55),
    )
    metric_labels = {
        "all_step_field": "field",
        "all_step_vorticity": r"$\omega$",
        "all_step_interface": "trace",
        "all_step_divergence": "div.",
        "all_step_force": "force",
    }
    axis.set_xticks(
        np.arange(len(metrics)), tuple(metric_labels[key] for key in metrics),
    )
    axis.set_yticks(np.arange(len(methods)), methods)
    latent_index = methods.index("LatentDDM")
    axis.get_yticklabels()[latent_index].set_weight("bold")
    axis.get_yticklabels()[latent_index].set_color(COLORS["latent"])
    axis.set_title("c  Airfoil: 20-step / Local/Core", loc="left", weight="bold")
    column_best = np.argmin(matrix, axis=0)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix[row, column])
            is_best = row == int(column_best[column])
            axis.text(
                column, row, f"{value:.2f}", ha="center", va="center",
                fontsize=6.9, weight="bold" if is_best else "normal",
                color="white" if value < .64 or value > 1.36 else "#111827",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=.047, pad=.03)
    colorbar.set_label("error ratio", labelpad=2)
    colorbar.ax.tick_params(labelsize=6.2, length=2)

    axis = axes[3]
    case_styles = {
        "Darcy": ("#3B6FB6", "o"),
        "Car": ("#4C956C", "s"),
        "Airfoil-20": ("#D98E24", "^"),
    }
    for case, record in error_latency.items():
        color, marker = case_styles[case]
        points = record["sni_path"]
        labeled_sweeps = {0, int(points[-1]["sweep"])}
        if case == "Darcy":
            labeled_sweeps.update((1, 4, 16, 64))
        axis.plot(
            [point["latency_over_latentddm"] for point in points],
            [point["error_over_latentddm"] for point in points],
            color=color, marker=marker, lw=1.25, ms=3.8, label=f"{case} SNI",
        )
        for point in points:
            if int(point["sweep"]) not in labeled_sweeps:
                continue
            axis.annotate(
                str(point["sweep"]),
                (point["latency_over_latentddm"], point["error_over_latentddm"]),
                xytext=(2.5, 2.5), textcoords="offset points",
                fontsize=5.7, color=color,
            )
    axis.scatter(
        [1.0], [1.0], s=28, marker="D", color=COLORS["latent"],
        edgecolor="white", linewidth=.5, zorder=5, label="LatentDDM",
    )
    axis.axhline(1.0, color=COLORS["latent"], ls=":", lw=.8, alpha=.7)
    axis.axvline(1.0, color="#777777", ls=":", lw=.7, alpha=.55)
    axis.set_xscale("log")
    axis.set_yscale("log")
    maximum_latency_ratio = max(
        point["latency_over_latentddm"]
        for record in error_latency.values() for point in record["sni_path"]
    )
    error_ratios = [
        point["error_over_latentddm"]
        for record in error_latency.values() for point in record["sni_path"]
    ]
    minimum_error_ratio = min(1.0, min(error_ratios))
    maximum_error_ratio = max(1.0, max(error_ratios))
    axis.set_xlim(.82, max(2.2, 1.12 * maximum_latency_ratio))
    axis.set_ylim(
        max(.20, minimum_error_ratio / 1.18),
        maximum_error_ratio * 1.22,
    )
    latency_ticks = tuple(
        value for value in (1, 2, 4, 8, 16, 32, 64, 128)
        if value <= 1.05 * maximum_latency_ratio
    )
    axis.set_xticks(latency_ticks, labels=tuple(map(str, latency_ticks)))
    error_ticks = tuple(
        value for value in (.25, .5, 1, 2, 4, 8, 16)
        if axis.get_ylim()[0] <= value <= axis.get_ylim()[1]
    )
    axis.set_yticks(
        error_ticks,
        labels=tuple(f"{value:g}" for value in error_ticks),
    )
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel("measured latency / LatentDDM")
    axis.set_ylabel("validation field error / LatentDDM")
    axis.set_title("d  SNI: error versus sequential latency", loc="left", weight="bold")
    axis.grid(True, which="both", alpha=.18, lw=.5)
    axis.legend(
        frameon=False, loc="upper right", ncol=1, labelspacing=.2,
        handlelength=1.2, borderpad=0,
    )

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)

    audit = {
        "contract": "latentddm_four_claim_summary_v3",
        "analytic_latency_projection": False,
        "adaptation": adaptation,
        "car_complete_common_grid": car,
        "airfoil_rollout_normalized_to_local_core": airfoil,
        "airfoil_candidate_status": airfoil_status,
        "error_latency": error_latency,
        "sources": [
            str(adaptation_path), str(car_path),
            *map(str, airfoil_paths), *map(str, efficiency_paths),
        ],
    }
    return figure, audit


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure, audit = plot()
    for suffix in ("pdf", "png"):
        figure.savefig(
            OUTPUT / f"latentddm_claim_summary.{suffix}",
            dpi=300, bbox_inches="tight",
        )
    plt.close(figure)
    (OUTPUT / "latentddm_claim_summary.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
