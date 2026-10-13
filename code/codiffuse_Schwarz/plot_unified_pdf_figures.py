"""Generate explanatory and quantitative figures for the unified PDF report."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle,
)
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent / "figures" / "unified_report"


def save_figure(figure: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / f"{stem}.png", dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def method_schematic() -> None:
    """Draw the method as a visual story with minimal prose."""
    figure, axis = plt.subplots(figsize=(14.5, 4.25))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    panel_x = (0.018, 0.265, 0.512, 0.759)
    panel_w, panel_y, panel_h = 0.223, 0.235, 0.655
    panel_colors = ("#eff6ff", "#f5f3ff", "#fff1f2", "#ecfdf5")
    titles = ("New geometry", "Frozen local physics", "Field-token exchange", "Decode + owned cores")
    equations = (
        r"$x_i=P_iX$",
        r"$v_i=\mathcal{F}_\theta(x_i)\in\mathcal{M}_i$",
        r"$z_i^0=C_\phi(x_i,v_i),\quad z_P^{\ell+1}=\tau_\phi^\ell(z_P^\ell;P)$",
        r"$\widehat u=\mathcal{A}_{\rm core}(v_i+\delta_i)$",
    )
    for index, (x, color, title, equation) in enumerate(
        zip(panel_x, panel_colors, titles, equations), start=1,
    ):
        axis.add_patch(FancyBboxPatch(
            (x, panel_y), panel_w, panel_h,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            facecolor=color, edgecolor="#cbd5e1", linewidth=1.2,
        ))
        axis.add_patch(Circle(
            (x + .025, panel_y + panel_h - .045), .020,
            facecolor="#0f172a", edgecolor="none",
        ))
        axis.text(x + .025, panel_y + panel_h - .045, str(index),
                  color="white", ha="center", va="center", fontsize=9,
                  weight="bold")
        axis.text(x + .052, panel_y + panel_h - .045, title,
                  ha="left", va="center", fontsize=10.2, weight="bold",
                  color="#0f172a")
        axis.text(x + panel_w / 2, panel_y + .050, equation,
                  ha="center", va="center", fontsize=9.4, color="#334155")
    for left, right in zip(panel_x[:-1], panel_x[1:]):
        axis.add_patch(FancyArrowPatch(
            (left + panel_w + .004, .56), (right - .004, .56),
            arrowstyle="-|>", mutation_scale=16, linewidth=1.7,
            color="#475569", zorder=8,
        ))

    # 1: irregular world with disjoint owned cores and faint halo context.
    x = panel_x[0]
    domain = np.array([
        [x + .033, .40], [x + .043, .69], [x + .090, .80],
        [x + .176, .76], [x + .196, .60], [x + .168, .36],
        [x + .105, .31], [x + .055, .35],
    ])
    axis.add_patch(Polygon(
        domain, closed=True, facecolor="#bae6fd", edgecolor="#0369a1",
        linewidth=1.6,
    ))
    tile_colors = ("#2563eb", "#7c3aed", "#0891b2", "#ea580c")
    core_boxes = ((.047, .405), (.110, .405), (.047, .545), (.110, .545))
    for (dx, dy), color in zip(core_boxes, tile_colors):
        axis.add_patch(Rectangle(
            (x + dx - .009, dy - .012), .081, .164,
            facecolor="none", edgecolor=color, linewidth=.75,
            linestyle="--", alpha=.38,
        ))
        axis.add_patch(Rectangle(
            (x + dx, dy), .063, .140, facecolor="none", edgecolor=color,
            linewidth=1.45, alpha=.95,
        ))
    axis.add_patch(Circle((x + .112, .56), .022, facecolor="white",
                          edgecolor="#0369a1", linewidth=1.2))

    # 2: shared local operator produces detailed patch fields in parallel.
    x = panel_x[1]
    fbox = FancyBboxPatch(
        (x + .077, .66), .070, .075,
        boxstyle="round,pad=.008,rounding_size=.012",
        facecolor="#7c3aed", edgecolor="none",
    )
    axis.add_patch(fbox)
    axis.text(x + .112, .697, r"$\mathcal{F}_\theta$", ha="center", va="center",
              color="white", fontsize=12, weight="bold")
    patch_centers = (x + .055, x + .112, x + .169)
    for k, cx in enumerate(patch_centers):
        axis.add_patch(FancyArrowPatch(
            (x + .112, .66), (cx, .598), arrowstyle="-|>",
            mutation_scale=11, linewidth=1.0, color="#7c3aed",
        ))
        rect = Rectangle((cx - .034, .405), .068, .175,
                         facecolor=("#dbeafe", "#ede9fe", "#cffafe")[k],
                         edgecolor=tile_colors[k], linewidth=1.25)
        axis.add_patch(rect)
        yy = np.linspace(.425, .555, 4)
        for j, yline in enumerate(yy):
            bend = .006 * np.sin((j + k) * 1.4)
            axis.plot([cx - .026, cx, cx + .026],
                      [yline, yline + bend, yline - bend],
                      color=("#2563eb", "#7c3aed", "#0891b2")[k],
                      linewidth=.9, alpha=.85)

    # 3: completed local fields become tokens before a shallow latent stack.
    x = panel_x[2]
    layer_y = (.625, .535, .445)
    node_x = (x + .048, x + .093, x + .138, x + .183)
    for layer, y0 in enumerate(layer_y):
        for a in range(len(node_x)):
            for b in range(a + 1, len(node_x)):
                axis.plot(
                    (node_x[a], node_x[b]), (y0, y0),
                    color="#f43f5e", linewidth=.52, alpha=.40,
                )
        for node in node_x:
            axis.add_patch(Circle(
                (node, y0), .014,
                facecolor=("#fb7185", "#e11d48", "#be123c")[layer],
                edgecolor="white", linewidth=.65, zorder=4,
            ))
        if layer < len(layer_y) - 1:
            axis.add_patch(FancyArrowPatch(
                (x + .204, y0 - .010), (x + .204, layer_y[layer + 1] + .010),
                arrowstyle="-|>", mutation_scale=9, linewidth=1.0,
                color="#be123c",
            ))
    axis.text(x + .017, .625, r"$z^0$", ha="center", va="center",
              fontsize=9.3, color="#9f1239", weight="bold")
    axis.text(x + .017, .535, r"$z^1$", ha="center", va="center",
              fontsize=9.3, color="#9f1239", weight="bold")
    axis.text(x + .017, .445, r"$z^L$", ha="center", va="center",
              fontsize=9.3, color="#9f1239", weight="bold")
    axis.text(x + .112, .715, r"$C_\phi(x_i,v_i)$", ha="center", va="center",
              fontsize=9.5, color="#9f1239", weight="bold")
    axis.text(x + .112, .365, r"$P\in\{G,I\}$; relative geometry", ha="center",
              va="center", fontsize=8.2, color="#9f1239")

    # 4: the centered code decodes a correction on disjoint owned cores.
    x = panel_x[3]
    xx = np.linspace(0, 1, 90)
    yy = np.linspace(0, 1, 65)
    gx, gy = np.meshgrid(xx, yy)
    field = (
        .75 * np.sin(2.2 * np.pi * gx) * np.exp(-1.7 * gy)
        + .32 * np.cos(2.4 * np.pi * gy + 1.1 * gx)
    )
    extent = (x + .030, x + .193, .385, .695)
    axis.imshow(field, extent=extent, origin="lower", cmap="RdBu_r",
                interpolation="bilinear", aspect="auto", alpha=.92,
                rasterized=True, zorder=1)
    body = np.array([
        [x + .072, .525], [x + .111, .548], [x + .158, .536],
        [x + .111, .514],
    ])
    axis.add_patch(Polygon(body, closed=True, facecolor="white",
                           edgecolor="#0f172a", linewidth=1.0, zorder=3))
    for dx in (.084, .138):
        axis.plot([x + dx, x + dx], [.385, .695], color="white",
                  linewidth=.65, alpha=.38, zorder=2)
    axis.text(
        x + .111, .752,
        r"$a_i=H_\phi(z_i^0,z_{G,i}^L)-H_\phi(z_i^0,z_{I,i}^L)$",
        ha="center", va="center", fontsize=7.8, color="#166534",
        weight="bold",
    )
    axis.text(
        x + .111, .716, r"$\delta_i=\mathcal{L}_\phi(x_i,v_i,a_i)$",
        ha="center", va="center", fontsize=8.3, color="#166534",
        weight="bold",
    )
    axis.text(x + .111, .345, "one owner per output cell", ha="center",
              va="center", fontsize=8.2, color="#166534")

    axis.text(.5, .955, "pretrain local physics once  +  learn how local fields compose",
              ha="center", va="center", fontsize=12.0, weight="bold",
              color="#0f172a")
    # The ordering, rather than the use of patches, distinguishes the models.
    axis.text(.020, .105, "Feature-first", ha="left", va="center", fontsize=9,
              weight="bold", color="#334155")
    axis.text(.105, .105,
              r"$\mathrm{anchors/slices/patches}\rightarrow t_i"
              r"\rightarrow\mathrm{mix}"
              r"\rightarrow\widehat u$",
              ha="left", va="center", fontsize=9.5, color="#475569")
    axis.text(.515, .105, "LatentDDM", ha="left", va="center", fontsize=9,
              weight="bold", color="#166534")
    axis.text(.600, .105,
              r"$x_i\rightarrow v_i\in\mathcal{M}_i\rightarrow z_i^0"
              r"\rightarrow z_i^L\rightarrow\widehat u$",
              ha="left", va="center", fontsize=9.5, color="#15803d")
    axis.text(.972, .105,
              r"freeze $\mathcal{F}_\theta$; learn $C_\phi,\tau_\phi,H_\phi,\mathcal{L}_\phi$",
              ha="right", va="center", fontsize=8.5,
              color="#166534")
    save_figure(figure, "method_schematic")


def huang_comparison() -> None:
    figure, axis = plt.subplots(figsize=(13.4, 4.7))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    def box(x: float, y: float, title: str, subtitle: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y), 0.17, 0.18, boxstyle="round,pad=0.010,rounding_size=0.014",
            facecolor=color, edgecolor="#475569", linewidth=1.2,
        )
        axis.add_patch(patch)
        axis.text(x + 0.085, y + 0.120, title, ha="center", va="center", fontsize=10, weight="bold")
        axis.text(x + 0.085, y + 0.055, subtitle, ha="center", va="center", fontsize=8.4)

    axis.text(0.015, 0.79, "Huang et al.: Schwarz Neural Inference", fontsize=12, weight="bold", color="#9a3412")
    box(0.04, 0.52, "Mesh partition", "METIS + overlap", "#ffedd5")
    box(0.29, 0.52, "Local BVP", "artificial BC from $u^k$", "#ffedd5")
    box(0.54, 0.52, "Additive update", "$u^{k+1}=u^k+\\tau\\sum R_i^T d_i$", "#ffedd5")
    box(0.79, 0.52, "Repeat", "until convergence", "#fee2e2")
    for left in (0.21, 0.46, 0.71):
        axis.add_patch(FancyArrowPatch((left, 0.61), (left + 0.08, 0.61), arrowstyle="-|>", mutation_scale=14,
                                       linewidth=1.4, color="#9a3412"))
    axis.add_patch(FancyArrowPatch((0.875, 0.52), (0.375, 0.49), connectionstyle="arc3,rad=-0.28",
                                   arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color="#9a3412"))
    axis.text(0.50, 0.445, "contractive theory: self-adjoint coercive elliptic PDEs", ha="center",
              fontsize=9.4, color="#9a3412")

    axis.text(0.015, 0.34, "Target: one-pass neural two-level composition", fontsize=12, weight="bold", color="#075985")
    box(0.04, 0.07, "Canonical tiles", "2D cells / 3D cubes", "#e0f2fe")
    box(0.29, 0.07, "Local endpoint", "state + geometry", "#dbeafe")
    box(0.54, 0.07, "Coarse solve", "Galerkin / Schur", "#fce7f3")
    box(0.79, 0.07, "Decode + cores", "one endpoint / time step", "#dcfce7")
    for left in (0.21, 0.46, 0.71):
        axis.add_patch(FancyArrowPatch((left, 0.16), (left + 0.08, 0.16), arrowstyle="-|>", mutation_scale=14,
                                       linewidth=1.4, color="#075985"))
    axis.text(0.50, 0.015, "local networks amortize subdomain physics; the reduced solve enforces global consistency",
              ha="center", fontsize=9.4, color="#075985")
    save_figure(figure, "huang_comparison")


def darcy_curves() -> None:
    sweeps = np.arange(1, 17)
    root = ROOT / "codiffuse_Schwarz_Darcy" / "outputs"
    seeds = (20260805, 20260806, 20260807)
    domains = ("A", "B", "C", "D")
    records: dict[tuple[int, str], dict[str, dict[str, float]]] = {}
    for seed in seeds:
        for domain in domains:
            source = (
                root / f"full_domain_local_pou_seed{seed}" /
                f"domain_{domain}_composition" / "summary.json"
            )
            records[(seed, domain)] = json.loads(source.read_text())["summary"]

    def curve(metric: str, *, domain: str | None = None) -> np.ndarray:
        selected_domains = (domain,) if domain is not None else domains
        return np.asarray([
            np.mean([
                records[(seed, selected)][f"pou/sweep_{sweep:03d}"][metric]
                for seed in seeds for selected in selected_domains
            ]) for sweep in sweeps
        ])

    figure, axes = plt.subplots(1, 3, figsize=(13.8, 3.7), constrained_layout=True)
    palette = {"A": "#2563eb", "B": "#dc2626", "C": "#7c3aed", "D": "#059669"}
    for domain in domains:
        axes[0].plot(sweeps, curve("relative_l2", domain=domain), marker="o", markersize=3.0,
                     linewidth=1.7, label=f"domain {domain}", color=palette[domain])
    axes[0].set_title("Frozen Local/Core by topology")
    axes[0].set_ylabel("relative $L_2$")
    for metric, label, color, marker in (
        ("relative_l2", "field $L_2$", "#2563eb", "o"),
        ("energy_relative_error", "energy", "#059669", "s"),
        ("residual_rms", "FV residual", "#7c3aed", "^"),
    ):
        axes[1].plot(sweeps, curve(metric), marker=marker, markersize=3.2,
                     linewidth=1.8, label=label, color=color)
    axes[1].set_title("Equal-domain physical diagnostics")
    axes[1].set_ylabel("error")
    for axis in axes[:2]:
        axis.set_xlabel("local-composition sweep")
        axis.set_xticks((1, 4, 8, 12, 16))
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8.0)
    axes[1].legend(frameon=False, fontsize=8.0)

    pou_error = curve("relative_l2")
    pou_value = curve("value_jump")
    pou_flux = curve("flux_jump")
    for values, label, color, marker in (
        (pou_error / pou_error[0], "field error", "#2563eb", "o"),
        (pou_value / pou_value[0], "value jump", "#f59e0b", "s"),
        (pou_flux / pou_flux[0], "flux jump", "#ef4444", "^"),
    ):
        axes[2].plot(sweeps, values, marker=marker, markersize=3.5, linewidth=1.8, label=label, color=color)
    axes[2].axhline(1.0, color="#64748b", linestyle="--", linewidth=1.0)
    axes[2].set_title("Interface agreement is not PDE convergence")
    axes[2].set_xlabel("local-composition sweep")
    axes[2].set_ylabel("quantity / sweep-1 value")
    axes[2].set_xticks((1, 4, 8, 12, 16))
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8.5)
    save_figure(figure, "darcy_three_seed_curves")


def darcy_modalities() -> None:
    source = ROOT / "codiffuse_Schwarz_Darcy" / "data" / "darcy_complex_v1.h5"
    names = {
        "A": "concave",
        "B": "two holes",
        "C": "cornered hole",
        "D": "long graph / four holes",
    }
    keys = ("mask", "logk", "source", "solution")
    titles = ("active domain", r"$\log_{10}\kappa$", "source $f$", "FV solution $u$")
    arrays: dict[str, list[np.ndarray]] = {}
    with h5py.File(source, "r") as handle:
        for domain in names:
            group = handle[f"domains/{domain}"]
            arrays[domain] = [np.asarray(group[key][0, 0], dtype=np.float32) for key in keys]
    ranges = (
        (0.0, 1.0),
        (min(float(arrays[d][1].min()) for d in names), max(float(arrays[d][1].max()) for d in names)),
        (0.0, max(float(arrays[d][2].max()) for d in names)),
        (0.0, max(float(arrays[d][3].max()) for d in names)),
    )
    figure, axes = plt.subplots(4, 4, figsize=(8.6, 8.2), constrained_layout=True)
    images = [None] * 4
    for row, (domain, label) in enumerate(names.items()):
        mask = arrays[domain][0] > 0.5
        for column, field in enumerate(arrays[domain]):
            shown = field if column == 0 else np.where(mask, field, np.nan)
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad("#f1f5f9")
            images[column] = axes[row, column].imshow(
                shown, origin="lower", cmap=cmap, vmin=ranges[column][0], vmax=ranges[column][1]
            )
            if row == 0:
                axes[row, column].set_title(titles[column], fontsize=10)
            if column == 0:
                axes[row, column].set_ylabel(f"{domain}: {label}", fontsize=9.5)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    for column, image in enumerate(images):
        assert image is not None
        figure.colorbar(image, ax=axes[:, column], location="bottom", shrink=0.72, pad=0.02)
    save_figure(figure, "darcy_geometry_modalities")


def airfoil_focus_crop() -> None:
    source = ROOT / "codiffuse_Schwarz_Airfoil" / "outputs" / "global100_v3_finest" / "head_to_head" / "vorticity_fields.png"
    image = Image.open(source)
    # Focus the publication figure on reference, local composer, and the
    # strongest attention baseline for the first three displayed trajectories.
    cropped = image.crop((0, 0, int(image.width * 3.02 / 7.0), int(image.height * 3.12 / 6.0)))
    pixels = np.asarray(cropped).copy()
    # The source has a page-wide title beginning just beyond the third model
    # column. Remove the clipped title fragment while preserving all three
    # column labels and fields.
    pixels[:90, int(pixels.shape[1] * 0.84):] = 255
    cropped = Image.fromarray(pixels)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cropped.save(OUTPUT / "airfoil_vorticity_focus.png")


def structured_correction_results() -> None:
    """Show the accepted and rejected structured corrections without hiding gates."""
    darcy = json.loads((
        ROOT / "codiffuse_Schwarz_Darcy" / "outputs" / "darcy_two_level_galerkin_multiseed.json"
    ).read_text())
    airfoil = json.loads((
        ROOT / "codiffuse_Schwarz_Airfoil" / "outputs" /
        "two_level_pressure_projection_unified_v4" / "validation_h20.json"
    ).read_text())
    car = json.loads((
        ROOT / "codiffuse_Schwarz_Car" / "outputs" / "interface_schur_v1" / "test_metrics.json"
    ).read_text())

    darcy_metrics = ("field $L_2$", "energy", "residual")
    darcy_base = darcy["equal_domain"]["local_pou"]
    darcy_alt = darcy["equal_domain"]["galerkin_linear"]
    darcy_values = 100.0 * (
        np.asarray([darcy_alt["relative_l2"], darcy_alt["energy_relative_error"], darcy_alt["residual_rms"]]) /
        np.asarray([darcy_base["relative_l2"], darcy_base["energy_relative_error"], darcy_base["residual_rms"]]) - 1.0
    )

    airfoil_metrics = ("step-20\nvorticity", "coarse\ndivergence", "fine\ndivergence", "drag", "lift")
    airfoil_values = 100.0 * np.asarray([
        airfoil["results"]["pressure_projection_1"]["relative_to_local"][key]
        for key in (
            "step20_amr_vorticity_l2_mean", "step20_coarse_divergence_rms_l2_mean",
            "step20_fine_divergence_rms_l2_mean", "c_drag_relative_l2", "c_lift_relative_l2",
        )
    ])

    car_metrics = ("native\n$|U|$", "lattice\n$|U|$", "near body", "wake", "far field", "divergence")
    car_base = car["mean"]["local_pou"]
    car_alt = car["mean"]["interface_r0.01_s1"]
    car_values = 100.0 * (
        np.asarray([car_alt[key] for key in (
            "native_magnitude_rel_l2", "magnitude_rel_l2", "near_body_magnitude_rel_l2",
            "wake_magnitude_rel_l2", "far_field_magnitude_rel_l2", "divergence_rms",
        )]) /
        np.asarray([car_base[key] for key in (
            "native_magnitude_rel_l2", "magnitude_rel_l2", "near_body_magnitude_rel_l2",
            "wake_magnitude_rel_l2", "far_field_magnitude_rel_l2", "divergence_rms",
        )]) - 1.0
    )

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), gridspec_kw={"width_ratios": [0.75, 1.2, 1.45]})
    panels = (
        (darcy_metrics, darcy_values, "Darcy: linear Galerkin\nheld-out, accepted"),
        (airfoil_metrics, airfoil_values, "Airfoil: compatible projection\nvalidation, rejected by force gate"),
        (car_metrics, car_values, "DrivAerNet++: interface Schur\n24 held-out cars, accepted"),
    )
    for axis, (labels, values, title) in zip(axes, panels):
        colors = ["#15803d" if value <= 0 else "#dc2626" for value in values]
        bars = axis.bar(np.arange(len(values)), values, color=colors, alpha=0.88)
        axis.axhline(0.0, color="#334155", linewidth=0.9)
        axis.set_xticks(np.arange(len(values)), labels, fontsize=8.3)
        axis.set_title(title, fontsize=10.5, weight="bold")
        axis.grid(axis="y", alpha=0.22)
        for bar, value in zip(bars, values):
            offset = -1.0 if value < -8 else (-0.14 if value < 0 else 0.10)
            va = "top" if value < 0 else "bottom"
            axis.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.2f}%",
                      ha="center", va=va, fontsize=8.0)
    axes[0].set_ylabel("relative change from frozen Local/Core (lower is better)")
    axes[0].set_ylim(min(-70, float(darcy_values.min()) - 5), 5)
    axes[1].set_ylim(-5, 3)
    axes[2].set_ylim(-2.0, 0.25)
    figure.suptitle("One contract, case-specific coarse physics, protected acceptance gates", fontsize=12, weight="bold")
    figure.tight_layout()
    save_figure(figure, "structured_correction_results")


def main() -> None:
    method_schematic()
    huang_comparison()
    darcy_curves()
    darcy_modalities()
    airfoil_focus_crop()
    structured_correction_results()
    print(f"wrote unified report figures to {OUTPUT}")


if __name__ == "__main__":
    main()
