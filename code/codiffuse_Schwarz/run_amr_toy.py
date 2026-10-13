from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "outputs/mplconfig")

import matplotlib.pyplot as plt
import numpy as np
import torch

from codiffuse_Schwarz.amr_flow import (
    AMRFlowModel,
    SampleAMRLayout,
    TwoLevelAMRConfig,
    amr_flow_loss,
    build_amr_layouts,
    layout_costs,
    sample_amr_flow,
    sample_coarse_flow,
    sample_uniform_fine_flow,
)
from codiffuse_Schwarz.core import boundary_error_ratio, relative_l2, relative_residual


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train a true two-level Cartesian AMR flow-matching toy.")
    parser.add_argument("--data", type=Path, default=root / "data/screened_poisson_pure_v1.pt")
    parser.add_argument("--out-dir", type=Path, default=root / "outputs/amr_flow_v1")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--consistency-weight", type=float, default=0.10)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--tag-threshold", type=float, default=0.48)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--viz-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11831)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def batches(split: dict[str, torch.Tensor], layouts: list[SampleAMRLayout], batch_size: int, seed: int, shuffle: bool):
    count = split["field"].shape[0]
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(count, generator=generator) if shuffle else torch.arange(count)
    for start in range(0, count, batch_size):
        indices = order[start : start + batch_size]
        yield {key: value[indices] for key, value in split.items()}, [layouts[int(index)] for index in indices]


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def mean(rows: list[dict[str, float]], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


@torch.no_grad()
def validate(
    model: AMRFlowModel,
    split: dict[str, torch.Tensor],
    layouts: list[SampleAMRLayout],
    config: TwoLevelAMRConfig,
    batch_size: int,
    device: torch.device,
    seed: int,
    consistency_weight: float,
) -> float:
    torch.manual_seed(seed)
    values = []
    for batch, batch_layouts in batches(split, layouts, batch_size, seed, False):
        batch = move(batch, device)
        loss, _ = amr_flow_loss(
            model,
            batch["field"],
            batch["mask"],
            batch["source"],
            batch["reaction"],
            batch_layouts,
            config,
            consistency_weight=consistency_weight,
        )
        values.append(float(loss))
    return float(np.mean(values))


def train(
    model: AMRFlowModel,
    data: dict,
    layouts: dict[str, list[SampleAMRLayout]],
    config: TwoLevelAMRConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.05)
    history = []
    best = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_rows = []
        started = time.perf_counter()
        for batch_index, (batch, batch_layouts) in enumerate(
            batches(data["train"], layouts["train"], args.batch_size, args.seed + epoch, True)
        ):
            torch.manual_seed(args.seed + 1000 * epoch + batch_index)
            batch = move(batch, device)
            loss, metrics = amr_flow_loss(
                model,
                batch["field"],
                batch["mask"],
                batch["source"],
                batch["reaction"],
                batch_layouts,
                config,
                consistency_weight=args.consistency_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_rows.append(metrics)
        scheduler.step()
        model.eval()
        val_loss = validate(
            model,
            data["val"],
            layouts["val"],
            config,
            args.batch_size,
            device,
            args.seed + 50_000 + epoch,
            args.consistency_weight,
        )
        row = {
            "epoch": float(epoch),
            "train_loss": mean(epoch_rows, "loss"),
            "coarse_velocity_loss": mean(epoch_rows, "coarse_velocity_loss"),
            "fine_velocity_loss": mean(epoch_rows, "fine_velocity_loss"),
            "consistency_loss": mean(epoch_rows, "consistency_loss"),
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{args.epochs} train={row['train_loss']:.5f} "
            f"coarse={row['coarse_velocity_loss']:.5f} fine={row['fine_velocity_loss']:.5f} "
            f"sync={row['consistency_loss']:.5f} val={val_loss:.5f} seconds={row['seconds']:.2f}",
            flush=True,
        )
        if val_loss < best:
            best = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def final_metrics(field: torch.Tensor, test: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        "relative_l2_mean": float(relative_l2(field, test["field"], test["mask"]).mean()),
        "relative_l2_std": float(relative_l2(field, test["field"], test["mask"]).std()),
        "relative_residual_mean": float(
            relative_residual(field, test["source"], test["mask"], test["reaction"]).mean()
        ),
        "interface_error_ratio_mean": float(
            boundary_error_ratio(field, test["field"], test["mask"], patch_size=16).mean()
        ),
    }


@torch.no_grad()
def evaluate(
    model: AMRFlowModel,
    test_cpu: dict[str, torch.Tensor],
    layouts: list[SampleAMRLayout],
    config: TwoLevelAMRConfig,
    args: argparse.Namespace,
    device: torch.device,
):
    test = move(test_cpu, device)
    generator = torch.Generator(device=device).manual_seed(args.seed + 90_000)
    noise = torch.randn(test["field"].shape, generator=generator, device=device) * test["mask"]
    print("sampling coarse-only flow", flush=True)
    coarse = sample_coarse_flow(
        model, test["mask"], test["source"], test["reaction"], test["field"], noise, config, args.steps
    )
    print("sampling adaptive AMR flow", flush=True)
    amr = sample_amr_flow(
        model,
        test["mask"],
        test["source"],
        test["reaction"],
        test["field"],
        noise,
        layouts,
        config,
        args.steps,
    )
    print("sampling uniform fine flow", flush=True)
    uniform = sample_uniform_fine_flow(
        model, test["mask"], test["source"], test["reaction"], test["field"], noise, config, args.steps
    )
    predictions = {"coarse": coarse.field.cpu(), "amr": amr.field.cpu(), "uniform": uniform.field.cpu()}
    trajectories = {"coarse": coarse.trajectory, "amr": amr.trajectory, "uniform": uniform.trajectory}
    metrics = {
        "coarse": final_metrics(coarse.field, test),
        "amr": final_metrics(amr.field, test),
        "uniform": final_metrics(uniform.field, test),
    }
    return metrics, predictions, trajectories


def level_map(layout: SampleAMRLayout, shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.float32)
    for y0, y1, x0, x1 in layout.fine_boxes:
        result[y0:y1, x0:x1] = 1.0
    return result


def masked(field: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    array = field.squeeze().numpy().copy()
    array[mask.squeeze().numpy() < 0.5] = np.nan
    return array


def plot_fields(
    test: dict[str, torch.Tensor],
    layouts: list[SampleAMRLayout],
    predictions: dict[str, torch.Tensor],
    count: int,
    out_path: Path,
) -> None:
    count = min(count, test["field"].shape[0])
    columns = ("AMR levels", "source", "reference", "coarse", "adaptive AMR", "uniform fine", "AMR error")
    fig, axes = plt.subplots(count, len(columns), figsize=(19, 2.9 * count), squeeze=False)
    for row in range(count):
        mask = test["mask"][row]
        reference = test["field"][row]
        values = [
            level_map(layouts[row], reference.shape[-2:]),
            masked(test["source"][row], mask),
            masked(reference, mask),
            masked(predictions["coarse"][row], mask),
            masked(predictions["amr"][row], mask),
            masked(predictions["uniform"][row], mask),
            masked((predictions["amr"][row] - reference).abs(), mask),
        ]
        for column, (title, value) in enumerate(zip(columns, values)):
            cmap = "viridis" if title == "AMR levels" else ("magma" if "error" in title else "coolwarm")
            image = axes[row, column].imshow(value, origin="lower", cmap=cmap)
            axes[row, column].set_title(title if row == 0 else f"sample {row}")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if title in {"AMR levels", "adaptive AMR"}:
                for y0, y1, x0, x1 in layouts[row].fine_boxes:
                    axes[row, column].add_patch(
                        plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec="white", lw=0.7)
                    )
            fig.colorbar(image, ax=axes[row, column], fraction=0.045, pad=0.02)
    fig.suptitle("Two-level Cartesian AMR flow matching", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_trajectories(trajectories: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for axis, key, label in zip(axes, ("relative_l2", "residual"), ("relative L2", "relative PDE residual")):
        for name, rows in trajectories.items():
            axis.plot([row["step"] for row in rows], [row[key] for row in rows], label=name, lw=1.7)
        axis.set_yscale("log")
        axis.set_xlabel("reverse-flow step")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_training(history: list[dict[str, float]], out_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train")
    axis.plot([row["epoch"] for row in history], [row["val_loss"] for row in history], label="validation")
    axis.set_yscale("log")
    axis.set_xlabel("epoch")
    axis.set_ylabel("multilevel flow loss")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.batch_size = 8
        args.width = 16
        args.depth = 2
        args.steps = 3
        if args.out_dir == Path(__file__).resolve().parent / "outputs/amr_flow_v1":
            args.out_dir = args.out_dir.parent / "amr_smoke"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    if args.smoke:
        data = {
            "train": {key: value[:32] for key, value in data["train"].items()},
            "val": {key: value[:8] for key, value in data["val"].items()},
            "test": {key: value[:8] for key, value in data["test"].items()},
        }
    config = TwoLevelAMRConfig(tag_threshold=args.tag_threshold, halo=args.halo)
    print("building block-structured layouts", flush=True)
    layouts = {
        split: build_amr_layouts(values["mask"], values["source"], values["reaction"], config)
        for split, values in data.items()
        if split in {"train", "val", "test"}
    }
    costs = layout_costs(layouts["test"], config)
    print(json.dumps(costs, indent=2), flush=True)
    model = AMRFlowModel(args.width, args.depth).to(device)
    print(f"device={device} parameters={sum(value.numel() for value in model.parameters()):,}", flush=True)
    history = train(model, data, layouts, config, args, device)
    model.eval()
    metrics, predictions, trajectories = evaluate(model, data["test"], layouts["test"], config, args, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "device": str(device),
        "config": config.__dict__,
        "training": {
            "epochs": args.epochs,
            "parameters": sum(value.numel() for value in model.parameters()),
            "best_val_loss": min(row["val_loss"] for row in history),
        },
        "layout_costs": costs,
        "metrics": metrics,
    }
    torch.save({"model": model.state_dict(), "args": vars(args), "config": config.__dict__}, args.out_dir / "checkpoint.pt")
    torch.save({"predictions": predictions, "test": data["test"]}, args.out_dir / "predictions.pt")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "trajectories.json").write_text(json.dumps(trajectories, indent=2), encoding="utf-8")
    write_csv(args.out_dir / "training.csv", history)
    plot_fields(data["test"], layouts["test"], predictions, args.viz_samples, args.out_dir / "field_gallery.png")
    plot_trajectories(trajectories, args.out_dir / "convergence.png")
    plot_training(history, args.out_dir / "training.png")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
