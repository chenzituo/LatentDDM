from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import torch

from codiffuse_Schwarz.core import (
    CoDiffuseSchwarz,
    ToyConfig,
    boundary_error_ratio,
    coordinate_grid,
    generate_dataset,
    predict_patch_velocity,
    relative_l2,
    relative_residual,
    sample_flow,
    save_dataset,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train and evaluate the CoDiffuse-Schwarz flow-matching toy.")
    parser.add_argument("--data", type=Path, default=root / "data" / "screened_poisson_32.pt")
    parser.add_argument("--out-dir", type=Path, default=root / "outputs" / "toy")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--train", type=int, default=320)
    parser.add_argument("--val", type=int, default=64)
    parser.add_argument("--test", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--m-dim", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--use-eos", action="store_true", help="Optional extension; disabled in the core experiment.")
    parser.add_argument("--use-attention", action="store_true", help="Optional extension; requires --use-eos.")
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--endpoint-weight", type=float, default=0.15)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--assembly", choices=("ras", "average"), default="ras")
    parser.add_argument("--viz-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7411)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is visible")
        return torch.device("cuda")
    if name == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def batches(split: dict[str, torch.Tensor], batch_size: int, shuffle: bool, seed: int):
    count = split["field"].shape[0]
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(count, generator=generator) if shuffle else torch.arange(count)
    for start in range(0, count, batch_size):
        indices = order[start : start + batch_size]
        yield {key: value[indices] for key, value in split.items()}


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def flow_loss(
    model: CoDiffuseSchwarz,
    batch: dict[str, torch.Tensor],
    config: ToyConfig,
    endpoint_weight: float,
    assembly: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["field"]
    mask = batch["mask"]
    source = batch["source"]
    reaction = batch["reaction"]
    count = target.shape[0]
    time_value = torch.rand(count, device=target.device).clamp(1.0e-3, 1.0 - 1.0e-3)
    noise = torch.randn_like(target) * mask
    time_image = time_value[:, None, None, None]
    state = ((1.0 - time_image) * target + time_image * noise) * mask
    target_velocity = (noise - target) * mask
    coords = coordinate_grid(config.height, config.width, target.device).expand(count, -1, -1, -1)
    predicted_velocity = predict_patch_velocity(
        model,
        state,
        mask,
        source,
        reaction,
        coords,
        time_value,
        config.patch_size,
        config.halo,
        connected=True,
        assembly=assembly,
    )
    denominator = mask.sum().clamp_min(1.0)
    velocity_loss = ((predicted_velocity - target_velocity).square() * mask).sum() / denominator
    endpoint = (state - time_image * predicted_velocity) * mask
    endpoint_loss = ((endpoint - target).square() * mask).sum() / denominator
    loss = velocity_loss + endpoint_weight * endpoint_loss
    return loss, {
        "loss": float(loss.detach()),
        "velocity_loss": float(velocity_loss.detach()),
        "endpoint_loss": float(endpoint_loss.detach()),
    }


@torch.no_grad()
def validation_loss(
    model: CoDiffuseSchwarz,
    split: dict[str, torch.Tensor],
    config: ToyConfig,
    batch_size: int,
    endpoint_weight: float,
    assembly: str,
    device: torch.device,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    values = []
    for batch in batches(split, batch_size, False, seed):
        loss, _ = flow_loss(model, to_device(batch, device), config, endpoint_weight, assembly)
        values.append(float(loss))
    return float(np.mean(values))


def train_model(
    model: CoDiffuseSchwarz,
    train: dict[str, torch.Tensor],
    val: dict[str, torch.Tensor],
    config: ToyConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        rows = []
        start_time = time.perf_counter()
        for batch_index, batch in enumerate(batches(train, args.batch_size, True, args.seed + epoch)):
            torch.manual_seed(args.seed + epoch * 1000 + batch_index)
            loss, metrics = flow_loss(model, to_device(batch, device), config, args.endpoint_weight, args.assembly)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            rows.append(metrics)
        scheduler.step()
        model.eval()
        val_loss = validation_loss(
            model, val, config, args.batch_size, args.endpoint_weight, args.assembly, device, args.seed + 50_000 + epoch
        )
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean([item["loss"] for item in rows])),
            "train_velocity_loss": float(np.mean([item["velocity_loss"] for item in rows])),
            "train_endpoint_loss": float(np.mean([item["endpoint_loss"] for item in rows])),
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - start_time,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{args.epochs} train={row['train_loss']:.5f} "
            f"val={val_loss:.5f} seconds={row['seconds']:.1f}",
            flush=True,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def evaluate(
    model: CoDiffuseSchwarz,
    test: dict[str, torch.Tensor],
    config: ToyConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict, dict[str, torch.Tensor], dict[str, list[dict[str, float]]]]:
    batch = to_device(test, device)
    generator = torch.Generator(device=device).manual_seed(args.seed + 90_000)
    initial_noise = torch.randn(batch["field"].shape, generator=generator, device=device) * batch["mask"]
    modes = ("global", "independent", "schwarz")
    predictions: dict[str, torch.Tensor] = {}
    trajectories: dict[str, list[dict[str, float]]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for mode in modes:
        print(f"sampling {mode}", flush=True)
        result = sample_flow(
            model,
            batch["mask"],
            batch["source"],
            batch["reaction"],
            batch["field"],
            config,
            mode=mode,
            inference_steps=args.inference_steps,
            initial_noise=initial_noise.clone(),
            assembly=args.assembly,
        )
        predictions[mode] = result.field.cpu()
        trajectories[mode] = result.trajectory
        metrics[mode] = {
            "relative_l2_mean": float(relative_l2(result.field, batch["field"], batch["mask"]).mean()),
            "relative_l2_std": float(relative_l2(result.field, batch["field"], batch["mask"]).std()),
            "relative_residual_mean": float(
                relative_residual(result.field, batch["source"], batch["mask"], batch["reaction"]).mean()
            ),
            "interface_error_ratio_mean": float(
                boundary_error_ratio(result.field, batch["field"], batch["mask"], config.patch_size).mean()
            ),
            "flow_steps": float(args.inference_steps),
        }
    payload = {
        "device": str(device),
        "test_samples": int(test["field"].shape[0]),
        "config": config.__dict__,
        "metrics": metrics,
    }
    return payload, predictions, trajectories


def masked(array: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    values = array.squeeze().numpy().copy()
    values[mask.squeeze().numpy() < 0.5] = np.nan
    return values


def plot_gallery(
    test: dict[str, torch.Tensor],
    predictions: dict[str, torch.Tensor],
    config: ToyConfig,
    count: int,
    out_path: Path,
) -> None:
    count = min(count, test["field"].shape[0])
    columns = ("source", "reference", "independent", "schwarz", "independent error", "schwarz error")
    fig, axes = plt.subplots(count, len(columns), figsize=(17, 3.0 * count), squeeze=False)
    for row in range(count):
        mask = test["mask"][row]
        reference = test["field"][row]
        values = [
            masked(test["source"][row], mask),
            masked(reference, mask),
            masked(predictions["independent"][row], mask),
            masked(predictions["schwarz"][row], mask),
            masked((predictions["independent"][row] - reference).abs(), mask),
            masked((predictions["schwarz"][row] - reference).abs(), mask),
        ]
        for column, (title, value) in enumerate(zip(columns, values)):
            cmap = "magma" if "error" in title else "coolwarm"
            image = axes[row, column].imshow(value, cmap=cmap, origin="lower")
            axes[row, column].set_title(title if row == 0 else f"sample {row}")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if title in {"independent", "schwarz"}:
                for position in range(config.patch_size, config.width, config.patch_size):
                    axes[row, column].axvline(position - 0.5, color="white", lw=0.65, alpha=0.8)
                for position in range(config.patch_size, config.height, config.patch_size):
                    axes[row, column].axhline(position - 0.5, color="white", lw=0.65, alpha=0.8)
            fig.colorbar(image, ax=axes[row, column], fraction=0.045, pad=0.02)
    fig.suptitle("Conditional flow matching with Schwarz-coupled Cartesian patches", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_trajectories(trajectories: dict[str, list[dict[str, float]]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    keys = (("relative_l2", "relative L2"), ("residual", "relative PDE residual"), ("interface_error_ratio", "interface error ratio"))
    for axis, (key, label) in zip(axes, keys):
        for mode, rows in trajectories.items():
            axis.plot([row["step"] for row in rows], [row[key] for row in rows], label=mode, lw=1.7)
        if key != "interface_error_ratio":
            axis.set_yscale("log")
        axis.set_xlabel("reverse-flow step")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_training(history: list[dict[str, float]], out_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train")
    axis.plot([row["epoch"] for row in history], [row["val_loss"] for row in history], label="validation")
    axis.set_yscale("log")
    axis.set_xlabel("epoch")
    axis.set_ylabel("flow-matching loss")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.train, args.val, args.test = 32, 8, 8
        args.epochs, args.batch_size = 1, 8
        args.width, args.depth = 16, 2
        args.inference_steps = 3
        args.out_dir = args.out_dir.parent / "smoke"
        args.data = args.data.parent / "screened_poisson_smoke.pt"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    config = ToyConfig()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.regenerate or not args.data.exists():
        print(f"generating dataset at {args.data}", flush=True)
        save_dataset(args.data, config, args.train, args.val, args.test, args.seed)
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    model = CoDiffuseSchwarz(
        width=args.width,
        depth=args.depth,
        m_dim=args.m_dim,
        heads=args.heads,
        use_eos=args.use_eos,
        use_attention=args.use_attention,
    ).to(device)
    print(f"device={device} parameters={sum(value.numel() for value in model.parameters()):,}", flush=True)
    history = train_model(model, payload["train"], payload["val"], config, args, device)
    torch.save({"model": model.state_dict(), "args": vars(args), "config": config.__dict__}, args.out_dir / "checkpoint.pt")
    summary, predictions, trajectories = evaluate(model.eval(), payload["test"], config, args, device)
    summary["training"] = {
        "epochs": args.epochs,
        "best_val_loss": min(row["val_loss"] for row in history),
        "parameters": sum(value.numel() for value in model.parameters()),
        "use_eos": args.use_eos,
        "use_attention": args.use_attention,
        "assembly": args.assembly,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "trajectories.json").write_text(json.dumps(trajectories, indent=2), encoding="utf-8")
    write_csv(args.out_dir / "training.csv", history)
    torch.save({"predictions": predictions, "test": payload["test"]}, args.out_dir / "predictions.pt")
    plot_training(history, args.out_dir / "training.png")
    plot_gallery(payload["test"], predictions, config, args.viz_samples, args.out_dir / "field_gallery.png")
    plot_trajectories(trajectories, args.out_dir / "convergence.png")
    print(json.dumps(summary["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
