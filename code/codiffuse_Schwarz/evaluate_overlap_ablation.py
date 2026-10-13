from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "outputs/mplconfig")

import matplotlib.pyplot as plt
import torch

from codiffuse_Schwarz.core import (
    CoDiffuseSchwarz,
    ToyConfig,
    boundary_error_ratio,
    relative_l2,
    relative_residual,
    sample_flow,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate halo width and assembly for the pure flow model.")
    parser.add_argument("--checkpoint", type=Path, default=root / "outputs/pure_schwarz_v1/checkpoint.pt")
    parser.add_argument("--data", type=Path, default=root / "data/screened_poisson_pure_v1.pt")
    parser.add_argument("--out-dir", type=Path, default=root / "outputs/overlap_ablation_v1")
    parser.add_argument("--halos", type=int, nargs="+", default=(0, 1, 2, 4, 8))
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=9741)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = saved["args"]
    model = CoDiffuseSchwarz(
        width=int(train_args["width"]),
        depth=int(train_args["depth"]),
        m_dim=int(train_args.get("m_dim", 16)),
        heads=int(train_args.get("heads", 4)),
        use_eos=bool(train_args.get("use_eos", False)),
        use_attention=bool(train_args.get("use_attention", False)),
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()

    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    test = {key: value.to(device) for key, value in payload["test"].items()}
    base_config = ToyConfig(**saved["config"])
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(test["field"].shape, generator=generator, device=device) * test["mask"]
    rows = []
    trajectories = {}

    for halo in args.halos:
        for assembly in ("ras", "average"):
            if halo == 0 and assembly == "average":
                continue
            config = replace(base_config, halo=halo)
            name = f"halo{halo}_{assembly}"
            print(f"sampling {name}", flush=True)
            result = sample_flow(
                model,
                test["mask"],
                test["source"],
                test["reaction"],
                test["field"],
                config,
                mode="schwarz",
                inference_steps=args.steps,
                initial_noise=noise.clone(),
                assembly=assembly,
            )
            trajectories[name] = result.trajectory
            rows.append(
                {
                    "name": name,
                    "halo": halo,
                    "assembly": assembly,
                    "relative_l2": float(relative_l2(result.field, test["field"], test["mask"]).mean()),
                    "relative_residual": float(
                        relative_residual(result.field, test["source"], test["mask"], test["reaction"]).mean()
                    ),
                    "interface_error_ratio": float(
                        boundary_error_ratio(result.field, test["field"], test["mask"], config.patch_size).mean()
                    ),
                }
            )

    (args.out_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.out_dir / "trajectories.json").write_text(json.dumps(trajectories, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for assembly, marker in (("ras", "o"), ("average", "s")):
        selected = [row for row in rows if row["assembly"] == assembly]
        for axis, key, label in zip(
            axes,
            ("relative_l2", "relative_residual", "interface_error_ratio"),
            ("relative L2", "relative PDE residual", "interface-error ratio"),
        ):
            axis.plot(
                [row["halo"] for row in selected],
                [row[key] for row in selected],
                marker=marker,
                label=assembly,
            )
            axis.set_xlabel("halo width")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "halo_assembly_ablation.png", dpi=180)
    plt.close(fig)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
