#!/usr/bin/env python3
"""Command-line entry point for the complex-boundary Darcy dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from codiffuse_Schwarz_Darcy.dataset import (
    DarcyDatasetConfig,
    generate_hdf5,
    inspect_dataset,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=root / "data" / "darcy_complex_face_v2.h5",
    )
    parser.add_argument("--train-shapes", type=int, default=4000)
    parser.add_argument("--train-conditions", type=int, default=10)
    parser.add_argument("--val-shapes", type=int, default=250)
    parser.add_argument("--val-conditions", type=int, default=10)
    parser.add_argument("--test-shapes", type=int, default=100)
    parser.add_argument("--test-conditions", type=int, default=1)
    parser.add_argument("--domain-conditions", type=int, default=100)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--eval-grid", type=int, default=128)
    parser.add_argument("--large-eval-grid", type=int, default=192)
    parser.add_argument("--kappa-floor", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--training-only", action="store_true",
        help="write only train/validation local BVPs; never construct test or A-D",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        config = DarcyDatasetConfig.smoke(args.seed)
    else:
        config = DarcyDatasetConfig(
            core=args.core,
            halo=args.halo,
            train_shapes=args.train_shapes,
            train_conditions=args.train_conditions,
            val_shapes=args.val_shapes,
            val_conditions=args.val_conditions,
            test_shapes=args.test_shapes,
            test_conditions=args.test_conditions,
            domain_conditions=args.domain_conditions,
            eval_grid=args.eval_grid,
            large_eval_grid=args.large_eval_grid,
            kappa_floor=args.kappa_floor,
            seed=args.seed,
        )
    start = time.perf_counter()
    path = generate_hdf5(
        args.output, config, training_only=args.training_only,
    )
    summary = inspect_dataset(path)
    summary["path"] = str(path.resolve())
    summary["bytes"] = path.stat().st_size
    summary["generation_seconds"] = time.perf_counter() - start
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
