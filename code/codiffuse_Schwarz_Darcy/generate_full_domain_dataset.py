#!/usr/bin/env python3
"""Generate the shared 96x96 full-domain Darcy fine-tuning set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .full_domain_dataset import (
    DarcyFullDomainConfig,
    generate_full_domain_hdf5,
    inspect_full_domain_dataset,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=root / "data" / "darcy_full_domain_face_96_v2.h5",
    )
    parser.add_argument("--grid", type=int, default=96)
    parser.add_argument("--train-shapes", type=int, default=4000)
    parser.add_argument("--train-conditions", type=int, default=1)
    parser.add_argument("--val-shapes", type=int, default=250)
    parser.add_argument("--val-conditions", type=int, default=1)
    parser.add_argument("--coefficient-coarse-grid", type=int, default=7)
    parser.add_argument("--kappa-floor", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = (
        DarcyFullDomainConfig.smoke(args.seed)
        if args.smoke else
        DarcyFullDomainConfig(
            grid=args.grid,
            train_shapes=args.train_shapes,
            train_conditions=args.train_conditions,
            val_shapes=args.val_shapes,
            val_conditions=args.val_conditions,
            coefficient_coarse_grid=args.coefficient_coarse_grid,
            kappa_floor=args.kappa_floor,
            seed=args.seed,
        )
    )
    started = time.perf_counter()
    path = generate_full_domain_hdf5(args.output, config)
    summary = inspect_full_domain_dataset(path)
    summary["path"] = str(path.resolve())
    summary["bytes"] = path.stat().st_size
    summary["generation_seconds"] = time.perf_counter() - started
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
