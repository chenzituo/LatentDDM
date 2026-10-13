#!/usr/bin/env python3
"""Generate one fixed-resolution Darcy domain-extension scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .domain_extension_dataset import (
    DarcyDomainExtensionConfig,
    generate_domain_extension_hdf5,
    inspect_domain_extension_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, choices=(4, 8, 16), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-shapes", type=int, default=64)
    parser.add_argument("--train-conditions", type=int, default=1)
    parser.add_argument("--val-shapes", type=int, default=64)
    parser.add_argument("--val-conditions", type=int, default=1)
    parser.add_argument("--test-shapes", type=int, default=64)
    parser.add_argument("--test-conditions", type=int, default=1)
    parser.add_argument("--coefficient-coarse-grid", type=int)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = (
        DarcyDomainExtensionConfig.smoke(args.blocks, args.seed)
        if args.smoke else
        DarcyDomainExtensionConfig(
            blocks=args.blocks,
            train_shapes=args.train_shapes,
            train_conditions=args.train_conditions,
            val_shapes=args.val_shapes,
            val_conditions=args.val_conditions,
            test_shapes=args.test_shapes,
            test_conditions=args.test_conditions,
            coefficient_coarse_grid=args.coefficient_coarse_grid,
            seed=args.seed,
        )
    )
    started = time.perf_counter()
    path = generate_domain_extension_hdf5(args.output, config)
    result = inspect_domain_extension_dataset(path)
    result["path"] = str(path.resolve())
    result["bytes"] = path.stat().st_size
    result["generation_seconds"] = time.perf_counter() - started
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
