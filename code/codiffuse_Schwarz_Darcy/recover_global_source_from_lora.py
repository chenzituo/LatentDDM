"""Recover an exact frozen global source from audited low-rank checkpoints.

Low-rank training freezes every source parameter and stores the wrapped state
separately from the merged adapted model.  This utility removes the wrappers,
checks that two or more independent LoRA runs contain bitwise-identical base
weights, and writes a minimal ordinary checkpoint suitable for a fresh,
full-model adaptation audit.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unwrap_base_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    if payload.get("adaptation_mode") != "low_rank":
        raise ValueError("source recovery requires a low-rank checkpoint")
    wrapped = payload.get("adaptation_model")
    merged = payload.get("model")
    if not isinstance(wrapped, dict) or not isinstance(merged, dict):
        raise ValueError("low-rank checkpoint is missing wrapped or merged state")
    result: dict[str, torch.Tensor] = {}
    for name, value in wrapped.items():
        if any(part.startswith("lora_") for part in name.split(".")):
            continue
        ordinary = name.replace(".base.", ".")
        if ordinary in result:
            if not torch.equal(result[ordinary], value):
                raise ValueError(f"conflicting recovered tensor: {ordinary}")
            continue
        result[ordinary] = value.detach().cpu().clone()
    if set(result) != set(merged):
        missing = sorted(set(merged) - set(result))
        unexpected = sorted(set(result) - set(merged))
        raise ValueError(
            f"recovered state keys differ: missing={missing}, unexpected={unexpected}",
        )
    return result


def main() -> None:
    args = parse_args()
    if len(args.checkpoints) < 2:
        raise ValueError("at least two independent LoRA checkpoints are required")
    if args.output.exists():
        raise FileExistsError(args.output)
    payloads = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.checkpoints
    ]
    reference = payloads[0]
    states = [unwrap_base_state(payload) for payload in payloads]
    for index, (payload, state) in enumerate(zip(payloads[1:], states[1:]), 1):
        for key in states[0]:
            if not torch.equal(states[0][key], state[key]):
                raise ValueError(
                    f"LoRA source bases differ at checkpoint {index}: {key}",
                )
        for key in ("contract", "statistics", "model_spec"):
            if payload.get(key) != reference.get(key):
                raise ValueError(f"LoRA checkpoint metadata differs: {key}")
    output = {
        "contract": reference["contract"],
        "epoch": -1,
        "model": states[0],
        "statistics": reference["statistics"],
        "model_spec": reference["model_spec"],
        "training_stage": "recovered_frozen_source",
        "recovery": {
            "contract": "bitwise_consensus_lora_frozen_base_v1",
            "checkpoint_count": len(args.checkpoints),
            "checkpoints": [str(path.resolve()) for path in args.checkpoints],
            "checkpoint_sha256": [sha256(path) for path in args.checkpoints],
            "bitwise_identical_across_runs": True,
            "adaptation_parameters_excluded": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print({
        "output": str(args.output.resolve()),
        "tensors": len(states[0]),
        "parameters": sum(value.numel() for value in states[0].values()),
        "source_runs": len(args.checkpoints),
    })


if __name__ == "__main__":
    main()
