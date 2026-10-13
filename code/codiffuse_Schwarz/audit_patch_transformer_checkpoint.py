"""Audit a frozen-backbone same-cover Patch Transformer checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--communicator-name", default="grid_communicator",
        help="substring required in every tensor added to the baseline",
    )
    return parser.parse_args()


def state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    model = payload.get("model")
    if not isinstance(model, dict) or not model:
        raise ValueError("checkpoint has no nonempty model state")
    return model


def inferred_cover_contract(payload: Mapping[str, Any]) -> Any:
    """Recover a saved cover description from legacy training arguments."""
    explicit = payload.get("cover_contract")
    if explicit is not None:
        return explicit
    saved_args = payload.get("args")
    if not isinstance(saved_args, Mapping):
        return None
    shape = saved_args.get("group_shape")
    if isinstance(shape, (list, tuple)) and shape:
        counts = [int(value) for value in shape]
        return {
            "grid_shape": counts,
            "subdomains": int(torch.tensor(counts).prod().item()),
            "core": int(saved_args["core"]) if "core" in saved_args else None,
            "halo": int(saved_args["halo"]) if "halo" in saved_args else None,
        }
    data_path = saved_args.get("data")
    core = saved_args.get("core")
    if data_path is not None and core is not None and Path(data_path).exists():
        import h5py

        with h5py.File(data_path, "r") as handle:
            grid_size = handle.attrs.get("grid_size")
            if grid_size is None:
                split = "train" if "train" in handle else next(iter(handle))
                grid_shape = tuple(int(value) for value in handle[split]["solution"].shape[-2:])
            else:
                grid_shape = (int(grid_size), int(grid_size))
        counts = [size // int(core) for size in grid_shape]
        if any(count * int(core) != size for count, size in zip(counts, grid_shape)):
            raise ValueError("full-domain shape is not divisible by the saved core")
        return {
            "grid_shape": counts,
            "subdomains": int(torch.tensor(counts).prod().item()),
            "core": int(core),
            "halo": int(saved_args["halo"]) if "halo" in saved_args else None,
        }
    return None


def inferred_local_input_contract(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any],
) -> Any:
    """Expose the inherited endpoint input contract in older checkpoints."""
    explicit = candidate.get("local_input_contract")
    if explicit is not None:
        return explicit
    channels = candidate.get("physical_channels", baseline.get("physical_channels"))
    if channels is not None:
        return {
            "backbone_contract": baseline.get("contract"),
            "physical_channels": list(channels),
        }
    model_spec = baseline.get("model_spec")
    if isinstance(model_spec, Mapping):
        return {
            "architecture": model_spec.get("architecture"),
            "physical_channels": model_spec.get("physical_channels"),
            "history_steps": model_spec.get("history_steps"),
        }
    return None


def main() -> None:
    args = parse_args()
    baseline = torch.load(args.baseline, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    baseline_sha256 = digest(args.baseline)
    before, after = state(baseline), state(candidate)
    missing = sorted(set(before) - set(after))
    extra = sorted(set(after) - set(before))
    changed = sorted(
        name for name in set(before) & set(after)
        if not torch.equal(before[name], after[name])
    )
    wrong_extra = [name for name in extra if args.communicator_name not in name]
    trainable_names = list(candidate.get("trainable_parameter_names", ()))
    wrong_trainable = [
        name for name in trainable_names if args.communicator_name not in name
    ]
    missing_trainable = sorted(set(extra) - set(trainable_names))
    source_hash_matches = (
        candidate.get("source_checkpoint_sha256") == baseline_sha256
    )
    passed = (
        bool(extra) and source_hash_matches
        and not (
            missing or changed or wrong_extra or wrong_trainable
            or missing_trainable
        )
    )
    result = {
        "contract": "same_cover_patch_transformer_identity_audit_v1",
        "passed": passed,
        "baseline": str(args.baseline.resolve()),
        "baseline_sha256": baseline_sha256,
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": digest(args.candidate),
        "inherited_tensor_count": len(before),
        "communicator_tensor_count": len(extra),
        "communicator_parameter_count": sum(after[name].numel() for name in extra),
        "missing": missing,
        "changed": changed,
        "wrong_extra": wrong_extra,
        "wrong_trainable": wrong_trainable,
        "missing_trainable": missing_trainable,
        "source_hash_matches": source_hash_matches,
        "extra": extra,
        "training_contract": candidate.get("training_contract"),
        "local_input_contract": inferred_local_input_contract(
            baseline, candidate,
        ),
        "cover_contract": inferred_cover_contract(candidate),
        "source_checkpoint": candidate.get("source_checkpoint"),
        "source_checkpoint_sha256": candidate.get("source_checkpoint_sha256"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise SystemExit("Patch Transformer checkpoint failed frozen-backbone audit")


if __name__ == "__main__":
    main()
