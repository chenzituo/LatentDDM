"""Prospective reporting-set contract shared by Airfoil evaluators."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


LEGACY_FINAL_HOLDOUT_CONTRACT = "airfoil_unopened_boundary_final_holdout_v2"
FINAL_HOLDOUT_CONTRACT = "airfoil_unopened_boundary_final_holdout_v3"
FINAL_HOLDOUT_CONTRACTS = {
    LEGACY_FINAL_HOLDOUT_CONTRACT,
    FINAL_HOLDOUT_CONTRACT,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_final_holdout(manifest: dict[str, Any] | None) -> bool:
    return bool(
        manifest is not None
        and manifest.get("contract") in FINAL_HOLDOUT_CONTRACTS
    )


def validate_unopened_manifest(
    manifest: dict[str, Any],
    *,
    selection_path: Path | None = None,
    selected_checkpoint_hashes: set[str | None] | None = None,
) -> None:
    """Reject an unknown, opened, or selection-unbound v3 manifest.

    The legacy v2 contract remains readable for reproducibility. New v3
    reporting sets are stricter: their trajectory contents and metrics must
    still be unopened, and the manifest is cryptographically bound to the
    already frozen validation selection and its checkpoint.
    """
    contract = manifest.get("contract")
    if contract not in FINAL_HOLDOUT_CONTRACTS:
        raise ValueError("unexpected prospective manifest contract")
    if contract != FINAL_HOLDOUT_CONTRACT:
        return
    if manifest.get("trajectory_contents_opened") is not False:
        raise ValueError("prospective v3 trajectory contents are already opened")
    if manifest.get("metrics_opened") is not False:
        raise ValueError("prospective v3 metrics are already opened")
    if selection_path is None:
        raise ValueError("prospective v3 manifest requires a selection record")
    expected_selection_sha = manifest.get("validation_selection_sha256")
    if expected_selection_sha != file_sha256(selection_path):
        raise ValueError("prospective v3 manifest and selection record differ")
    if selected_checkpoint_hashes is not None:
        frozen_sha = manifest.get("selected_checkpoint_sha256")
        if frozen_sha not in selected_checkpoint_hashes:
            raise ValueError("prospective v3 manifest binds another checkpoint")


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("prospective manifest is not a JSON object")
    return payload
