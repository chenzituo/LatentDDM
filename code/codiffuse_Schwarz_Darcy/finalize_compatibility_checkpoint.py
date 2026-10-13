"""Add canonical q-residual metadata to a legacy-internal analyzer checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from .train_residual_port_communicator import (
    CANONICAL_ANALYZER_CONTRACT, CONTRACT,
)


def finalize(path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != CONTRACT:
        raise ValueError("not a Darcy phase-2 communicator checkpoint")
    spec = checkpoint.get("communicator_spec", {})
    if spec.get("analyzer_mode") == "value_flux":
        if spec.get("canonical_analyzer_contract") != CANONICAL_ANALYZER_CONTRACT:
            raise ValueError("value+flux checkpoint lacks its canonical contract")
        return
    if spec.get("analyzer_mode") != "source_linear":
        raise ValueError("checkpoint does not use the legacy internal alias")
    spec.update({
        "canonical_analyzer_contract": CANONICAL_ANALYZER_CONTRACT,
        "legacy_internal_alias": "source_linear",
        "compatibility_residual_sign": "q=d=P_chi(v)-v",
        "q_d_coordinates": [],
        "end_to_end_source_linearity_claimed": False,
        "signal_definition": (
            "homogeneous POU consensus defect: port projection plus eight "
            "fixed linear defect moments"
        ),
        "raw_source_or_boundary_coordinates": False,
        "compatible_endpoint_signal_zero_up_to_roundoff": True,
    })
    checkpoint["communicator_spec"] = spec
    temporary = path.with_suffix(path.suffix + ".canonical.partial")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    finalize(parser.parse_args().checkpoint)
