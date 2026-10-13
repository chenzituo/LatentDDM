"""Durable history reconstruction for preemptible global-model campaigns."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def atomic_write_json(path: Path, payload: dict) -> None:
    """Replace a JSON artifact only after its complete payload is durable."""
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict) -> None:
    """Preserve the prior checkpoint until a complete replacement is written."""
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def merged_training_history(
    start_epoch: int,
    *,
    summary_path: Path | None = None,
    history_log: Path | None = None,
) -> list[dict]:
    """Merge summary and JSONL records, retaining one row per completed epoch."""
    records: dict[int, dict] = {}
    if summary_path is not None and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for record in summary.get("history", []):
            epoch = int(record.get("epoch", 0))
            if 0 < epoch <= start_epoch:
                records[epoch] = record
    if history_log is not None and history_log.exists():
        for line in history_log.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            epoch = int(record.get("epoch", 0))
            if 0 < epoch <= start_epoch:
                records[epoch] = record
    return [records[epoch] for epoch in sorted(records)]


def require_complete_training_prefix(history: list[dict], start_epoch: int) -> None:
    epochs = [int(record["epoch"]) for record in history]
    if epochs != list(range(1, start_epoch + 1)):
        raise RuntimeError(
            f"resume history is not the complete epoch prefix 1..{start_epoch}"
        )
