"""Aggregate theorem-decomposition metrics across Darcy topology families."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANTS = ("local", "predicted_constant", "predicted_affine",
            "oracle_constant", "oracle_affine")
LABELS = ("local", "pred. const.", "pred. affine", "oracle const.", "oracle affine")
DOMAINS = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = {
        domain: json.loads((args.input_dir / f"domain_{domain}" / "summary.json").read_text())
        for domain in DOMAINS
    }
    rows = []
    for domain in DOMAINS:
        for variant in VARIANTS:
            rows.append({"domain": domain, "variant": variant,
                         **summaries[domain]["variant_summary"][variant]})
    metric_keys = tuple(summaries["A"]["variant_summary"]["local"])
    equal_domain = {
        variant: {key: float(np.mean([
            summaries[domain]["variant_summary"][variant][key] for domain in DOMAINS
        ])) for key in metric_keys}
        for variant in VARIANTS
    }
    theorem_keys = tuple(summaries["A"]["theorem_summary"])
    theorem = {key: float(np.mean([
        summaries[domain]["theorem_summary"][key] for domain in DOMAINS
    ])) for key in theorem_keys}
    result = {
        "contract": "darcy_one_pass_theorem_decomposition_aggregate_v1",
        "domains": list(DOMAINS), "equal_domain_weighting": True,
        "variant_summary": equal_domain, "theorem_summary": theorem,
        "source_summaries": {
            domain: str((args.input_dir / f"domain_{domain}" / "summary.json").resolve())
            for domain in DOMAINS
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output_dir / "domain_variant_metrics.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    values = np.asarray([[summaries[d]["variant_summary"][v]["relative_l2"]
                          for v in VARIANTS] for d in DOMAINS])
    x = np.arange(len(DOMAINS)); width = .16
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.45))
    colors = ("#777777", "#4c78a8", "#2b5b84", "#f58518", "#b85c00")
    for index, (label, color) in enumerate(zip(LABELS, colors)):
        axes[0].bar(x + (index - 2) * width, values[:, index], width,
                    label=label, color=color)
    axes[0].set_xticks(x, [f"Topology {domain}" for domain in DOMAINS])
    axes[0].set_ylabel("relative $L^2$ error")
    axes[0].set_title("Predicted versus identifiable oracle code")
    axes[0].grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    terms = ("oracle_affine_broken_rms", "decoded_code_error_broken_rms",
             "predicted_affine_broken_rms")
    term_labels = ("chart floor", "code gap", "total")
    term_values = np.asarray([[summaries[d]["theorem_summary"][term]
                               for term in terms] for d in DOMAINS])
    term_width = .24
    for index, (label, color) in enumerate(zip(term_labels, ("#f58518", "#54a24b", "#2b5b84"))):
        axes[1].bar(x + (index - 1) * term_width, term_values[:, index],
                    term_width, label=label, color=color)
    axes[1].set_xticks(x, DOMAINS)
    axes[1].set_ylabel("broken-space RMS")
    axes[1].set_title("Empirical terms in Theorem 1")
    axes[1].grid(axis="y", alpha=.25)
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(args.output_dir / f"darcy_theorem_decomposition.{suffix}", dpi=220)
    plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
