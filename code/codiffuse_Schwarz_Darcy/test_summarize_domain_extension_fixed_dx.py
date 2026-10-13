from __future__ import annotations

import json

import pytest

from .summarize_domain_extension_fixed_dx import adaptation_metadata
from .summarize_domain_extension_fixed_dx import (
    input_chart_oracle_rows,
    input_correction_rows,
    sni_operating_points,
    sni_selection,
    relative_reduction_of_means,
)


def write_summary(path, *, seconds: float, trainable: int = 7) -> None:
    path.write_text(json.dumps({
        "wall_seconds": seconds,
        "trainable_parameter_count": trainable,
        "train_cases": 16,
        "samples_per_epoch": 256,
        "history": [{"epoch": epoch} for epoch in range(24)],
    }))


def test_adaptation_metadata_uses_equal_sample_exposure(tmp_path) -> None:
    paths = [tmp_path / f"seed{seed}.json" for seed in range(3)]
    for index, path in enumerate(paths):
        write_summary(path, seconds=10.0 + index)
    result = adaptation_metadata(paths)
    assert result["wall_seconds_mean"] == pytest.approx(11.0)
    assert result["trainable_parameter_count"] == 7
    assert result["target_train_cases"] == 16
    assert result["sample_exposures"] == 24 * 256


def test_adaptation_metadata_rejects_unequal_parameter_contract(tmp_path) -> None:
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    write_summary(paths[0], seconds=1.0, trainable=7)
    write_summary(paths[1], seconds=1.0, trainable=8)
    with pytest.raises(ValueError, match="training contract"):
        adaptation_metadata(paths)


def test_input_correction_rows_select_one_method(tmp_path) -> None:
    path = tmp_path / "validation.json"
    path.write_text(json.dumps({
        "contract": "darcy_input_correction_matched_validation_v1",
        "protected_test_cases_opened": 0,
        "rows": [
            {"case_index": 3, "method": "local_core", "relative_l2": .4},
            {"case_index": 3, "method": "latentddm", "relative_l2": .1},
            {"case_index": 8, "method": "latentddm", "relative_l2": .2},
        ],
    }))
    assert input_correction_rows(path, "latentddm") == {3: .1, 8: .2}


def test_input_chart_oracle_rows_use_matching_trace_family(tmp_path) -> None:
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps({
        "contract": "darcy_boundary_chart_capacity_oracle_v1",
        "protected_test_cases_opened": 0,
        "rows": [{
            "case_index": 5,
            "projected_trace_modes_2_relative_l2": .09,
            "projected_trace_modes_4_relative_l2": .03,
        }],
    }))
    assert input_chart_oracle_rows(path, modes=4) == {5: .03}


def test_sni_selection_keeps_case_aligned_timing(tmp_path) -> None:
    path = tmp_path / "sni.json"
    rows = []
    for case, local in ((7, .4), (2, .2)):
        rows.extend((
            {
                "case_index": case, "tau": .5, "sweep": 0,
                "relative_l2": local, "seconds": 0.0,
            },
            {
                "case_index": case, "tau": .5, "sweep": 1,
                "relative_l2": local * .75, "seconds": case / 10,
            },
            {
                "case_index": case, "tau": .5, "sweep": 2,
                "relative_l2": local * .5, "seconds": case / 5,
            },
        ))
    path.write_text(json.dumps({"rows": rows}))
    result = sni_selection(path, tau=.5, sweep=2)
    assert result["sweep"] == 2
    assert result["cases"] == [2, 7]
    assert result["seconds"].tolist() == pytest.approx([.4, 1.4])
    assert result["gain"].tolist() == pytest.approx([.5, .5])
    points = sni_operating_points(path, tau=.5)
    assert [(point["tau"], point["sweep"]) for point in points] == [
        (.5, 1), (.5, 2),
    ]
    assert points[0]["seconds"].tolist() == pytest.approx([.2, .7])


def test_aggregate_reduction_is_distinct_from_mean_paired_gain() -> None:
    baseline = [1.0, 9.0]
    candidate = [.5, 6.0]
    assert relative_reduction_of_means(baseline, candidate) == pytest.approx(.35)
    paired = ((1.0 - .5) / 1.0 + (9.0 - 6.0) / 9.0) / 2.0
    assert paired == pytest.approx(5.0 / 12.0)
