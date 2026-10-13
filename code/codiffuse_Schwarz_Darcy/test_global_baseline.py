from __future__ import annotations

from argparse import Namespace

import torch

from codiffuse_Schwarz_Darcy.adapter import H5DarcyDataset, compute_statistics
from codiffuse_Schwarz_Darcy.dataset import DarcyDatasetConfig, generate_hdf5
from codiffuse_Schwarz_Darcy.evaluate_global import evaluate_global
from codiffuse_Schwarz_Darcy.full_domain_dataset import (
    DarcyFullDomainConfig,
    generate_full_domain_hdf5,
)
from codiffuse_Schwarz_Darcy.global_operator import MaskedTransolverOperator
from codiffuse_Schwarz_Darcy.train_global import train_global


def test_masked_transolver_supports_variable_domains_and_gradients() -> None:
    model = MaskedTransolverOperator(
        12, 4, width=32, depth=2, heads=4, slices=8, mlp_ratio=2,
    )
    spatial = torch.randn(2, 12, 12, 16, requires_grad=True)
    scalars = torch.randn(2, 4)
    mask = torch.ones(2, 1, 12, 16)
    mask[:, :, :3, :4] = 0.0
    output = model(spatial, scalars, mask)
    assert output.shape == (2, 1, 12, 16)
    assert torch.all(output[:, :, :3, :4] == 0.0)
    output.square().mean().backward()
    assert spatial.grad is not None
    assert torch.isfinite(spatial.grad).all()


def test_outside_values_do_not_change_active_prediction() -> None:
    torch.manual_seed(3)
    model = MaskedTransolverOperator(
        5, 2, width=24, depth=1, heads=4, slices=6, mlp_ratio=2,
    ).eval()
    torch.nn.init.normal_(model.output[-1].weight)
    spatial = torch.randn(1, 5, 8, 8)
    mask = torch.ones(1, 1, 8, 8)
    mask[:, :, :2] = 0.0
    changed = spatial.clone()
    changed[:, :, :2] = 1000.0 * torch.randn_like(changed[:, :, :2])
    first = model(spatial, torch.zeros(1, 2), mask)
    second = model(changed, torch.zeros(1, 2), mask)
    torch.testing.assert_close(first[:, :, 2:], second[:, :, 2:])


def test_global_train_and_evaluate_cpu_smoke(tmp_path) -> None:
    data = generate_full_domain_hdf5(
        tmp_path / "full_smoke.h5", DarcyFullDomainConfig.smoke(seed=17),
    )
    held_out = generate_hdf5(
        tmp_path / "held_out_smoke.h5", DarcyDatasetConfig.smoke(seed=17),
    )
    statistics_checkpoint = tmp_path / "local_statistics.pt"
    torch.save({
        "contract": "darcy_arbitrary_current_direct_endpoint_v1",
        "statistics": compute_statistics(
            H5DarcyDataset(data, "train"), max_cases=4, seed=17,
        ).to_json(),
    }, statistics_checkpoint)
    output = tmp_path / "train"
    train_args = Namespace(
        data=data, output_dir=output, epochs=1, samples_per_epoch=4,
        batch_size=2, workers=0, stats_cases=4,
        max_train_cases=4, max_val_cases=2,
        validation_sweeps=1, validation_batches=1, local_weight=1.0,
        current_policy="zero", use_discrete_residual=False,
        selection_residual_weight=0.0,
        lr=2.0e-4, min_lr=2.0e-6, weight_decay=1.0e-4,
        width=24, depth=1, heads=4, slices=6, mlp_ratio=2,
        seed=17, resume=None, cpu=True,
        statistics_checkpoint=statistics_checkpoint,
    )
    result = train_global(train_args)
    assert result["parameter_count"] > 0
    assert result["current_policy"] == "zero"
    assert result["use_discrete_residual"] is False
    assert result["inference_contract"] == "pde_agnostic_direct_solution_v1"
    checkpoint = output / "checkpoint.pt"
    assert checkpoint.exists()

    pretrain_output = tmp_path / "global_local_pretraining"
    pretrain_args = Namespace(**{
        **vars(train_args),
        "data": held_out,
        "output_dir": pretrain_output,
        "training_stage": "local_pretraining",
        "initial_checkpoint": None,
    })
    pretrain_result = train_global(pretrain_args)
    assert (
        pretrain_result["training_data_schema"]
        == "darcy_complex_cartesian_v2_face_dirichlet"
    )
    matched_output = tmp_path / "global_matched_finetune"
    matched_args = Namespace(**{
        **vars(train_args),
        "output_dir": matched_output,
        "training_stage": "full_domain",
        "initial_checkpoint": pretrain_output / "checkpoint.pt",
    })
    matched_result = train_global(matched_args)
    assert matched_result["training_stage"] == "full_domain"
    assert matched_result["initial_checkpoint"].endswith("checkpoint.pt")

    evaluation = evaluate_global(Namespace(
        data=held_out, checkpoint=checkpoint, output_dir=tmp_path / "evaluation",
        groups=("A", "B"), sweeps=1, batch_size=1, workers=0,
        relaxation=1.0, initialization="zero", max_cases=1, cpu=True,
    ))
    assert evaluation["evaluated_records"] == 2
    assert "A/sweep_001" in evaluation["summary"]
    assert "B/sweep_001" in evaluation["summary"]
