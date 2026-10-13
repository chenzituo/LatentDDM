# Minimal reproduction of the current ICLR manuscript

Snapshot for `iclr2027/overleaf/main.tex`, packaged September 22, 2026.
This directory is self-contained **code and small result artifacts**. Large datasets
and checkpoints remain external; their relative paths, sizes and SHA-256 hashes
are in `assets.json`. No new simulations or source pretraining are needed.

## Scope

| Study | Included reproduction |
|---|---|
| Darcy domain extension | Composition, Point, Slice, Patch target training; zero-shot/adapted held-out evaluation at 4×4, 8×8, 16×16; SNI-16 evaluation; archived validation field/timing plots |
| Darcy training budgets | Exact recorded target-training commands and test evaluation for 3 methods × 3 budgets × 2 scales × 3 seeds, plus zero-shot references; recomputed table means and paired bootstrap intervals |
| Airfoil zero-shot | Five methods, source-ID and target-OOD splits, H=20 and 3 starts per trajectory; frozen checkpoints/splits; archived per-trajectory rollout outputs |
| Darcy adaptation location | Complete two-arm implementation and staged rerun; all 768 test rows, 128 reference rows, validation history, parameter counts and complexity analysis |
| Airfoil per-trajectory calibration, Figure 4(b) | **Excluded at the author's request.** No substitute experiment is presented as its reproduction. |

This is reproduction from the selected source checkpoints, not a from-scratch
source-pretraining release. The historical supporting modules are retained to keep
checkpoint loading and frozen ablation checks compatible. `source_provenance.json`
records their origins and hashes. `recipes.json` contains the explicit commands;
the runner does not choose hyperparameters using test data.

## Quick start: no GPU or model/data assets

Use Python 3.12. For the small result checks, only NumPy is needed; figures also
need Matplotlib. `requirements.txt` records the versions tested here.

```bash
cd iclr2027/reproduction
python reproduce.py summarize --output build/tables
python figures/make_figure3.py --output-dir build/figures
```

The first command recomputes Darcy budget means and 20,000-draw intervals from
per-domain errors, and recomputes the ablation's paired intervals while checking
matching and the error norms. It verifies these against the archived analysis.
Airfoil and domain-extension tables are extracted from archived aggregates; that
step is explicitly not a new field evaluation. The figure script's historical
name says Figure 3; its Darcy field/timing panels belong to the current paper's
Darcy figure. It uses archived validation plot arrays.

The complete ablation delivery is `results/ablation/`, including
`complexity_analysis.md`, `complexity_analysis.tex`, and `parameter_counts.json`.
The local-minus-composition intervals include zero at both scales. The measured
adaptation cost advantage should not be described as an accuracy superiority result.

To recompute the Airfoil aggregate and bootstrap intervals from the included
per-trajectory rollout records (also requires only NumPy):

```bash
PYTHONPATH="$PWD/code" python -m codiffuse_Schwarz_Airfoil.aggregate_pitch_rate_ood_rollouts \
  --evaluation-root results/airfoil/raw --output build/airfoil_reaggregated.json \
  --split-part test --zero-shot-only --bootstrap-samples 2000 \
  --zero-shot-freeze results/airfoil/zero_shot_freeze_validation.json
```

## External assets

Set `ASSET_ROOT` to the project root or an external asset mirror preserving the
relative paths in `assets.json`. The package does not require the original project
source code when executing the recipes.

```bash
ASSET_ROOT=/path/to/AMR_latentDDM
python reproduce.py check-assets --assets "$ASSET_ROOT"
# Full content verification (reads all listed files):
python reproduce.py check-assets --assets "$ASSET_ROOT" --sha256
# Or check only the direct inputs of one recipe:
python reproduce.py check-assets --assets "$ASSET_ROOT" \
  --recipe budget/eval/sweep_8x8_m16_latent_seed20260918 --sha256
```

Existing workspace asset root:
`/orcd/home/002/zituo/projects/AMR_Diffusion_Foundation/AMR_latentDDM`.
Dataset documentation is in that project's `datasets/latentddm_physworldai_v1/`.

Airfoil additionally needs `--airfoil-data`. **The retained evaluator opens paired
L5 and L6 files, even though reported field scores are on L5.** The L5-only dataset
release directory therefore does not suffice for this evaluator. The 68 required
files are listed in `airfoil_data_manifest.json`; sizes, but not freshly recomputed
content hashes, are recorded there. Existing paired data root:
`/orcd/scratch/orcd/003/zituo/AMR_Diffusion_Foundation/codiffuse_airfoil`.

Historical absolute paths inside checkpoint metadata are remapped in memory to
`ASSET_ROOT`; weights and checkpoint files are not rewritten. Budget manifests
are copied into each new output directory with only their dataset path remapped.
The permutations and dataset hashes are retained.

## Evaluate or adapt one model

```bash
python reproduce.py list --filter budget/
python reproduce.py list --filter domain/
python reproduce.py list --filter airfoil/

# Prints the resolved command. Add --execute to run.
python reproduce.py run budget/eval/sweep_8x8_m16_latent_seed20260918 \
  --assets "$ASSET_ROOT" --output /path/to/new-evaluation

# One-domain wiring check; not a reported experiment.
python reproduce.py run budget/eval/sweep_8x8_m16_latent_seed20260918 \
  --assets "$ASSET_ROOT" --output /path/to/new-smoke --smoke --execute

# Complete 24-epoch target adaptation from the selected source checkpoint.
python reproduce.py run budget/train/sweep_8x8_m16_latent_seed20260918 \
  --assets "$ASSET_ROOT" --output /path/to/new-training --execute

# Evaluate the newly trained model rather than the archived checkpoint.
python reproduce.py run budget/eval/sweep_8x8_m16_latent_seed20260918 \
  --assets "$ASSET_ROOT" --output /path/to/new-trained-evaluation \
  --checkpoint /path/to/new-training/training/checkpoint_best.pt --execute

python reproduce.py run airfoil/eval/latent_ood --assets "$ASSET_ROOT" \
  --airfoil-data /path/to/paired-airfoil-data \
  --output /path/to/new-airfoil-evaluation --execute
```

Every execution requires a fresh output directory and records its command and exit
status. Evaluation defaults to the frozen archived checkpoint; use `--checkpoint`
to evaluate a new run. Global models write `training/checkpoint.pt`, composition
models `training/checkpoint_best.pt`. Training is intended for a CUDA node. No job
is submitted automatically, and listing recipes does not launch the matrix.

Keep the studies separate: domain-extension training uses indices 0–15 and seeds
20260810–12, with its original composite validation criterion. Budget training
uses the saved subset permutations and seeds 20260918–20, selecting by validation
relative L2. The budget Patch model's positional encoding differs from the
domain-extension Patch. The recipes preserve these distinctions.

## Rerun the controlled ablation

The full protocol has **six learning-rate screening runs and twelve final training
runs**, followed by fourteen evaluations (twelve adapted and two zero-shot).
On an allocated CUDA node:

```bash
REPRO_PYTHON=python bash run_ablation.sh "$ASSET_ROOT" /path/to/new-ablation
```

To inspect or schedule stages individually:

```bash
python reproduce.py ablation init --assets "$ASSET_ROOT" --work-dir /path/to/new-ablation
python reproduce.py ablation prepare --assets "$ASSET_ROOT" --work-dir /path/to/new-ablation
python reproduce.py ablation preflight --assets "$ASSET_ROOT" --work-dir /path/to/new-ablation
# Next: screen --index 0..5; select; train --index 0..11;
# freeze-test; test --index 0..13; collect.
```

`init` creates an isolated source workspace and links only the required read-only
source checkpoints and Darcy data. The original experiment then prepares fresh
provenance and identical paired sample schedules, performs its numerical checks,
and selects rates/checkpoints on validation. Source checkpoints are already
trained: the local-only arm freezes a **source-trained composition module**, not a
random communication module. Both local evaluations share the same adapted local
operator.

New results are written beneath
`NEW_WORK_DIR/workspace/iclr2027/experiments/darcy_adaptation_location_20260921/ablation/`;
new large checkpoints are in that workspace's Darcy outputs directory. The
published deliverable and complexity accounting remain in `results/ablation/`.

## Validation and limits

`verification.json` records packaging checks. They cover saved-result numerical
reproduction, import/CLI isolation, representative real-checkpoint inference,
and the ablation's initialization/gradient/parameter-update preflight at both
scales. This packaging task does not repeat the full training campaigns or all
held-out evaluations. Different hardware/PyTorch kernels may change timing and
floating-point results; original timings are archived measurements.
