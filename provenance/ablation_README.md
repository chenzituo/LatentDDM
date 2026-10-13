# Darcy adaptation location ablation

Question: with the same complete source-trained LatentDDM and target simulations, does updating composition yield lower field error than updating parameter-matched convolutional adapters in the local operator?

## Controlled intervention

- `composition_only`: 796,400 composition parameters trainable; original local operator frozen.
- `local_adapter_only`: 810,794 adapter parameters trainable (+1.807%); original local weights and all composition weights frozen. All 32 local Conv2d maps receive flattened-kernel low-rank updates. Rank is 24 except the two scalar output convolutions (rank 1). A is Kaiming initialized and B is zero; scaling is 1. FiLM linear maps and biases are frozen.
- Both start from the exact complete budget-study source pair. No new simulations or source training. Both retain two local evaluations with shared weights, input correction, masks, normalization, and normalized-overlap assembly.
- Both use only field + 0.2 gradient + 0.25 near-boundary loss. No trace loss or code penalty. Published composition results are not substituted.

## Data and optimization

Budget-study first 16 indices of each existing seed/scale manifest, scales 8x8 and 16x16, seeds 20260918–20260920. Same 64 validation and 64 test domains per scale. Geometry seeds were checked for disjointness within each scale. These are reused test domains, previously evaluated in the budget study; this is not a newly unopened test corpus.

24 epochs, 256 replacement-sampled examples per epoch, batches 2/1, AdamW, cosine decay to 1% of initial LR, gradient clipping 1, bf16 training with GradScaler, fp32 validation and scoring. A dedicated sample generator saves exact 24x256 orders before training; both arms consume the same orders, independently of model initialization RNG. Actual consumed IDs are saved and compared before test release. Sample ordering need not reproduce the old published trainer's RNG stream.

Six LR screen trials: both arms × {3e-5, 1e-4, 3e-4}, at 8x8, m16, seed20260918. Per-arm validation-selected LR is reused for all six final runs of that arm. All 12 final runs are fresh runs, including the screen seed. Only after all selected final checkpoints are frozen does test evaluation begin, with two additional zero-shot evaluations.

## Gradient and numerical checks

The source training path disables gradients through the first local call. This isolated trainer allows gradients through both calls and the frozen composer for the local-adapter arm. No existing trainer or published artifact was edited.

Source RMS features use sqrt of mean square. Empty/all-zero faces have an undefined sqrt derivative that produced NaNs when enabling local adaptation. The isolated encoder keeps the exact source forward values and uses a zero subgradient at zero RMS in both arms. Numerical preflight confirmed source equivalence and finite gradients through both local evaluations.

Local responses use 64-tile microbatches and nonreentrant activation checkpointing when gradients are needed, identically for both arms. Source local modules have no cross-tile attention or batch statistics. TF32 is disabled: an initial check found batch-dependent TF32 differences; fp32 agrees with the unmodified source forward to 1.91e-6 in normalized pressure. A and B predictions agree exactly at initialization on the tested batches.

Five numerical unit tests pass: zero-adapter convolution and input gradient, nonzero adapter versus dense kernel, scalar rank cap, shared-loss agreement with legacy output terms, and source-identical finite-gradient RMS features. GPU preflight passed at both actual batch sizes and scales, with exhaustive parameter-change checks; peak allocated memory was about 2.4–2.8 GiB. Historical failed development checks are retained in `preflight_*_failed.log`.

## Artifacts and status

The requested package is `ablation/`:

- `protocol.json`: configuration, adapter ranks, source/data/code hashes, metric, selection and inference rules.
- `splits.csv`: actual geometry IDs for each scale, paired seed and split.
- `runs.csv`, `test_errors.csv`, `validation_history.csv`: populated by collection jobs. Empty headers before completed runs are not results.
- `orders/`: exact shared sample schedules.
- `preflight.json`, `implementation_freeze.json`, `normalizer.json`: verification and provenance.
- `screen_runs.csv`, `screen_validation_history.csv`: complete LR-screen records after collection.
- `zero_shot_errors.csv`: 128 reference rows, separate from the required 768 adapted test rows.
- `selection.json`, `test_freeze.json`: appear only after their respective stage completes.
- `analysis.json`: per-seed means, mean paired differences and paired 95% bootstrap intervals, generated only after all 768 adapted and 128 reference rows exist.
- `failed_runs.json`: failure records, pointing to preserved error logs. Missing/failed runs are never dropped from the planned comparison.

Large checkpoints, per-run histories, consumed sample IDs and Slurm logs reside at `codiffuse_Schwarz_Darcy/outputs/darcy_adaptation_location_20260921/`. `jobs.json` records the submitted dependency chain. Frozen inputs and implementation are verified at every stage. Do not resubmit this campaign or change frozen code while it is running.

Bootstrap uses 20,000 draws, independently resampling the three seed rows and 64 domain columns with pairing preserved between arms and identical domain draws across seeds. Positive local-minus-composition differences favor composition. Intervals are pointwise and conditional on the common source checkpoint; three adaptation seeds provide limited retraining uncertainty. A favorable result establishes an advantage over this one controlled local-adapter alternative, not all forms of local fine-tuning.

## Final delivery requirement

Deliver the entire `ablation/` folder, including `complexity_analysis.md`, `parameter_counts.json`, and the manuscript excerpt `complexity_analysis.tex`, together with the requested result CSVs. The complexity analysis is part of the deliverable, not only the manuscript.
