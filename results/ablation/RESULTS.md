# Final Darcy adaptation-location ablation

All six learning-rate trials, twelve final adaptation runs, fourteen test/reference evaluations, and paired analysis completed successfully. No failed production runs.

## Test accuracy

Mean per-domain relative L2 over three paired adaptation seeds and 64 shared test domains per scale. Delta = local adapter minus composition; positive favors composition.

| Scale | Zero-shot | Composition | Local adapters | Delta [paired 95% CI] |
|---|---:|---:|---:|---|
| 8x8 | 0.12157 | 0.07866 | 0.07484 | -0.00382 [-0.01031, +0.00177] |
| 16x16 | 0.12649 | 0.06685 | 0.06839 | +0.00154 [-0.00188, +0.00463] |

Neither interval excludes zero. This study does not establish accuracy superiority of either adaptation location. It also does not establish statistical equivalence or noninferiority, for which no margin was specified. Both arms improve substantially over their common zero-shot reference.

## Per-seed test means

| Scale | Seed | Composition | Local adapters | Delta |
|---|---:|---:|---:|---:|
| 8x8 | 20260918 | 0.07272 | 0.07305 | +0.00033 |
| 8x8 | 20260919 | 0.07935 | 0.07827 | -0.00108 |
| 8x8 | 20260920 | 0.08391 | 0.07321 | -0.01070 |
| 16x16 | 20260918 | 0.06704 | 0.06967 | +0.00263 |
| 16x16 | 20260919 | 0.06748 | 0.07093 | +0.00345 |
| 16x16 | 20260920 | 0.06605 | 0.06457 | -0.00148 |

## Adaptation cost

All final runs used NVIDIA L40S GPUs. Means below cover the three final runs per arm and scale. Timing covers the full 24-epoch run, including initialization, validation, and checkpoint writes; it is not time to a shared accuracy threshold or an inference benchmark.

| Scale | Composition minutes | Local-adapter minutes | Speed ratio (local/composition) | Time reduction | Composition peak MiB | Local-adapter peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| 8x8 | 8.90 | 13.81 | 1.553x | 35.6% | 2468.2 | 2591.3 |
| 16x16 | 33.51 | 53.74 | 1.604x | 37.6% | 2569.9 | 2793.9 |

The supported result is lower measured adaptation cost for composition, with no clear accuracy separation in this paired test. This is an implementation- and workload-specific efficiency finding. Parameter counts and backward-path differences are explained in `complexity_analysis.md`; future benefits for much larger local physics models remain a scaling motivation.

## Statistical scope

20,000 paired bootstrap draws independently resample the three seed rows and 64 domain columns, preserving arm pairing and shared domain draws across seeds. Intervals are pointwise and conditional on the common source checkpoint. Only three adaptation seeds are available. Test domains were reused from the earlier budget study; no test-driven tuning occurred in this ablation.

## Delivery

The folder contains all requested protocol/split/run/history/per-domain-error files, zero-shot errors, screening records, frozen selections and hashes, complexity analysis in Markdown and LaTeX, and machine-readable parameter counts. `verification_final.json` records the final checks. Large checkpoint binaries and original execution logs remain in the campaign output directory documented in the parent README.
