# Darcy adaptation-location deliverable

The final deliverable is this entire `ablation/` directory.

- `protocol.json`, `splits.csv`: frozen configuration, provenance, and paired domain membership.
- `runs.csv`, `validation_history.csv`, `test_errors.csv`: final training records and 768 per-domain adapted test errors on completion.
- `zero_shot_errors.csv`: 128 common-source reference errors on completion.
- `analysis.json`: paired accuracy effects and confidence intervals, generated when complete.
- `complexity_analysis.md`: self-contained parameter, forward/backward work, memory, and scaling analysis.
- `parameter_counts.json`: exact machine-readable parameter counts.
- `complexity_analysis.tex`: corresponding manuscript appendix source excerpt.
- `screen_runs.csv`, `screen_validation_history.csv`, `selection.json`: learning-rate screening and selection records.
- `preflight.json`, `implementation_freeze.json`, `normalizer.json`, `orders/`: checks, hashes, normalization, and paired sample orders.
- `failed_runs.json`: failure records, with links to retained logs.

Check `status.json` before interpreting incomplete CSVs as final results. The complexity analysis is available now; it does not assert that training or test evaluation has completed. Large checkpoints and execution logs remain in the campaign output directory documented in the parent README.

## Completed delivery

All stages are complete. Start with `RESULTS.md` for accuracy, paired intervals, per-seed results, and adaptation costs. See `verification_final.json` for delivery checks and `complexity_analysis.md` for parameter and theoretical accounting.
