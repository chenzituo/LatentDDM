# Final analysis specification — 2026-09-19

Recorded while recovery job 23118753 is running, before this campaign's test
scoring. Complements PROTOCOL.md without modifying its frozen contents.

- Evaluate all 54 validation-selected adaptation checkpoints, plus one source
  composer and one selected spatial Patch checkpoint per scale (58 evaluations).
  Full FT and LoRA share one m0 Patch reference; m0 has no adaptation replicates.
- Primary endpoint: arithmetic mean of physical-domain relative L2, first over
  the same 64 test cases and then equally over the three seeds.
- Use 20,000 percentile bootstrap draws, RNG seed 20260919. Resample three seed
  indices and 64 physical-domain indices independently with replacement. Each
  draw uses the SAME resampled domain vector across selected seeds, methods,
  and budgets because the test cases are crossed with seeds, not independent
  cases nested within seeds. This is a paired two-level/crossed bootstrap.
  m0 resamples domains only. Report 2.5th and 97.5th percentiles.
- Report full-minus-latent and LoRA-minus-latent paired error differences at
  every measured budget and scale. Intervals are pointwise, with no simultaneous
  coverage claim. Positive lower bounds for both baselines at m4 and m16 in
  both scales support the stated low-training-budget advantage. Report all
  comparisons, including unfavorable or inconclusive ones.
- Show individual seed means, bootstrap intervals, and measured budgets only.
  No accuracy threshold or interpolated label-savings claim will be made.
- Adaptation time uses subprocess end-minus-start from driver command records,
  including startup, training, validation and checkpoint writes, excluding queue
  time. Preserve trainer-reported time separately; its boundaries differ across
  implementations. Show individual times and hardware/job mapping. Timing is
  descriptive on the recorded hardware, not a controlled hardware speedup.
- Compute ledger retains source training, screening, sweep, recovery, evaluation,
  and the original six-hour timeout allocation; never discard interrupted work.
  Scheduler allocation and process elapsed time are separate quantities.
- Only three adaptation seeds and fixed source checkpoints; intervals do not
  quantify source-training variability. At m64 only optimization varies. Tests
  were historically inspected; this is not fresh held-out confirmation.
- Labels: "Target training simulations"; 64 target-validation simulations per
  scale are additional overhead, including selection informing m0. Complete
  source recipes differ. Existing SNI evidence remains separate.
- Frozen training hashes are checked before scoring. Evaluation records retain
  exact commands, checkpoint/data/code hashes and all per-case metrics. Incomplete
  outputs are not silently overwritten, nor are incomplete runs dropped.
