# Darcy ICLR boost protocol v1 — 2026-09-18

Status: protocol recorded before new training or test scoring. The historical
test set has previously been inspected. This is a new protocol on reused tests,
not preregistration on a fresh confirmation set. User authorized a few days of
Darcy GPU experiments. No new airfoil experiments or new dataset generation.

## Scientific question and limits

Does composer adaptation retain a low-target-training-budget field-error
advantage against a validation-selected spatial Patch model with full or
parameter-matched low-rank adaptation, at 8×8 and 16×16 fixed-spacing domains?
This compares complete source-training recipes, including different source
supervision. It is not a causal isolation of architecture or a claim that m
training cases are the total simulation acquisition cost.

## Corrected historical ledger

The actual selected Patch checkpoint records `normalized_linear_fourier_v1`.
Its implementation adds normalized Fourier patch positions. Thus the manuscript
and the earlier feasibility assessment's blanket “global controls omit positional
encoding” description are incorrect for Patch. The new experiment tests
fixed-spacing spatial representations, not the presence versus absence of positions.

The local BVP checkpoint uses 40,000 40×40 fields (4,000 shapes × 10 conditions),
not 36,000. Nominal core values are 40.96M, while full stored crop values are 64M
before masks. The selected global checkpoint uses `darcy_full_domain_face_96_v2.h5`
with 4,000 96×96 fields. LatentDDM additionally trains its source composer on
full-domain source labels, initialized from an earlier field-token model. This
study does not make an equal-source-cost claim or retrain the local BVP operator.
Checkpoint arguments and input hashes are retained in `input_provenance.json`.

Historical losses and selection rules differed. This new study uses common
output losses and validation field scoring, retaining the method-specific trace
loss for LatentDDM. Historical m=16 scores are not substituted into the new curve.

## Data, labels, and selection

- Target files: fixed-dx 8×8 and 16×16; each has 64 train, 64 val, 64 test domains.
- Training budgets m=4,16,64; m=0 is a frozen source-weight reference.
- Three paired subset/optimization seeds: 20260918, 20260919, 20260920.
  Each manifest permutes all 64 training IDs; budgets use nested prefixes.
  At m=64 all runs share the full pool, leaving optimization randomness only.
- Every method sees the same IDs within a seed/budget. Batch size is 2 at 8×8
  and 1 at 16×16, for all methods. Adaptation uses 24×256 sample exposures,
  AdamW, weight decay 1e-4, cosine scheduling to 0.01×initial LR.
- **64 labeled target-validation cases are additional overhead.** Plots will
  say “target training simulations.” Target-validation-informed configuration
  selection also applies to the m=0 source-weight reference; call this frozen
  source inference, not a completely target-blind model selection protocol.
- Primary checkpoint criterion: mean physical-domain relative L2 over all 64
  validation cases. All adaptation methods use the same field + 0.2 gradient +
  0.25 near-boundary MSE in source-normalized pressure units. LatentDDM retains
  0.5 trace loss and 1e-5 code penalty. No extra labels beyond the same field
  simulations are acquired for these derived targets. Trace supervision and
  source pretraining remain part of its recipe, and will be disclosed.
- The source normalizer is frozen. New global sources use the local BVP
  statistics. The historical normalized Patch candidate retains its own source
  checkpoint statistics; their equality/difference must be recorded.

## Stronger spatial controls and screening

Both new candidates use the existing full-domain Patch architecture: patch32,
width256, depth8, heads8, MLP ratio4. No local operator, overlap assembly, or
additional learned parameters. Source-train each for 80×4096 samples, batch16,
LR2e-4 to2e-6, seed20260918, on all 4,000 source training cases. Select on all
250 source-validation cases. Use the shared output-field losses above.

1. `fixed_tile_fourier_v1`: sinusoidal positions in fixed patch units, with
   geometrically spaced frequencies from 1 to 1e-4 radians per patch.
2. `fixed_tile_fourier_locality_v1`: same positions; four attention heads have
   Gaussian distance biases with scales geometrically spaced from one to four
   patch widths; four heads remain globally unrestricted.

These are controlled spatial Patch variants, not a reproduction of another
paper's rotary-position method. Fixed spacing preserves the meaning of local
distances as the domain grows. The unchanged normalized-position historical
Patch checkpoint remains eligible so that the study does not force selection
of an inferior newly proposed model.

Screen on 8×8,m16,seed20260918 using LR={3e-5,1e-4,3e-4}. Full-FT evaluates
all three source candidates × three LRs. Choose the lowest validation L2; retain
all trials. LoRA uses that same selected backbone and screens the three LRs.
LatentDDM screens the same three LRs. The global architecture search has nine
full-FT trials versus three LatentDDM trials; disclose this conservative extra
baseline search rather than calling total search budgets identical.
Freeze source choice and one LR per method in `selection.json` before the
budget sweep or test scoring. Reuse those choices at 16×16 with no second search.

LoRA rank24, attention+FFN placement, 786,432 trainable parameters versus
796,400 for the composer. Rank is chosen by parameter count, not validation.
Zero-delta and merged-checkpoint inference must pass before production.

## Gates, analysis, and execution stages

1. Unit checks: nested subsets/rejected invalid manifests, fixed-distance
   consistency, padding-mask isolation, checkpoint reconstruction, LoRA
   zero-delta and merge agreement, finite nonzero adapter gradients.
2. GPU smoke: source train, full/LoRA/composer adaptation and validation inference
   on two cases. This validates execution, not scientific performance.
3. Source training -> validation screening -> 8×8/16×16 sweeps.
4. Final evaluation is a separate stage after confirming completed training,
   frozen selection and recorded provenance. No training phase opens test.

Primary reporting: per-domain relative L2, paired hierarchical 95% bootstrap
intervals over aligned seed/subset runs and physical domains, and adaptation
wall time. Three adaptation seeds do not quantify source-pretraining variability.
SNI remains a separate label-free accuracy/time reference using existing evidence.
Do not pool its target-label usage with supervised methods.

Proposed decision: positive global-minus-LatentDDM interval lower bounds at both
m4 and m16 support low-training-budget advantage in that setting; overlapping
intervals are inconclusive/budget-specific, and negative intervals narrow the
claim. Report the stronger comparator even if it removes the advantage. No
unmeasured interpolation of label savings. Any common accuracy threshold must
be declared from validation before final scoring, or omitted entirely.

Reserve at most two concurrent one-GPU jobs initially. Source jobs receive six
hours each; screening and each scale sweep receive six hours each. These are
scheduler ceilings, not measured cost estimates. A failed or timed-out job stops
dependent phases; inspect the failure before retrying. Preserve logs and failed
runs. Actual cost and hardware are recorded per stage.

## Manuscript review contract

User requested `$review-edit-latex`. Preserve manuscript prose and add explicit
proposals/comments in the review layer; compile and render affected pages after
each edit. Integrate wording only after author approval. Experimental results
will be distinguished from planned work. The writing-only AI disclosure needs
a proposed correction because experiment-design and coding assistance are now
part of the workflow. No manuscript edits have yet been made for this campaign.
