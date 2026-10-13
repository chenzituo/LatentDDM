# Parameter counts and computational complexity

This analysis accompanies the Darcy adaptation-location ablation. It describes the actual forward and backward paths, independently of whether either arm obtains lower test error. See `protocol.json` for the frozen experimental configuration and source hashes, and `parameter_counts.json` for machine-readable counts.

## Parameter accounting

Let P_S denote the original local operator's parameters, P_C the composition module's parameters, and P_A the added local adapters' parameters.

| Quantity | Composition adaptation | Local-adapter adaptation |
|---|---:|---:|
| Original local operator, P_S | 3,957,666 frozen | 3,957,666 frozen |
| Composition module, P_C | 796,400 trainable | 796,400 frozen |
| Local adapters, P_A | 0 | 810,794 trainable |
| Distinct forward parameters, unmerged | 4,754,066 | 5,564,860 |
| Parameters updated during adaptation | 796,400 | 810,794 |

The local operator contains 2,755,841 proposal parameters and 1,201,825 projector parameters. Composition contains 56,160 field-encoder, 141,312 spatial-encoder, and 598,928 communication parameters. Composition updates 16.75% of the complete source model's parameters; P_S/P_C = 4.97. Adapter parameters exceed composition parameters by 1.81%.

“Forward parameters” counts distinct parameter tensors used during prediction. Frozen parameters remain active in forward computation. Reusing the local operator across N subdomains and two calls does not multiply its parameter count by 2N.

Each convolutional adapter forms W_eff = W + BA, with W flattened to shape C_out by (k² C_in). Its parameter count is r(k² C_in + C_out). All 32 local convolutional layers receive rank-24 updates, capped at rank 1 for the two scalar-output layers. A is Kaiming initialized and B starts at zero. These adapters can be merged into the original convolutional kernels for inference, returning the distinct deployed parameter count to P_S + P_C; inference benchmarks of this merged representation have not been run as part of this analysis.

## Forward work

Let N be the number of subdomains, q the grid points per local input (40×40 = 1,600 here), L the composition depth (4), and d the token width (128). Define F_S(N,q) = N f_S(q) as the work of one local-operator evaluation across the domain, and F_C as composition work, including token encoding and correction decoding.

With fixed physical-channel counts and trace resolution:

```text
F_C(N,q) = O(Nq + L N d² + L N² d)
F_prediction = 2 F_S(N,q) + F_C(N,q) + O(Nq)
```

The last term accounts for assembly. The global and diagonal executions of the centered composer affect constants but not this order. The diagonal branch is implemented without a dense attention matrix. Composition parameter count is independent of N; global attention work is still quadratic in N.

For this fixed-resolution CNN, leading convolutional work is Θ(q P_S,conv) per tile, where P_S,conv counts convolutional kernel weights. Scalar-conditioning and elementwise operations contribute additional work. Other local backbones can have different f_S(q); the two-local-call structure is unchanged. These are operation-count expressions, not GPU wall-time predictions.

## Backward work

Use superscripts (1) and (2) for the preliminary and corrected local evaluations. Define:

- B_S^x: backward work through a frozen local operator to its corrected input.
- B_C^phi: backward work computing composition parameter gradients.
- B_C^z: backward work through the frozen composer to preliminary local fields.
- B_S,A^(1): backward work computing local-adapter gradients through the first local evaluation.
- B_S,A^{x,(2)}: backward work computing both local-adapter and corrected-input gradients through the second local evaluation.
- F_tilde_S: local forward work including adapter kernel construction.

Excluding common loss evaluation, assembly-related backward operations, optimizer operations, and activation-checkpoint recomputation:

```text
T_composition = F_S^(1) + F_C + F_S^(2)
                + B_S^{x,(2)} + B_C^phi

T_local_adapter = F_tilde_S^(1) + F_C + F_tilde_S^(2)
                  + B_S,A^(1) + B_C^z + B_S,A^{x,(2)}
```

Composition adaptation does not backpropagate through the first local evaluation and does not compute local weight gradients. It DOES backpropagate through the second frozen local evaluation: the composer must learn how its predicted input correction affects the final field.

Local-adapter adaptation must propagate gradients through both local evaluations and the frozen composer. The same adapter weights receive contributions from both calls. Our implementation constructs dense updated convolutional kernels before convolution, so its backward pass also differentiates those effective kernels. Equal trainable parameter counts do not imply equal backward work.

Both methods retain leading dependence on the local-model computation. These formulas support a difference in required backward operations, not a universal speedup factor or an asymptotically local-model-independent training cost.

## Memory and actual implementation

Gradient and Adam-moment storage is O(P_C) for composition adaptation and O(P_A) for the local-adapter arm. Full local fine-tuning would instead require O(P_S) gradient/optimizer storage. Because P_A approximately matches P_C here, optimizer-state storage is similar between the two ablation arms.

Total memory also includes model weights, temporary kernel gradients, and retained/recomputed activations. Composition adaptation requires the second local evaluation's input-gradient path; local adapters require both local paths. Peak memory cannot be inferred from trainable parameter count alone.

The experiment uses 64-tile microbatches and nonreentrant activation checkpointing where gradients are required, in both arms. Recomputed forwards add work beyond the expressions above. TF32 is disabled. Training uses bf16 autocast with fp32 master weights; evaluation uses fp32. Both arms use source-identical RMS forward values with a zero subgradient at exactly zero RMS to allow finite backward propagation through empty/zero faces.

Use `runs.csv` for final adaptation wall time and peak allocated GPU memory. Time includes initialization, training, validation, and checkpoint writes, excluding queue and test evaluation; memory is allocated CUDA bytes divided by 2²⁰. `screen_runs.csv` records screening costs separately. Neither file is an inference-speed benchmark, and partial runs are not completed timing observations.

## Implication for larger local physics models

If P_S increases while P_C remains fixed, the updated fraction P_C/(P_S + P_C) decreases and composition's gradient/optimizer storage remains unchanged. This motivates a large reusable local physics model with a compact adaptive composition module.

Forward work and the second local call's input-gradient work still depend on the large local model. Composition adaptation is therefore not O(P_C) total computation. Parameter-matched local adapters also keep optimizer storage small; the controlled distinction here is the location and extent of backward computation. The current P_S/P_C ratio is about 4.97, so benefits at foundation-model scale remain a forward-looking hypothesis rather than a demonstrated scaling law.

## Provenance and reporting scope

Counts were obtained by loading the exact source checkpoint pair used by the ablation, through its frozen `experiment.py` loader. `protocol.json` provides source/local checkpoint hashes; `implementation_freeze.json` identifies the implementation. No training settings or selection rules were changed to add this analysis.

This document accompanies, but does not replace, the final paired accuracy analysis. `status.json` must report completion, `test_errors.csv` must contain all 768 adapted test rows, and `zero_shot_errors.csv` must contain 128 reference rows before making the planned final comparison. `analysis.json`, generated on completion, reports paired effects and intervals. Interpret compute and accuracy jointly.
