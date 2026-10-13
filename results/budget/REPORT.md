# Darcy final-test analysis

64 reused test domains per scale; three paired adaptation seeds. Fixed source checkpoints.
Intervals are pointwise paired seed/domain bootstrap 95% intervals (20,000 draws).
64 target-validation simulations per scale are additional overhead.

| Scale | Training cases | Method | Mean relative L2 | 95% interval |
|---|---:|---|---:|---|
| 8×8 | 0 | LatentDDM | 0.121680 | [0.115203, 0.128457] |
| 8×8 | 0 | Spatial Patch FT | 0.284002 | [0.279589, 0.288477] |
| 8×8 | 4 | LatentDDM | 0.094314 | [0.089002, 0.100767] |
| 8×8 | 4 | Spatial Patch FT | 0.194088 | [0.185186, 0.203996] |
| 8×8 | 4 | Spatial Patch LoRA | 0.178709 | [0.173658, 0.184162] |
| 8×8 | 16 | LatentDDM | 0.077235 | [0.071860, 0.082996] |
| 8×8 | 16 | Spatial Patch FT | 0.146033 | [0.141468, 0.151407] |
| 8×8 | 16 | Spatial Patch LoRA | 0.144294 | [0.140033, 0.149096] |
| 8×8 | 64 | LatentDDM | 0.058918 | [0.055893, 0.062414] |
| 8×8 | 64 | Spatial Patch FT | 0.119945 | [0.117158, 0.122833] |
| 8×8 | 64 | Spatial Patch LoRA | 0.127915 | [0.124812, 0.131018] |
| 16×16 | 0 | LatentDDM | 0.126611 | [0.121871, 0.131859] |
| 16×16 | 0 | Spatial Patch FT | 0.532879 | [0.528408, 0.537592] |
| 16×16 | 4 | LatentDDM | 0.086583 | [0.079841, 0.095041] |
| 16×16 | 4 | Spatial Patch FT | 0.182432 | [0.177742, 0.188020] |
| 16×16 | 4 | Spatial Patch LoRA | 0.188484 | [0.182443, 0.194051] |
| 16×16 | 16 | LatentDDM | 0.066103 | [0.062499, 0.070228] |
| 16×16 | 16 | Spatial Patch FT | 0.146289 | [0.143871, 0.148701] |
| 16×16 | 16 | Spatial Patch LoRA | 0.160865 | [0.157420, 0.164422] |
| 16×16 | 64 | LatentDDM | 0.049408 | [0.047202, 0.051845] |
| 16×16 | 64 | Spatial Patch FT | 0.118564 | [0.116688, 0.120478] |
| 16×16 | 64 | Spatial Patch LoRA | 0.144476 | [0.142376, 0.146714] |

m0 Patch is shared by FT and LoRA and evaluated once per scale. Its interval resamples domains only.

| Scale | Training cases | Comparator minus LatentDDM | Mean difference | 95% interval |
|---|---:|---|---:|---|
| 8×8 | 4 | Spatial Patch FT | 0.099774 | [0.088848, 0.111480] |
| 8×8 | 4 | Spatial Patch LoRA | 0.084395 | [0.078413, 0.089118] |
| 8×8 | 16 | Spatial Patch FT | 0.068799 | [0.061249, 0.075514] |
| 8×8 | 16 | Spatial Patch LoRA | 0.067059 | [0.059131, 0.073825] |
| 8×8 | 64 | Spatial Patch FT | 0.061027 | [0.057460, 0.064519] |
| 8×8 | 64 | Spatial Patch LoRA | 0.068997 | [0.064425, 0.073241] |
| 16×16 | 4 | Spatial Patch FT | 0.095849 | [0.091198, 0.099923] |
| 16×16 | 4 | Spatial Patch LoRA | 0.101901 | [0.089843, 0.111097] |
| 16×16 | 16 | Spatial Patch FT | 0.080186 | [0.076895, 0.083243] |
| 16×16 | 16 | Spatial Patch LoRA | 0.094762 | [0.089479, 0.099508] |
| 16×16 | 64 | Spatial Patch FT | 0.069156 | [0.066638, 0.071465] |
| 16×16 | 64 | Spatial Patch LoRA | 0.095068 | [0.092429, 0.097753] |

The predeclared low-training-budget criterion is met.

Three seeds provide limited uncertainty resolution; source-pretraining variability is unmeasured.
The comparison is between complete source recipes. The test sets were historically inspected.
Process timing includes startup, training, validation, and checkpoint I/O; see the hardware ledger.
No source-cost matching, total-label-budget, controlled hardware speedup, or interpolated label-savings claim is made.

Manuscript review proposals and visual inspection of generated figures remain separate tasks.
