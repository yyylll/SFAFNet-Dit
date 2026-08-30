# SFAFNet-DiT

Reference implementation of the core modules of the two-stage SFAFNet-DiT framework, provided to
support peer review. Reviewers can read and run the novel components. The full training and
evaluation pipeline, configurations, and dataset will be released upon acceptance.

## Contents
- `models/` — SFAF block, `DecompSFAFNetTF` (Stage-1 total-first dual-decoder), `PhysicsDiT`
  (Stage-2 decomposition-conditioned refiner), and the physics-prior encoder.
- `model.py` — public import surface.
- `physics_prior.py` — physics operators (Helmholtz curvature, interference field).
- `physics_prior_encoder.py` — physics-prior (M_phy) assembly.
- `losses_tf.py`, `losses.py` — the PCDL objective (power-consistent dual-decoder) and the
  physical-consistency (gradient/curvature) losses.
- `cagrad.py` — conflict-averse gradient balancing (Liu et al., NeurIPS 2021).
- `demo_forward.py` — minimal self-contained two-stage forward pass.

## Run the demo
```
pip install -r requirements.txt
python demo_forward.py
```
Runs one forward pass of both stages on random tensors and prints the output shapes. No dataset or
training loop is required.

## Not included (released upon acceptance)
Training and evaluation scripts, the data pipeline, tuned configurations, and the dataset.

## License & Complete Architecture Design & Data
Provided for peer-review evaluation only; see LICENSE.
### Layout

```
refiner_SFAFNet_DiT/
├── model.py                 # facade: re-exports every public model class from models/
├── models/                  # the model subpackage (self-contained; torch-only)
│   ├── __init__.py
│   ├── blocks.py            # ConvBNAct, ChannelAttention, REMHead, InterferenceLocalizationHead,
│   │                        #   peak_nms, SFAFBlock, ResBlock, ResCompress, _TransformerBlock,
│   │                        #   ConditionEncoder, ZeroInitProjection, _safe_groups
│   ├── sfafnet.py           # SharedEncoder, SFAFDecoder, RHead, DecompHeadsTF, DecompSFAFNetTF (TF Stage-1)
│   ├── physics_dit.py       # TimeEmbed, DiTBlock, PhysicsDiT, DiffusionSchedule (Stage-2)
│   └── prior_encoder.py     # PhysicsPriorEncoder
├── physics_prior.py         # I_field  / helmholtz_k2  / ... (vendored)
├── physics_prior_encoder.py # E_φ bridge: assemble_decomp_mphy (renders I_field then applies E_φ)
├── physics_dit.py           # thin shim -> models.physics_dit (so util modules' `import physics_dit` works)
├── refiner.py               # _stage5_conditioning, refiner_train_step, refiner_infer, DPS/σ-guidance
├── train_refiner.py         # Stage-2 training + viz + NPZ; build_sfaf loads frozen TF Stage-1
├── evaluate_refiner.py      # test-set metrics table {NMSE,RMSE,SSIM,PSNR} across sampling rates -> JSON
├── train_tf.py              # Stage-1 training: total-first DecompSFAFNetTF (SFAF block) + CAGrad
├── losses_tf.py             # TotalFirstLoss = L_R + DecompLoss + lambda_cons consistency anchor
├── cagrad.py                # CAGrad conflict-averse gradient (two task groups on the shared encoder)
├── stage1_train_helpers.py  # S_prior + periodic-trigger + viz-prior helpers (self-contained)
├── train_utils.py           # get_logger / make_loaders / cooperative_tx_from_map (extracted)
├── losses.py, metrics.py, dataset_adapter.py, instrumentation.py, neural_propagator.py  # vendored infra
├── requirements.txt
└── test_*.py                # self-containment, TF strict-load, E_φ golden regression, eval-table
```


### Data

Both trainers consume an HDF5 dataset via `dataset_adapter.py` (paths passed with `--h5`, `--split`,
`--norm`, `--l_ray`). Each batch (see `dataset_adapter.SFAFNetDiTDataset` + `collate_sfafnet_dit`)
provides: `sparse_obs`, `sparse_mask` (RSS samples + mask), `building_mask`, `tx_feature_map`
(cooperative-GBS presence + power), the dense target `rem`, the decomposition targets `gt_signal`,
`gt_interf`, `gt_sinr`, `gt_interference_positions`, and the optional cached free-space prior `s_prior`
(from `--l_ray`). Supply your own data in this format, or use the released dataset. `--split` is a JSON of
train/val scene ids and `--norm` a JSON of RSS `{center, scale}` normalization stats.

