# ViewCrafter + DiFix-fused As-real Pseudo-view Pipeline

This note records the currently best pseudo-view augmentation route we tested on
`DL3DV-2`: generate pseudo views with ViewCrafter, enhance/fuse them with DiFix,
then treat the fused pseudo RGB as normal training views in InstantSplat.

Current best result among these pseudo-view variants:

| Experiment | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| 10 real baseline | 16.143 | 0.408 | 0.406 |
| raw ViewCrafter pseudo as real | 20.154 | 0.614 | 0.333 |
| **DiFix-fused pseudo as real** | **20.026** | **0.613** | **0.291** |
| DiFix-fused pseudo masked | 20.166 | 0.594 | 0.349 |
| 60 real reference | 21.663 | 0.660 | 0.256 |

The best practical choice for now is **DiFix-fused pseudo as real** because its
PSNR/SSIM are close to raw pseudo-as-real, while LPIPS improves clearly. The
current hard/continuous masked-loss version is not recommended as default.

## 0. Machines And Paths

InstantSplat server:

```bash
ssh -p 49025 -i ~/.ssh/id_ed25519_3dgs root@123.127.15.155
source /root/miniconda3/etc/profile.d/conda.sh
conda activate instantsplat
cd /root/autodl-tmp/AIAA3201_3DGS_Project/third_party/InstantSplat
```

ViewCrafter can be run on a separate server. Exchange data through `/root/autodl-fs`.

Important paths from the DL3DV-2 run:

```bash
REAL_SOURCE=/root/autodl-fs/DL3DV-2
VC_EXP=/root/autodl-fs/DL3DV-2-exp/DL3DV-10views-reconx-viewcrafter45-512-steps10
FUSION_DIR=$VC_EXP/difix_256_schemeA/fusion_stage0_v1
ASREAL_SOURCE=/root/autodl-fs/DL3DV-2-difix-fused-asreal
MODEL_OUT=/root/autodl-fs/DL3DV-2-exp/DL3DV-55views-difix-fused-asreal-stage0-30k-3dgslr
```

## 1. Split Real Views

For this experiment:

- Dataset: `DL3DV-2`
- Real sparse input: `10` training views
- Held-out test views: `12`
- Real train views are sampled in InstantSplat's explicit split logic.
- Test views are real frames only. Pseudo views are not used as GT for evaluation.

The train/test split should be kept fixed across ablations. Otherwise the result
is not comparable.

## 2. Generate Pseudo Views With ViewCrafter

Current setting:

| Parameter | Value Used | Notes |
| --- | ---: | --- |
| real train views | 10 | Sparse input frames |
| adjacent real pairs | 9 | `n_real - 1` |
| generated frames per pair | 25 | ViewCrafter generation length |
| selected pseudo frames per pair | 5 | Main knob; can be raised/lowered |
| total pseudo views | 45 | `9 * 5` |
| final train views | 55 | `10 real + 45 pseudo` |
| generation resolution | 512x512 | Used in this DL3DV-2 run |
| final InstantSplat RGB resolution | 256x256 | Match the actual training images |
| ViewCrafter steps | 10 | From `...512-steps10` experiment name |

Recommended pseudo count formula:

```text
pseudo_total = (n_real - 1) * pseudo_per_pair
n_train_for_pipeline = n_real + pseudo_total
```

For example:

| `pseudo_per_pair` | Total pseudo with 10 real views | Final train views |
| ---: | ---: | ---: |
| 3 | 27 | 37 |
| **5** | **45** | **55** |
| 7 | 63 | 73 |
| 10 | 90 | 100 |

Generation principle:

1. Use adjacent real-view pairs, not the whole sparse sequence at once.
2. For each pair, let ViewCrafter generate a short video/interpolation sequence.
3. Select evenly spaced intermediate frames from each pair.
4. Keep the pseudo frames temporally ordered between their two real endpoints.

This ordering matters because MASt3R/InstantSplat scene graph construction uses
ordered image lists. Use zero-padded names or an explicit split manifest so that
the real/pseudo order is unambiguous.

Example ordered layout:

```text
000001_real_frame_00001.png
000002_pseudo_00001_00036_01.png
000003_pseudo_00001_00036_02.png
000004_pseudo_00001_00036_03.png
000005_pseudo_00001_00036_04.png
000006_pseudo_00001_00036_05.png
000007_real_frame_00036.png
...
```

Do not rely on names like `real_00001`, `pseudo_00001_to_00036_01`,
`real_00036` unless the builder explicitly sorts by numeric order in the
manifest.

## 3. DiFix Enhancement And Stage-0 Fusion

The raw ViewCrafter pseudo views are smoother and may lose details. We therefore
run DiFix on the pseudo RGBs before giving them to InstantSplat.

DiFix setting used in this run:

```text
model/pipeline: nvidia/difix_ref
prompt: "remove degradation"
num_inference_steps: 1
timestep: 199
guidance_scale: 0.0
output resolution for this experiment: 256x256
```

The stage-0 fusion output directory was:

```bash
/root/autodl-fs/DL3DV-2-exp/DL3DV-10views-reconx-viewcrafter45-512-steps10/difix_256_schemeA/fusion_stage0_v1
```

Expected files:

```text
fusion_stage0_v1/
  images/                         # fused pseudo RGB, used by best as-real run
  masks/                          # continuous confidence masks, not used in best run
  masks_trinary/                  # trinary masks, experimental
  pseudo_manifest_fused_nomask.json
  pseudo_manifest_fused_continuous.json
  pseudo_manifest_fused_trinary.json
  fusion_stats.json
```

Observed fusion stats in the DL3DV-2 run:

```text
num_views = 45
continuous_mask_mean_mean = 0.40268
trinary_mask_mean_mean = 0.46147
agreement_mean_mean = 0.67215
```

For the current best run, use:

```text
fusion_stage0_v1/images
```

Do **not** use `masks` or `masks_trinary` for the default as-real experiment.
The masked version was tested and did not improve the final reconstruction.

## 4. Build 55-view As-real InstantSplat Source

Create a new source directory that contains:

```text
DL3DV-2-difix-fused-asreal/
  images/                         # 10 real + 45 DiFix-fused pseudo images
  explicit_split_manifest.json    # marks train/test and real/pseudo provenance
  sparse_55/0/                    # generated by init_geo.py
  sparse_55/1/                    # test pose init if produced
```

The pseudo RGB should come from:

```bash
$FUSION_DIR/images
```

Use the no-mask manifest:

```bash
$FUSION_DIR/pseudo_manifest_fused_nomask.json
```

The source used for the successful run:

```bash
/root/autodl-fs/DL3DV-2-difix-fused-asreal
```

The builder script used in the current codebase is the pseudo-as-real source
builder, referred to in our notes as:

```bash
tools/build_pseudo_as_real_split.py
```

Its job is:

1. Copy or symlink the selected 10 real train images.
2. Insert 45 pseudo images between the corresponding real frames.
3. Keep the original 12 real test frames as test-only entries.
4. Write `explicit_split_manifest.json`.
5. Ensure pseudo entries have no `mask_path` for the as-real ablation.

If the exact CLI differs in the checked-out code, run:

```bash
/root/miniconda3/envs/instantsplat/bin/python tools/build_pseudo_as_real_split.py -h
```

The critical output contract is more important than the exact parameter names:

```json
{
  "train": [
    {"image_name": "...", "kind": "real"},
    {"image_name": "...", "kind": "pseudo", "mask_path": null}
  ],
  "test": [
    {"image_name": "frame_00112.png", "kind": "real"}
  ]
}
```

## 5. Run InstantSplat Training And Evaluation

Run from InstantSplat root:

```bash
cd /root/autodl-tmp/AIAA3201_3DGS_Project/third_party/InstantSplat

/root/miniconda3/envs/instantsplat/bin/python run_eval_pipeline.py \
  -s /root/autodl-fs/DL3DV-2-difix-fused-asreal \
  -m /root/autodl-fs/DL3DV-2-exp/DL3DV-55views-difix-fused-asreal-stage0-30k-3dgslr \
  --n_train 55 \
  --n_test 12 \
  --split_manifest /root/autodl-fs/DL3DV-2-difix-fused-asreal/explicit_split_manifest.json \
  -i 30000 \
  --train_test_iterations 7000 15000 30000 \
  --train_save_iterations 7000 15000 30000 \
  --render_iterations 7000 15000 30000 \
  -r 1 \
  --scene_graph logwin-3-noncyclic \
  --max_init_points 100000 \
  --sampling_grid_size 24 \
  --min_point_distance_px 12 \
  --point_conf_threshold 0 \
  --no_pp_optimizer \
  --no_optim_pose \
  --use_densification \
  --position_lr_init 0.00016 \
  --position_lr_final 0.0000016 \
  --feature_lr 0.00025 \
  --opacity_lr 0.025 \
  --scaling_lr 0.0005 \
  --rotation_lr 0.0001 \
  --percent_dense 0.01 \
  --densification_interval 100 \
  --opacity_reset_interval 3000 \
  --densify_from_iter 500 \
  --densify_until_iter 15000 \
  --densify_grad_threshold 0.0002 \
  --optim_test_pose_iter 500
```

## 6. Important Parameters

### Pseudo-view parameters

| Parameter | Current value | Effect |
| --- | ---: | --- |
| `n_real` | 10 | Number of real sparse training views |
| `pseudo_per_pair` | 5 | How many pseudo frames are selected between adjacent real views |
| ViewCrafter frames per pair | 25 | Number generated before selecting subset |
| final pseudo count | 45 | `(10 - 1) * 5` |
| final `--n_train` | 55 | `10 real + 45 pseudo` |

`pseudo_per_pair` is the main knob for future runs. If it changes, update
`--n_train` and rebuild the as-real source.

### MASt3R / init_geo parameters

| Parameter | Current value | Effect |
| --- | ---: | --- |
| `--scene_graph` | `logwin-3-noncyclic` | Pair graph for MASt3R global alignment |
| `--max_init_points` | `100000` | Cap before later filtering/saving |
| `--point_sampling` | default `grid_uniform_confidence` | Spatially spread high-confidence points |
| `--sampling_grid_size` | `24` | Larger means coarser grid and more uniform spreading |
| `--min_point_distance_px` | `12` | Rejects points too close in image space |
| `--point_conf_threshold` | `0` | Keep all confidence values before sampling |

Observed in the successful DL3DV-2 run:

```text
Number of points: 10813440
Number of points after downsampling: 49412
```

### Training / densification parameters

| Parameter | Current value |
| --- | ---: |
| iterations | `30000` |
| `--use_densification` | enabled |
| `--position_lr_init` | `0.00016` |
| `--position_lr_final` | `0.0000016` |
| `--feature_lr` | `0.00025` |
| `--opacity_lr` | `0.025` |
| `--scaling_lr` | `0.0005` |
| `--rotation_lr` | `0.0001` |
| `--percent_dense` | `0.01` |
| `--densification_interval` | `100` |
| `--opacity_reset_interval` | `3000` |
| `--densify_from_iter` | `500` |
| `--densify_until_iter` | `15000` |
| `--densify_grad_threshold` | `0.0002` |

These are the 3DGS-effective learning-rate/densification settings we used for
the fair comparison with the 60-real-view run.

### Pose parameters

| Parameter | Current value | Notes |
| --- | ---: | --- |
| `--no_pp_optimizer` | enabled | Disable point/pose post-optimizer |
| `--no_optim_pose` | enabled | Do not optimize training poses during 3DGS training |
| `--optim_test_pose_iter` | `500` | Optimize test pose before rendering/evaluation |

For comparable reported metrics, keep `--optim_test_pose_iter 500`.

## 7. Outputs To Check

Main result files:

```text
$MODEL_OUT/results.json
$MODEL_OUT/pipeline.log
$MODEL_OUT/test/ours_30000/metrics.txt
$MODEL_OUT/test/ours_30000/renders/
$MODEL_OUT/test/ours_30000/gt/
$MODEL_OUT/train/ours_30000/renders/
$MODEL_OUT/point_cloud/iteration_30000/point_cloud.ply
```

Expected `results.json` for the successful DL3DV-2 run:

```json
{
  "ours_7000": {
    "SSIM": 0.6027138829231262,
    "PSNR": 19.482418060302734,
    "LPIPS": 0.3625962734222412,
    "ATE": 0.09076211224208958
  },
  "ours_15000": {
    "SSIM": 0.6120865941047668,
    "PSNR": 19.795995712280273,
    "LPIPS": 0.3168555200099945,
    "ATE": 0.09076211224208958
  },
  "ours_30000": {
    "SSIM": 0.6126300096511841,
    "PSNR": 20.025781631469727,
    "LPIPS": 0.29094281792640686,
    "ATE": 0.09076211224208958
  }
}
```

Training curve observed:

```text
[ITER 5000]  train L1 0.046623 PSNR 22.1919
[ITER 7000]  train L1 0.041405 PSNR 23.1818
[ITER 10000] train L1 0.035267 PSNR 24.5382
[ITER 15000] train L1 0.028288 PSNR 26.3831
[ITER 20000] train L1 0.024265 PSNR 27.7292
[ITER 25000] train L1 0.022291 PSNR 28.4373
[ITER 30000] train L1 0.021114 PSNR 29.0220
```

Important test view `frame_00112.png`:

```text
ours_7000:  PSNR 21.45, SSIM 0.6761, LPIPS 0.2915
ours_15000: PSNR 21.20, SSIM 0.6574, LPIPS 0.2944
ours_30000: PSNR 21.08, SSIM 0.6595, LPIPS 0.2683
```

## 8. Code Files Involved

Core InstantSplat pipeline:

```text
run_eval_pipeline.py
init_geo.py
train.py
render.py
metrics.py
```

MASt3R/COLMAP conversion and point filtering:

```text
utils/sfm_utils.py
scene/dataset_readers.py
utils/camera_utils.py
scene/gaussian_model.py
arguments/__init__.py
```

Pseudo-view tools used or relevant in this branch:

```text
tools/build_pseudo_as_real_split.py        # build 10 real + pseudo as-real source
tools/enhance_pseudo_with_difix.py         # earlier DiFix enhancement route
tools/build_pseudo_masks.py                # mask route, not default
tools/build_feature_correspondence_masks.py# future confidence-mask route
tools/visualize_pseudo_mask_contact_sheet.py
```

Local analysis/visualization artifacts from this run:

```text
/Users/q/Documents/Codex/2026-05-12/ssh-p-49025-i-ssh-id/difix_stage0_compare/
  compare_frame_00112.png
  compare_all12_ours30000.png
```

## 9. What To Change For Other Datasets

For `Re10K-1`, `Waymo-405841`, or another DL3DV scene:

1. Change `REAL_SOURCE`.
2. Recreate the same 10-real train / test split.
3. Run ViewCrafter on adjacent real-view pairs.
4. Choose `pseudo_per_pair`.
5. Resize final pseudo RGBs to the actual InstantSplat training image resolution.
6. Run DiFix and stage-0 fusion.
7. Build the as-real source.
8. Set `--n_train = n_real + (n_real - 1) * pseudo_per_pair`.
9. Keep the same 3DGS LR/densification settings for the first comparable run.

Recommended first sweep:

```text
pseudo_per_pair = 3, 5, 7
max_init_points = 100000
iterations = 30000
optim_test_pose_iter = 500
```

Only after the as-real route is stable should masks be reintroduced.

## 10. Current Lessons

1. ViewCrafter pseudo views help substantially compared with 10 real views.
2. DiFix-fused RGB improves perceptual quality (`LPIPS`) more than raw pseudo RGB.
3. Current hard/continuous masked pseudo loss is too conservative and hurts some
   views such as `frame_00112`.
4. The next mask direction should be soft confidence weighting, not binary
   suppression.
5. If pseudo views participate in MASt3R initialization, filename/manifest order
   must be correct; otherwise `logwin-3-noncyclic` can connect wrong neighbors.

