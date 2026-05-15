# Final Reproduction Guide

This document is the step-by-step recipe for reproducing the final reported
method: real views initialize geometry, ViewCrafter + DiFix3D+ generate pseudo
RGB supervision, and MASt3R/DUSt3R confidence masks down-weight unreliable
pseudo regions during InstantSplat optimization.

## 1. Expected Inputs and Outputs

Real dataset layout:

```text
DATASET_ROOT/
  images/
    frame_00001.png
    frame_00036.png
    ...
```

Pseudo-view stage output expected by this repository:

```text
PSEUDO_ROOT/
  images/                                      # DiFix-fused pseudo RGBs
  pseudo_manifest_fused_nomask_external.json  # pseudo RGB + interpolated pose/time
```

Confidence-mask stage output:

```text
CONF_ROOT/
  masks/
  confidence_vis/
  confidence_stats.json
  pseudo_manifest_reconx_dust3r_confidence.json
```

InstantSplat final output:

```text
MODEL_OUT/
  pipeline.log
  results.json
  train/ours_5000/
  test/ours_5000/
  point_cloud/iteration_5000/
```

## 2. Sparse Real-View Split

Use a fixed split for all ablations on the same dataset.

Report settings:

| Dataset | Real train views | Test views | Resolution flag |
| --- | ---: | ---: | ---: |
| DL3DV-2 | 10 | 12 | `-r 1` |
| Re10k-1 | 9 | 12 | `-r 1` |
| Waymo-405841 FRONT | 20 | 12 | `-r 4` |

If an explicit split manifest is available, always pass it with:

```bash
--split_manifest /path/to/*-explicit-train-test.json
```

If it is not available, omit `--split_manifest` and let `run_eval_pipeline.py`
deterministically sample train/test images from sorted image names. Save the
generated `<MODEL_OUT>/split_manifest.json` and reuse it for all ablations.

## 3. Generate Pseudo Views With ViewCrafter

Run ViewCrafter on adjacent sparse real-view pairs. Do not feed the whole long
sequence at once.

Final pseudo-view sampling policy:

```text
pseudo_per_pair = 5
pseudo_total = (n_real - 1) * pseudo_per_pair
```

Examples:

| Dataset | Real views | Real-view pairs | Pseudo per pair | Total pseudo |
| --- | ---: | ---: | ---: | ---: |
| DL3DV-2 | 10 | 9 | 5 | 45 |
| Re10k-1 | 9 | 8 | 5 | 40 |
| Waymo-405841 FRONT | 20 | 19 | 5 | 95 |

Keep pseudo views ordered between their two real endpoints. The pseudo manifest
must record each pseudo frame's left/right real references, interpolated time,
image path, and pose.

## 4. Enhance Pseudo Views With DiFix3D+

After raw ViewCrafter generation, run DiFix3D+ in reference-conditioned mode.
The project script expects DiFix3D's `src` directory through `--difix_repo`.

```bash
python tools/enhance_pseudo_with_difix.py \
  --pseudo_manifest /path/to/pseudo_manifest_raw.json \
  --source_images_dir /path/to/DATASET_ROOT/images \
  --output_dir /path/to/difix_outputs \
  --difix_repo /path/to/Difix3D/src \
  --model_id nvidia/difix_ref \
  --prompt "remove degradation" \
  --num_inference_steps 1 \
  --timestep 199 \
  --guidance_scale 0.0
```

The script writes:

```text
difix_outputs/
  enhanced_left/
  enhanced_right/
  pseudo_manifest_enhanced.json
```

Then fuse the left/right candidates into the RGBs used for training. In our
final experiments, the fused pseudo RGBs are treated as pseudo supervision only;
they do not seed geometry and do not trigger densification.

## 5. Build Cross-View Confidence Masks

Use MASt3R/DUSt3R matching to estimate where each pseudo view agrees with its
neighboring real frames.

```bash
python tools/build_reconx_dust3r_confidence.py \
  --pseudo_manifest /path/to/pseudo_manifest_fused_nomask_external.json \
  --source_images_dir /path/to/DATASET_ROOT/images \
  --output_dir /path/to/CONF_ROOT \
  --image_size 512 \
  --batch_size 1 \
  --bidirectional_pairs \
  --left_right_weight 1.0 \
  --neighbor_weight 0.75 \
  --all_pair_weight 0.5 \
  --aggregate weighted_mean \
  --mask_blur 1.0
```

Expected log line:

```text
[ReconX-mask] mask mean: min=..., mean=..., max=...
```

The output manifest for final training is:

```text
/path/to/CONF_ROOT/pseudo_manifest_reconx_dust3r_confidence.json
```

## 6. Final Training Logic

The final method is intentionally conservative about geometry:

- real views only initialize the Gaussian scene;
- no initialization point filtering cap is used: `--max_init_points 0`;
- no 3DGS densification is enabled;
- pseudo views only add confidence-weighted supervision;
- pseudo views do not update densification statistics.

For each iteration, the real-view photometric loss is still the primary 3DGS
loss. When pseudo supervision is sampled, the pseudo loss is:

```text
L_pseudo = mean(w * (0.8 L1 + 0.2 SSIM + 0.5 LPIPS)) / mean(w)
w = (0.25 + 0.75 c)^0.5
```

where `c` is the MASt3R/DUSt3R confidence mask. The soft floor is important:
raw confidence underweights sky, distant buildings, and low-texture walls too
aggressively, which can hurt background reconstruction.

## 7. Final Commands

DL3DV-2:

```bash
python run_eval_pipeline.py \
  -s /path/to/DL3DV-2 \
  -m /path/to/outputs/DL3DV-2-final-softfloor-5k \
  --n_train 10 \
  --n_test 12 \
  --split_manifest /path/to/splits/DL3DV-2-10real-explicit-train-test.json \
  -i 5000 \
  -r 1 \
  --scene_graph logwin-3-noncyclic \
  --max_init_points 0 \
  --point_conf_threshold 0 \
  --optim_test_pose_iter 500 \
  --use_pseudo_views \
  --pseudo_manifest /path/to/CONF_ROOT/pseudo_manifest_reconx_dust3r_confidence.json \
  --pseudo_start_iter 1 \
  --pseudo_ramp_until 1 \
  --pseudo_loss_weight 1.0 \
  --pseudo_sample_ratio 1.0 \
  --pseudo_pair_with_real \
  --pseudo_rgb_weight 0.8 \
  --pseudo_ssim_weight 0.2 \
  --pseudo_lpips_weight 0.5 \
  --pseudo_lpips_net vgg \
  --pseudo_confidence_floor 0.25 \
  --pseudo_mask_gamma 0.5
```

Re10k-1 uses the same command with `--n_train 9`.

Waymo-405841 FRONT uses the same command with `--n_train 20`, `-r 4`, and:

```bash
--position_lr_init 0.000016 \
--position_lr_final 0.00000016 \
--scaling_lr 0.0005
```

## 8. What to Check After a Run

Check the training log:

```bash
tail -n 80 /path/to/outputs/*/pipeline.log
```

Check final metrics:

```bash
cat /path/to/outputs/*/results.json
```

Check rendered images:

```text
MODEL_OUT/test/ours_5000/renders/
MODEL_OUT/test/ours_5000/gt/
MODEL_OUT/test/ours_5000/metrics.txt
```

For DL3DV-2, visually inspect the held-out view around `frame_00112.png`; it was
one of the clearest examples where the softfloor confidence improved background
stability compared with raw confidence masking.
