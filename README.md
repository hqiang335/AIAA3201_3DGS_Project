# AIAA3201 Generative Sparse-View 3D Reconstruction

This is the public code repository for our AIAA3201 computer-vision course project on **generative sparse-view, unposed 3D Gaussian reconstruction**. The final system extends InstantSplat with ViewCrafter/DiFix pseudo views and cross-view confidence weighting, while preserving real-view-only geometry initialization.

**Core idea.** Sparse real views are used to estimate camera poses and initialize the Gaussian scene. Generated pseudo views are used only as confidence-weighted photometric supervision; they do **not** initialize geometry and they do **not** trigger densification in the final setting.

![Final system overview](docs/assets/final_system_overview.png)

## What Is Included

```text
AIAA3201_3DGS_Project/
  Part1_Scripts/                         # posed 3DGS / VGGT initialization studies
  third_party/
    gaussian-splatting/                  # original 3DGS baseline
    InstantSplat/                        # main codebase modified for this project
      run_eval_pipeline.py               # train/test pipeline wrapper
      init_geo.py                        # MASt3R/Dust3R geometry and split initialization
      train.py                           # pseudo-view confidence-weighted training
      scene/, arguments/, utils/          # dataset/camera/pipeline extensions
      tools/
        build_reconx_dust3r_confidence.py
        build_soft_confidence_masks.py
        build_pseudo_as_real_split.py
        filter_pseudo_manifest.py
        repair_test_poses_timeaware.py
        score_pseudo_temporal_consistency.py
      docs/
        FINAL_REPRODUCTION.md
        VIEWCRAFTER_DIFIX_FUSED_ASREAL_PIPELINE.md
```

The repository intentionally does **not** include generated datasets, model checkpoints, experiment outputs, or third-party weight files.

## Dependencies

The main reconstruction code was run on Linux with CUDA GPUs. We used a Conda environment named `instantsplat`.

```bash
git clone --recursive https://github.com/hqiang335/AIAA3201_3DGS_Project.git
cd AIAA3201_3DGS_Project/third_party/InstantSplat

conda create -n instantsplat python=3.10.13 cmake=3.14.0 -y
conda activate instantsplat

# Pick the CUDA build matching your machine. This is the version family we used.
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install -r requirements.txt
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
pip install submodules/fused-ssim

# Extra packages used by pseudo-view supervision / perceptual losses.
pip install lpips imageio imageio-ffmpeg

# Optional DiFix dependencies if you run DiFix enhancement on the same machine.
pip install -r tools/requirements_difix.txt
```

Optional CUDA acceleration for MASt3R/DUSt3R positional embeddings:

```bash
cd croco/models/curope
python setup.py build_ext --inplace
cd ../../../
```

## Reproducibility Checklist

To reproduce the reported experiments from a fresh machine, prepare these items in order:

1. Clone this repository with submodules.
2. Install the `instantsplat` environment and compile the Gaussian rasterizer extensions.
3. Download the MASt3R checkpoint used by InstantSplat geometry initialization.
4. Prepare real datasets with an `images/` directory.
5. Generate ViewCrafter pseudo views between adjacent sparse real-view pairs.
6. Enhance pseudo views with DiFix3D+.
7. Build MASt3R/DUSt3R cross-view confidence masks.
8. Run `run_eval_pipeline.py` with `--use_pseudo_views`, `--pseudo_confidence_floor 0.25`, and `--pseudo_mask_gamma 0.5`.

Quick sanity checks before launching long runs:

```bash
cd AIAA3201_3DGS_Project/third_party/InstantSplat
python run_eval_pipeline.py --help
python tools/build_pseudo_as_real_split.py --help

test -f mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
```

## Required Weights

The repository does not store large weights. Use the copy-paste commands below.

### InstantSplat Geometry: MASt3R

This checkpoint is required by `init_geo.py` and `tools/build_reconx_dust3r_confidence.py`.

```bash
cd AIAA3201_3DGS_Project/third_party/InstantSplat
mkdir -p mast3r/checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  -O mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
```

### ViewCrafter Pseudo-View Generation

ViewCrafter is run as a separate project, often on a separate GPU server. Official repository:

```text
https://github.com/Drexubery/ViewCrafter
```

Recommended setup:

```bash
git clone https://github.com/Drexubery/ViewCrafter.git
cd ViewCrafter
conda create -n viewcrafter python=3.9.16 -y
conda activate viewcrafter
pip install -r requirements.txt
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py39_cu117_pyt1131.tar.bz2 -y
```

ViewCrafter also expects the original DUSt3R checkpoint:

```bash
mkdir -p checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  -O checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
```

Download one ViewCrafter checkpoint. We used the smaller 512 checkpoint for the main DL3DV-2 pseudo-view generation because it fits more easily on 24GB GPUs:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Drexubery/ViewCrafter_25_512 model.ckpt \
  --local-dir checkpoints \
  --local-dir-use-symlinks False
```

If you have more memory and want the sparse-view-specific model:

```bash
huggingface-cli download Drexubery/ViewCrafter_25_sparse model_sparse.ckpt \
  --local-dir checkpoints \
  --local-dir-use-symlinks False
```

Model pages:

```text
https://huggingface.co/Drexubery/ViewCrafter_25_512
https://huggingface.co/Drexubery/ViewCrafter_25_sparse
```

### DiFix3D+ Pseudo-View Enhancement

Official DiFix3D+ repository and model:

```text
https://github.com/nv-tlabs/Difix3D
https://huggingface.co/nvidia/difix_ref
```

Install beside this repository or pass its location with `--difix_repo`:

```bash
cd AIAA3201_3DGS_Project/third_party
git clone https://github.com/nv-tlabs/Difix3D.git

cd AIAA3201_3DGS_Project/third_party/InstantSplat
conda activate instantsplat
pip install -r tools/requirements_difix.txt
pip install -U diffusers transformers accelerate huggingface_hub

# Optional: pre-download the referenced DiFix3D+ weights into the HF cache.
huggingface-cli download nvidia/difix_ref --local-dir /path/to/hf-cache/nvidia/difix_ref
```

The DiFix script default is:

```text
model_id = nvidia/difix_ref
prompt = "remove degradation"
num_inference_steps = 1
timestep = 199
guidance_scale = 0.0
```

### Summary of External Components

| Component | Purpose | Weight / model identifier |
| --- | --- | --- |
| MASt3R | InstantSplat pose / pointmap / confidence matching | `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` |
| DUSt3R | ViewCrafter internal sparse reconstruction dependency | `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` |
| ViewCrafter | interpolation between adjacent sparse real views | `Drexubery/ViewCrafter_25_512` or `Drexubery/ViewCrafter_25_sparse` |
| DiFix3D+ | pseudo-view restoration / sharpening | `nvidia/difix_ref` |
| LPIPS | perceptual pseudo-view loss | downloaded automatically by `lpips` |

Datasets and generated pseudo views are not committed. Keep them under a large-data directory such as `/root/autodl-fs`.

## Data Layout

Each real dataset should contain an `images` folder:

```text
/path/to/DL3DV-2/
  images/
    frame_00001.png
    frame_00036.png
    ...
```

Pseudo-view supervision uses a JSON manifest produced by the pseudo-view generation pipeline. Each pseudo entry points to a pseudo RGB image, an interpolated pose/time, and optionally a confidence mask. The training code loads this manifest through `--pseudo_manifest`.

## Run the Real-Only InstantSplat Baseline

This runs sparse unposed reconstruction with real views only:

```bash
cd AIAA3201_3DGS_Project/third_party/InstantSplat
conda activate instantsplat

python run_eval_pipeline.py \
  -s /path/to/DL3DV-2 \
  -m /path/to/outputs/DL3DV-2-10real-baseline-5k \
  --n_train 10 \
  --n_test 12 \
  -i 5000 \
  -r 1 \
  --scene_graph logwin-3-noncyclic \
  --max_init_points 0 \
  --point_conf_threshold 0 \
  --optim_test_pose_iter 500
```

Outputs are written under the model directory:

```text
model_path/
  pipeline.log
  results.json
  train/ours_5000/
  test/ours_5000/
  point_cloud/iteration_5000/
```

## Generate Pseudo Views

Our final experiments generate pseudo views between adjacent sparse real-view pairs:

1. select sparse real views;
2. for every adjacent pair, run ViewCrafter interpolation;
3. sample intermediate frames, e.g. 5 per real-view pair;
4. enhance pseudo RGBs with DiFix3D+;
5. build a pseudo manifest for InstantSplat training.

For the DL3DV-2 setting in the report:

| Setting | Value |
| --- | ---: |
| sparse real views | 10 |
| adjacent windows | 9 |
| sampled pseudo views per window | 5 |
| total pseudo views | 45 |
| training iterations | 5000 |

See the detailed operational note:

```text
third_party/InstantSplat/docs/FINAL_REPRODUCTION.md
```

## Build MASt3R/DUSt3R Confidence Masks

After pseudo RGBs are generated and enhanced, build cross-view confidence maps:

```bash
python tools/build_reconx_dust3r_confidence.py \
  --pseudo_manifest /path/to/pseudo_manifest_fused_nomask_external.json \
  --source_images_dir /path/to/DL3DV-2/images \
  --output_dir /path/to/DL3DV-2-reconx-dust3r-confidence \
  --image_size 512 \
  --batch_size 1 \
  --bidirectional_pairs \
  --left_right_weight 1.0 \
  --neighbor_weight 0.75 \
  --all_pair_weight 0.5 \
  --aggregate weighted_mean \
  --mask_blur 1.0
```

This produces a new pseudo manifest whose entries include confidence masks. The final training remaps confidence `c` into a background-preserving soft weight:

```text
w = (0.25 + 0.75 c)^0.5
```

This down-weights unreliable generated regions while keeping low-texture backgrounds from disappearing entirely.

## Run the Final Method

Final setting: real views initialize geometry; pseudo views provide weighted supervision.

```bash
python run_eval_pipeline.py \
  -s /path/to/DL3DV-2 \
  -m /path/to/outputs/DL3DV-2-final-softfloor-5k \
  --n_train 10 \
  --n_test 12 \
  -i 5000 \
  -r 1 \
  --scene_graph logwin-3-noncyclic \
  --max_init_points 0 \
  --point_conf_threshold 0 \
  --optim_test_pose_iter 500 \
  --use_pseudo_views \
  --pseudo_manifest /path/to/pseudo_manifest_reconx_dust3r_confidence.json \
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

This command assumes the pseudo manifest already contains confidence-mask paths produced by:

```bash
python tools/build_reconx_dust3r_confidence.py \
  --pseudo_manifest /path/to/pseudo_manifest_fused_nomask_external.json \
  --source_images_dir /path/to/DL3DV-2/images \
  --output_dir /path/to/DL3DV-2-reconx-dust3r-confidence \
  --image_size 512 \
  --batch_size 1 \
  --bidirectional_pairs \
  --left_right_weight 1.0 \
  --neighbor_weight 0.75 \
  --all_pair_weight 0.5 \
  --aggregate weighted_mean \
  --mask_blur 1.0
```

For Waymo-405841 FRONT we found lower pose/scale learning rates more stable:

```bash
python run_eval_pipeline.py \
  -s /path/to/405841/FRONT \
  -m /path/to/outputs/FRONT-final-softfloor-5k-lowlr-r4 \
  --n_train 20 \
  --n_test 12 \
  -i 5000 \
  -r 4 \
  --scene_graph logwin-3-noncyclic \
  --max_init_points 0 \
  --point_conf_threshold 0 \
  --optim_test_pose_iter 500 \
  --position_lr_init 0.000016 \
  --position_lr_final 0.00000016 \
  --scaling_lr 0.0005 \
  --use_pseudo_views \
  --pseudo_manifest /path/to/FRONT_pseudo_manifest_reconx_dust3r_confidence.json \
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

### Dataset Commands Used for the Final Table

Use fixed train/test split manifests when comparing methods. If you do not pass `--split_manifest`, `run_eval_pipeline.py` will deterministically sample train/test images from the sorted `images/` folder, but exact table reproduction requires using the same split and pseudo manifest across ablations.

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
  --pseudo_manifest /path/to/DL3DV-2-reconx-dust3r-confidence/pseudo_manifest_reconx_dust3r_confidence.json \
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

Re10k-1:

```bash
python run_eval_pipeline.py \
  -s /path/to/Re10k-1 \
  -m /path/to/outputs/Re10k-1-final-softfloor-5k \
  --n_train 9 \
  --n_test 12 \
  --split_manifest /path/to/splits/Re10k-1-9real-explicit-train-test.json \
  -i 5000 \
  -r 1 \
  --scene_graph logwin-3-noncyclic \
  --max_init_points 0 \
  --point_conf_threshold 0 \
  --optim_test_pose_iter 500 \
  --use_pseudo_views \
  --pseudo_manifest /path/to/Re10k-1-reconx-dust3r-confidence/pseudo_manifest_reconx_dust3r_confidence.json \
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

Waymo-405841 FRONT:

```bash
python run_eval_pipeline.py \
  -s /path/to/405841/FRONT \
  -m /path/to/outputs/FRONT-final-softfloor-5k-lowlr-r4 \
  --n_train 20 \
  --n_test 12 \
  --split_manifest /path/to/splits/FRONT-20real-explicit-train-test.json \
  -i 5000 \
  -r 4 \
  --scene_graph logwin-3-noncyclic \
  --max_init_points 0 \
  --point_conf_threshold 0 \
  --optim_test_pose_iter 500 \
  --position_lr_init 0.000016 \
  --position_lr_final 0.00000016 \
  --scaling_lr 0.0005 \
  --use_pseudo_views \
  --pseudo_manifest /path/to/FRONT-reconx-dust3r-confidence/pseudo_manifest_reconx_dust3r_confidence.json \
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

## Notes and Limitations

- Generated pseudo views can improve coverage, but they can also introduce inconsistent content. The final pipeline therefore uses them as weighted supervision rather than as geometry seeds.
- MASt3R/DUSt3R confidence naturally favors textured and nearby regions. The softfloor remap was added because raw confidence can under-supervise sky, distant buildings, and other low-texture backgrounds.
- The repository contains scripts for additional failed or exploratory directions, including pseudo-as-real training and densification variants. The recommended final path is the confidence-weighted pseudo-supervision command above.

## Citation / Acknowledgements

This project builds on InstantSplat, DUSt3R/MASt3R, 3D Gaussian Splatting, ViewCrafter, DiFix3D+, ReconX-style confidence-aware optimization, and BRPO-style bidirectional pseudo-view reasoning. Please cite the corresponding original papers and repositories when reusing this code.
