#!/usr/bin/env bash
# Batch run InstantSplat on the prepared Part2 scenes.
#
# Output layout:
#   $OUTPUT_ROOT/<scene_name>/<N_VIEWS>_views/
#     sparse_<N_VIEWS>/0/              # MASt3R/DUSt3R COLMAP-style init, written inside source scene
#     point_cloud/iteration_<iter>/    # optimized Gaussian result
#     train/ours_<iter>/renders/       # rendered training views
#     interp/ours_<iter>/renders/      # interpolated video frames when INFER_VIDEO=1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/AIAA3201_3DGS_Project}"
INSTANTSPLAT_DIR="${INSTANTSPLAT_DIR:-${PROJECT_DIR}/third_party/InstantSplat}"
PART2_SCRIPT_DIR="${PART2_SCRIPT_DIR:-${PROJECT_DIR}/Part2_Scripts}"
ASSET_ROOT="${ASSET_ROOT:-${PART2_SCRIPT_DIR}/InstantSplat_assets/part2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PART2_SCRIPT_DIR}/InstantSplat_outputs}"

GPU_ID="${GPU_ID:-0}"

# Unique run log. It records the whole batch process while still printing to terminal.
# You can override it with: RUN_LOG=/path/to/my_log.txt ./run_instantsplat_part2_batch.sh
RUN_LOG_DIR="${RUN_LOG_DIR:-${PART2_SCRIPT_DIR}/InstantSplat_logs}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')_pid$$}"
RUN_LOG="${RUN_LOG:-${RUN_LOG_DIR}/instantsplat_part2_${RUN_ID}.txt}"

# =======================================================
# Default view count fallback.
# Normally SCENE_N_VIEWS below is "auto", so the script reads sampled_count from
# InstantSplat_assets/part2/<scene>/source_info.txt, which is produced by
# prepare_instantsplat_part2_data.sh according to the Part2 PDF sparsity.
# =======================================================
N_VIEWS="${N_VIEWS:-3}"

# Gaussian optimization iterations.
# Increase for better quality, decrease for quick smoke tests.
GS_TRAIN_ITER="${GS_TRAIN_ITER:-1500}"

# 1 = run init_geo.py, 0 = skip when sparse_<N_VIEWS>/0 already exists.
DO_INIT="${DO_INIT:-1}"
# 1 = run train.py and produce Gaussian point_cloud, 0 = skip.
DO_TRAIN="${DO_TRAIN:-1}"
# 1 = run render.py after training, 0 = skip.
DO_RENDER="${DO_RENDER:-1}"
# 1 = prepare test cameras, render test views, and run metrics.py after rendering.
DO_EVAL="${DO_EVAL:-1}"

# Keep this disabled for the official InstantSplat split_train_test path:
# images/ contains all source frames and n_views is computed from the Part2
# stride. init_geo.py then writes sparse_<N>/0 for train and sparse_<N>/1 for eval.
INIT_INFER_VIDEO="${INIT_INFER_VIDEO:-0}"

# Render an interpolated video after training. This does not affect which images
# were sampled for initialization; N_VIEWS still controls that above.
RENDER_INFER_VIDEO="${RENDER_INFER_VIDEO:-1}"

# Optional MASt3R init flags from InstantSplat's README scripts.
FOCAL_AVG="${FOCAL_AVG:-1}"
CO_VIS_DSP="${CO_VIS_DSP:-1}"
CONF_AWARE_RANKING="${CONF_AWARE_RANKING:-1}"
EVAL_OPTIM_TEST_POSE_ITER="${EVAL_OPTIM_TEST_POSE_ITER:-500}"

SCENE_NAMES=(
  # "re10k_1"
  # "front_405841"
  "dl3dv_2"
  # "dl3dv_2_5views"
)

# Per-scene view counts passed to InstantSplat.
# "auto" means read the actual sampled_count from source_info.txt.
# The order must match SCENE_NAMES above.
SCENE_N_VIEWS=(
  # auto   # re10k_1: sampled by 1/30 frames
  # auto   # front_405841: sampled by 1/10 frames
  auto   # dl3dv_2: sampled by 1/30 frames
)

require_path() {
  local path="$1"
  local what="$2"
  if [ ! -e "$path" ]; then
    echo "ERROR: missing ${what}: ${path}" >&2
    exit 1
  fi
}

run_cmd() {
  echo ""
  echo ">>> $*"
  "$@"
}

flag_if_enabled() {
  local enabled="$1"
  local flag="$2"
  if [ "$enabled" -eq 1 ]; then
    printf '%s\n' "$flag"
  fi
}

original_data_dir_for_scene() {
  local scene="$1"
  case "$scene" in
    re10k_1*)
      printf '%s\n' "/root/autodl-fs/Re10k-1"
      ;;
    front_405841*)
      printf '%s\n' "/root/autodl-fs/405841/FRONT"
      ;;
    dl3dv_2*)
      printf '%s\n' "/root/autodl-fs/DL3DV-2"
      ;;
    *)
      echo "ERROR: unknown original data dir for scene ${scene}" >&2
      exit 1
      ;;
  esac
}

resolve_scene_n_views() {
  local scene_dir="$1"
  local configured="$2"
  local info_path="${scene_dir}/source_info.txt"

  if [ "$configured" != "auto" ]; then
    printf '%s\n' "$configured"
    return
  fi

  if [ ! -f "$info_path" ]; then
    local image_dir="${scene_dir}/images"
    if [ -d "$image_dir" ]; then
      echo "WARNING: missing ${info_path}; falling back to counting images in ${image_dir}." >&2
      IMAGE_DIR="$image_dir" python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["IMAGE_DIR"])
exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
print(sum(1 for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts))
PY
      return
    fi

    echo "ERROR: cannot auto-detect n_views; missing ${info_path} and ${image_dir}. Run prepare_instantsplat_part2_data.sh first." >&2
    exit 1
  fi

  local sampled_count
  sampled_count="$(sed -n 's/^sampled_count=//p' "$info_path" | head -n 1)"
  if [ -z "$sampled_count" ]; then
    echo "ERROR: cannot find sampled_count in ${info_path}" >&2
    exit 1
  fi

  printf '%s\n' "$sampled_count"
}

require_path "$INSTANTSPLAT_DIR" "InstantSplat directory"
require_path "${INSTANTSPLAT_DIR}/init_geo.py" "init_geo.py"
require_path "${INSTANTSPLAT_DIR}/train.py" "train.py"
require_path "${INSTANTSPLAT_DIR}/render.py" "render.py"
require_path "${INSTANTSPLAT_DIR}/metrics.py" "metrics.py"
require_path "${PART2_SCRIPT_DIR}/prepare_instantsplat_part2_eval.py" "Part2 eval preparation script"
require_path "${INSTANTSPLAT_DIR}/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" "MASt3R checkpoint"

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$RUN_LOG_DIR"

# From here on, every echo and every command output that reaches stdout/stderr
# is written both to terminal and to the unique txt log.
exec > >(tee -a "$RUN_LOG") 2>&1

cd "$INSTANTSPLAT_DIR"

echo "======================================================="
echo "InstantSplat Part2 batch"
echo "Run log:          $RUN_LOG"
echo "InstantSplat dir: $INSTANTSPLAT_DIR"
echo "Asset root:       $ASSET_ROOT"
echo "Output root:      $OUTPUT_ROOT"
echo "GPU_ID:           $GPU_ID"
echo "N_VIEWS fallback: $N_VIEWS"
echo "SCENE_N_VIEWS:    ${SCENE_N_VIEWS[*]}   <-- auto reads PDF-sampled image counts"
echo "GS_TRAIN_ITER:    $GS_TRAIN_ITER"
echo "DO_INIT/TRAIN/RENDER/EVAL: ${DO_INIT}/${DO_TRAIN}/${DO_RENDER}/${DO_EVAL}"
echo "INIT_INFER_VIDEO: ${INIT_INFER_VIDEO}"
echo "RENDER_INFER_VIDEO: ${RENDER_INFER_VIDEO}"
echo "EVAL_OPTIM_TEST_POSE_ITER: ${EVAL_OPTIM_TEST_POSE_ITER}"
echo "======================================================="

for scene_idx in "${!SCENE_NAMES[@]}"; do
  scene="${SCENE_NAMES[$scene_idx]}"
  source_path="${ASSET_ROOT}/${scene}"
  scene_n_views="$(resolve_scene_n_views "$source_path" "${SCENE_N_VIEWS[$scene_idx]:-$N_VIEWS}")"
  image_path="${source_path}/images"
  test_image_path="${source_path}/test_images"
  original_data_dir="$(original_data_dir_for_scene "$scene")"
  model_path="${OUTPUT_ROOT}/${scene}/${scene_n_views}_views"
  log_dir="${model_path}/logs"
  sparse_dir="${source_path}/sparse_${scene_n_views}/0"

  require_path "$image_path" "${scene} images directory"
  mkdir -p "$model_path" "$log_dir"

  echo ""
  echo "======================================================="
  echo "Scene:       $scene"
  echo "N views:     $scene_n_views"
  echo "Source:      $source_path"
  echo "Original:    $original_data_dir"
  echo "Model path:  $model_path"
  echo "Sparse init: $sparse_dir"
  echo "======================================================="

  if [ "$DO_INIT" -eq 1 ]; then
    init_args=(
      -W ignore ./init_geo.py
      -s "$source_path"
      -m "$model_path"
      --n_views "$scene_n_views"
    )

    if [ "$FOCAL_AVG" -eq 1 ]; then
      init_args+=(--focal_avg)
    fi
    if [ "$CO_VIS_DSP" -eq 1 ]; then
      init_args+=(--co_vis_dsp)
    fi
    if [ "$CONF_AWARE_RANKING" -eq 1 ]; then
      init_args+=(--conf_aware_ranking)
    fi
    if [ "$INIT_INFER_VIDEO" -eq 1 ]; then
      init_args+=(--infer_video)
    fi

    echo ">>> [1/4] init_geo.py"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python "${init_args[@]}" 2>&1 | tee "${log_dir}/01_init_geo.log"
    echo "    log: ${log_dir}/01_init_geo.log"
  else
    echo ">>> [1/4] init_geo.py skipped"
  fi

  if [ "$DO_TRAIN" -eq 1 ]; then
    echo ">>> [2/4] train.py"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python ./train.py \
      -s "$source_path" \
      -m "$model_path" \
      -r 1 \
      --n_views "$scene_n_views" \
      --iterations "$GS_TRAIN_ITER" \
      --pp_optimizer \
      --optim_pose \
      2>&1 | tee "${log_dir}/02_train.log"
    echo "    log: ${log_dir}/02_train.log"
  else
    echo ">>> [2/4] train.py skipped"
  fi

  if [ "$DO_RENDER" -eq 1 ]; then
    echo ">>> [3/4] render.py"
    render_args=(
      ./render.py
      -s "$source_path"
      -m "$model_path"
      -r 1
      --n_views "$scene_n_views"
      --iterations "$GS_TRAIN_ITER"
    )
    if [ "$RENDER_INFER_VIDEO" -eq 1 ]; then
      render_args+=(--infer_video)
    fi

    CUDA_VISIBLE_DEVICES="$GPU_ID" python "${render_args[@]}" 2>&1 | tee "${log_dir}/03_render.log"
    echo "    log: ${log_dir}/03_render.log"
  else
    echo ">>> [3/4] render.py skipped"
  fi

  if [ "$DO_EVAL" -eq 1 ]; then
    eval_images_arg=(--images images)
    if [ ! -f "${source_path}/sparse_${scene_n_views}/1/images.txt" ]; then
      if [ ! -d "$test_image_path" ]; then
        echo ">>> [4/4] metrics skipped: missing ${source_path}/sparse_${scene_n_views}/1/images.txt and ${test_image_path}"
        echo "         Re-run init_geo.py with INIT_INFER_VIDEO=0 or rerun prepare_instantsplat_part2_data.sh."
        continue
      fi

      echo ">>> [4/4] official eval sparse missing; preparing fallback eval sparse from test_files.txt"
      eval_images_arg=(--images test_images)
      python "${PART2_SCRIPT_DIR}/prepare_instantsplat_part2_eval.py" \
        --source-path "$source_path" \
        --original-data-dir "$original_data_dir" \
        --n-views "$scene_n_views" \
        --overwrite \
        2>&1 | tee "${log_dir}/04_prepare_eval.log"
    fi

    if [ ! -e "${source_path}/sparse/0/images.bin" ]; then
      echo ">>> [4/4] metrics skipped: missing ${source_path}/sparse/0/images.bin for ATE."
      echo "         Re-run prepare_instantsplat_part2_data.sh so it can link the original sparse model."
    else
      echo ">>> [4/4] prepare eval sparse, render test views, and calculate metrics"
      CUDA_VISIBLE_DEVICES="$GPU_ID" python ./render.py \
        -s "$source_path" \
        -m "$model_path" \
        -r 1 \
        "${eval_images_arg[@]}" \
        --n_views "$scene_n_views" \
        --iterations "$GS_TRAIN_ITER" \
        --eval \
        --optim_test_pose_iter "$EVAL_OPTIM_TEST_POSE_ITER" \
        2>&1 | tee "${log_dir}/05_render_eval.log"

      CUDA_VISIBLE_DEVICES="$GPU_ID" python ./metrics.py \
        -s "$source_path" \
        -m "$model_path" \
        --n_views "$scene_n_views" \
        2>&1 | tee "${log_dir}/06_metrics.log"

      echo "    metrics: ${model_path}/results.json"
      echo "    per-view metrics: ${model_path}/per_view.json"
      echo "    detailed image metrics: ${model_path}/test/ours_${GS_TRAIN_ITER}/metrics.txt"
      echo "    pose metrics: ${model_path}/pose/ours_${GS_TRAIN_ITER}/pose_eval.txt"
    fi
  else
    echo ">>> [4/4] metrics skipped"
  fi

  echo "Done: $scene"
  echo "Gaussian result should be under: ${model_path}/point_cloud/iteration_${GS_TRAIN_ITER}/"
done

echo ""
echo "======================================================="
echo "All scenes finished."
echo "Outputs are under: $OUTPUT_ROOT"
echo "Per-scene sampling image counts were: ${SCENE_N_VIEWS[*]}."
echo "======================================================="
