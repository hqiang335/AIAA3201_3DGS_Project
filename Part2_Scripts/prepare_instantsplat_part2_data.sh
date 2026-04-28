#!/usr/bin/env bash
# Prepare Part2 datasets as InstantSplat scene folders.
#
# InstantSplat expects each scene to look like:
#   <scene_dir>/images/<frame images>
#
# This script creates lightweight scene folders under:
#   $PROJECT_DIR/Part2_Scripts/InstantSplat_assets/part2/<scene_name>/images
#
# To stay aligned with official InstantSplat, images/ contains all source images.
# The script only computes the Part2 n_views from the required stride and writes
# it to source_info.txt. run_instantsplat_part2_batch.sh then passes that n_views
# to init_geo.py without --infer_video, so InstantSplat's own split_train_test
# selects train/test views and writes sparse_<N>/0 and sparse_<N>/1.
#
# The sparse training sampling rate follows the Part2 PDF:
#   Re10k-1:        1/30 frames
#   Waymo-405841:   1/10 frames
#   DL3DV-2:        1/30 frames
#
# By default files are symlinks to original images. Set COPY_IMAGES=1 to copy.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/AIAA3201_3DGS_Project}"
INSTANTSPLAT_DIR="${INSTANTSPLAT_DIR:-${PROJECT_DIR}/third_party/InstantSplat}"
PART2_SCRIPT_DIR="${PART2_SCRIPT_DIR:-${PROJECT_DIR}/Part2_Scripts}"
ASSET_ROOT="${ASSET_ROOT:-${PART2_SCRIPT_DIR}/InstantSplat_assets/part2}"
COPY_IMAGES="${COPY_IMAGES:-0}"

SCENE_NAMES=(
  "re10k_1"
  "front_405841"
  "dl3dv_2"
)

IMAGE_DIRS=(
  "/root/autodl-fs/Re10k-1/images"
  "/root/autodl-fs/405841/FRONT/images"
  "/root/autodl-fs/DL3DV-2/images"
)

P2_STRIDES=(
  30  # re10k_1: 1/30 frames
  10  # front_405841: 1/10 frames
  30  # dl3dv_2: 1/30 frames
)

prepare_official_images() {
  local image_dir="$1"
  local target_images="$2"
  local target_test_images="$3"
  local stride="$4"
  local copy_images="$5"

IMAGE_DIR="$image_dir" TARGET_IMAGES="$target_images" TARGET_TEST_IMAGES="$target_test_images" STRIDE="$stride" COPY_IMAGES="$copy_images" python - <<'PY'
import math
import os
import re
import shutil
from pathlib import Path

import numpy as np

root = Path(os.environ["IMAGE_DIR"])
target = Path(os.environ["TARGET_IMAGES"])
target_test = Path(os.environ["TARGET_TEST_IMAGES"])
stride = int(os.environ["STRIDE"])
copy_images = os.environ["COPY_IMAGES"] == "1"
exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

def sort_key(path: Path):
    match = re.search(r"\d+", path.stem)
    return (int(match.group()) if match else 10**18, path.name)

files = sorted(
    [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts],
    key=sort_key,
)

if len(files) < 3:
    raise RuntimeError(f"Need at least 3 images for split_train_test-style sampling, got {len(files)}")

test_count = min(12, max(0, len(files) - 2))
test_idx = np.linspace(1, len(files) - 2, num=test_count, dtype=int).tolist()
test_idx_set = set(test_idx)
train_candidate_idx = [idx for idx in range(len(files)) if idx not in test_idx_set]

sampled_count = max(1, math.ceil(len(train_candidate_idx) / stride))
sample_positions = np.linspace(0, len(train_candidate_idx) - 1, num=sampled_count, dtype=int).tolist()
sampled_idx = [train_candidate_idx[pos] for pos in sample_positions]

sampled = [files[idx] for idx in sampled_idx]
test_files = [files[idx] for idx in test_idx]

target.mkdir(parents=True, exist_ok=True)
for p in files:
    out = target / p.name
    if copy_images:
        shutil.copy2(p, out)
    else:
        out.symlink_to(p)

target_test.mkdir(parents=True, exist_ok=True)
for p in test_files:
    out = target_test / p.name
    if copy_images:
        shutil.copy2(p, out)
    else:
        out.symlink_to(p)

(target.parent / "train_sampled_files.txt").write_text(
    "\n".join(f"{idx}\t{files[idx].name}" for idx in sampled_idx) + "\n"
)
(target.parent / "test_files.txt").write_text(
    "\n".join(f"{idx}\t{files[idx].name}" for idx in test_idx) + "\n"
)

print(f"{len(files)} {len(train_candidate_idx)} {len(sampled)} {len(test_files)}")
PY
}

link_original_sparse() {
  local image_dir="$1"
  local scene_dir="$2"
  local original_root
  original_root="$(dirname "$image_dir")"
  local original_sparse="${original_root}/sparse"
  local target_sparse="${scene_dir}/sparse"

  if [ ! -e "${original_sparse}/0/images.bin" ]; then
    echo "WARNING: original COLMAP sparse/0/images.bin not found under ${original_sparse}; metrics.py ATE may be unavailable." >&2
    return
  fi

  if [ -L "$target_sparse" ] || [ -e "$target_sparse" ]; then
    rm -rf "$target_sparse"
  fi
  ln -s "$original_sparse" "$target_sparse"
}

mkdir -p "$ASSET_ROOT"

echo "======================================================="
echo "Preparing InstantSplat Part2 scenes"
echo "InstantSplat dir: $INSTANTSPLAT_DIR"
echo "Asset root:       $ASSET_ROOT"
echo "COPY_IMAGES:      $COPY_IMAGES"
echo "======================================================="

for i in "${!SCENE_NAMES[@]}"; do
  scene="${SCENE_NAMES[$i]}"
  image_dir="${IMAGE_DIRS[$i]}"
  stride="${P2_STRIDES[$i]}"
  scene_dir="${ASSET_ROOT}/${scene}"
  target_images="${scene_dir}/images"
  target_test_images="${scene_dir}/test_images"

  if [ ! -d "$image_dir" ]; then
    echo "ERROR: image directory does not exist: $image_dir" >&2
    exit 1
  fi

  mkdir -p "$scene_dir"
  rm -rf "$target_images"
  rm -rf "$target_test_images"
  rm -rf "${scene_dir}"/sparse_*
  if [ -L "${scene_dir}/sparse" ]; then
    rm -f "${scene_dir}/sparse"
  fi
  rm -f "${scene_dir}/train_sampled_files.txt" "${scene_dir}/test_files.txt"

  counts="$(prepare_official_images "$image_dir" "$target_images" "$target_test_images" "$stride" "$COPY_IMAGES")"
  image_count="$(echo "$counts" | awk '{print $1}')"
  train_candidate_count="$(echo "$counts" | awk '{print $2}')"
  sampled_count="$(echo "$counts" | awk '{print $3}')"
  test_count="$(echo "$counts" | awk '{print $4}')"

  if [ "$image_count" -eq 0 ]; then
    echo "ERROR: no images found in: $image_dir" >&2
    exit 1
  fi

  if [ "$sampled_count" -eq 0 ]; then
    echo "ERROR: no sampled images produced for: $image_dir" >&2
    exit 1
  fi

  link_mode="official_all_symlinks"
  if [ "$COPY_IMAGES" -eq 1 ]; then
    link_mode="official_all_copies"
  fi
  link_original_sparse "$image_dir" "$scene_dir"

  cat > "${scene_dir}/source_info.txt" <<EOF
scene_name=${scene}
source_images=${image_dir}
source_image_count=${image_count}
split_method=official_instantsplat_split_train_test_with_pdf_rate_n_views
test_count=${test_count}
train_candidate_count=${train_candidate_count}
p2_stride=${stride}
sampled_count=${sampled_count}
images_mode=${link_mode}
EOF

  echo "OK: ${scene}"
  echo "    source images:       ${image_count}"
  echo "    test images:         ${test_count} -> ${target_test_images}"
  echo "    train candidates:    ${train_candidate_count}"
  echo "    n_views for init:    ${sampled_count} (target rate: 1/${stride})"
  echo "    InstantSplat images: ${target_images} (all source images; official split_train_test selects the sparse views)"
done

echo "======================================================="
echo "Done. Prepared scenes are under: $ASSET_ROOT"
echo "Next: run ./run_instantsplat_part2_batch.sh"
echo "======================================================="
