#!/usr/bin/env bash
# Prepare InstantSplat scene folders for sparse-input inference + uniform-span eval.
#
# This is intentionally similar to prepare_instantsplat_part2_data.sh, but the
# sampling semantics are different:
#   - TRAIN_IMAGE_DIR points at the sparse input views (n_views) you want InstantSplat to use.
#   - images/ in the output scene contains ALL original frames in the closed index span
#     between the min/max sparse input frame (so init_geo.py without --infer_video can run
#     the official split_train_test logic on the same contiguous subsequence).
#   - test_images/ contains an optional visualization list (same count as train views),
#     sampled uniformly from frames strictly between the first/last sparse input frame.
#   - sparse_<N>/1 is NOT authored here; it is created by init_geo.py (no --infer_video).
#
# Example:
#   SCENE_NAME=front_405841_6 \
#   TRAIN_IMAGE_DIR=/root/autodl-tmp/AIAA3201_3DGS_Project/third_party/InstantSplat/assets/part2/front_405841/front_405841_6/images \
#   ORIGINAL_DATA_DIR=/root/autodl-fs/405841/FRONT \
#   bash /root/autodl-tmp/AIAA3201_3DGS_Project/Part2_Scripts/prepare_instantsplat_uniform_span_data.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/AIAA3201_3DGS_Project}"
PART2_SCRIPT_DIR="${PART2_SCRIPT_DIR:-${PROJECT_DIR}/Part2_Scripts}"
INSTANTSPLAT_DIR="${INSTANTSPLAT_DIR:-${PROJECT_DIR}/third_party/InstantSplat}"

SCENE_NAME="${SCENE_NAME:-front_405841_6}"
TRAIN_IMAGE_DIR="${TRAIN_IMAGE_DIR:-${INSTANTSPLAT_DIR}/assets/part2/front_405841/front_405841_6/images}"
ORIGINAL_DATA_DIR="${ORIGINAL_DATA_DIR:-/root/autodl-fs/405841/FRONT}"
ASSET_ROOT="${ASSET_ROOT:-${INSTANTSPLAT_DIR}/assets/part2/front_405841_uniform_span}"
COPY_IMAGES="${COPY_IMAGES:-0}"

SCENE_DIR="${ASSET_ROOT}/${SCENE_NAME}"
TARGET_IMAGES="${SCENE_DIR}/images"
TARGET_TEST_IMAGES="${SCENE_DIR}/test_images"

echo "======================================================="
echo "Preparing InstantSplat uniform-span eval scene"
echo "Scene:             ${SCENE_NAME}"
echo "Train images:      ${TRAIN_IMAGE_DIR}"
echo "Original data:     ${ORIGINAL_DATA_DIR}"
echo "Output scene:      ${SCENE_DIR}"
echo "COPY_IMAGES:       ${COPY_IMAGES}"
echo "======================================================="

if [ ! -d "$TRAIN_IMAGE_DIR" ]; then
  echo "ERROR: TRAIN_IMAGE_DIR does not exist: $TRAIN_IMAGE_DIR" >&2
  exit 1
fi
if [ ! -d "${ORIGINAL_DATA_DIR}/images" ]; then
  echo "ERROR: original images directory does not exist: ${ORIGINAL_DATA_DIR}/images" >&2
  exit 1
fi
if [ ! -f "${ORIGINAL_DATA_DIR}/cameras.json" ]; then
  echo "ERROR: cameras.json does not exist: ${ORIGINAL_DATA_DIR}/cameras.json" >&2
  exit 1
fi

rm -rf "$SCENE_DIR"
mkdir -p "$TARGET_IMAGES" "$TARGET_TEST_IMAGES"

counts="$(
TRAIN_IMAGE_DIR="$TRAIN_IMAGE_DIR" \
ORIGINAL_DATA_DIR="$ORIGINAL_DATA_DIR" \
TARGET_IMAGES="$TARGET_IMAGES" \
TARGET_TEST_IMAGES="$TARGET_TEST_IMAGES" \
COPY_IMAGES="$COPY_IMAGES" \
python - <<'PY'
import os
import re
import shutil
from pathlib import Path

import numpy as np

train_dir = Path(os.environ["TRAIN_IMAGE_DIR"])
original = Path(os.environ["ORIGINAL_DATA_DIR"])
full_dir = original / "images"
target_train = Path(os.environ["TARGET_IMAGES"])
target_test = Path(os.environ["TARGET_TEST_IMAGES"])
copy_images = os.environ["COPY_IMAGES"] == "1"
exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

def sort_key(path: Path):
    match = re.search(r"\d+", path.stem)
    return (int(match.group()) if match else 10**18, path.name)

def link_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_images:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)

full_files = sorted(
    [p for p in full_dir.iterdir() if p.is_file() and p.suffix.lower() in exts],
    key=sort_key,
)
train_files = sorted(
    [p for p in train_dir.iterdir() if p.is_file() and p.suffix.lower() in exts],
    key=sort_key,
)

if len(train_files) < 2:
    raise RuntimeError(f"Need at least 2 train images, got {len(train_files)}")

name_to_idx = {p.name: i for i, p in enumerate(full_files)}
missing = [p.name for p in train_files if p.name not in name_to_idx]
if missing:
    raise RuntimeError(f"Train images are missing from original images: {missing[:10]}")

train_idx = [name_to_idx[p.name] for p in train_files]
n_views = len(train_idx)
i_lo, i_hi = min(train_idx), max(train_idx)

candidate_idx = [i for i in range(i_lo + 1, i_hi) if i not in set(train_idx)]
if len(candidate_idx) < n_views:
    raise RuntimeError(
        f"Only {len(candidate_idx)} non-train frames inside ({i_lo}, {i_hi}), "
        f"but need {n_views}. Use sparser train views or fewer views."
    )

if len(candidate_idx) == n_views:
    test_idx = candidate_idx
else:
    local = [int(round(float(x))) for x in np.linspace(0, len(candidate_idx) - 1, n_views)]
    test_idx = []
    used = set()
    for li in local:
        gi = candidate_idx[max(0, min(len(candidate_idx) - 1, li))]
        if gi in used:
            for alt in candidate_idx:
                if alt not in used:
                    gi = alt
                    break
        test_idx.append(gi)
        used.add(gi)

target_train.mkdir(parents=True, exist_ok=True)
target_test.mkdir(parents=True, exist_ok=True)

for idx in range(i_lo, i_hi + 1):
    p = full_files[idx]
    link_or_copy(p, target_train / p.name)

for gi in test_idx:
    p = full_files[gi]
    link_or_copy(p, target_test / p.name)

(target_train.parent / "train_sampled_files.txt").write_text(
    "\n".join(f"{idx}\t{full_files[idx].name}" for idx in train_idx) + "\n",
    encoding="utf-8",
)
(target_train.parent / "test_files.txt").write_text(
    "\n".join(f"{idx}\t{full_files[idx].name}" for idx in test_idx) + "\n",
    encoding="utf-8",
)

span_image_count = i_hi - i_lo + 1
print(f"{len(full_files)} {span_image_count} {n_views} {len(test_idx)} {i_lo} {i_hi}")
PY
)"

source_image_count="$(echo "$counts" | awk '{print $1}')"
span_image_count="$(echo "$counts" | awk '{print $2}')"
sampled_count="$(echo "$counts" | awk '{print $3}')"
test_count="$(echo "$counts" | awk '{print $4}')"
span_start="$(echo "$counts" | awk '{print $5}')"
span_end="$(echo "$counts" | awk '{print $6}')"

link_mode="uniform_span_symlinks"
if [ "$COPY_IMAGES" -eq 1 ]; then
  link_mode="uniform_span_copies"
fi

cat > "${SCENE_DIR}/source_info.txt" <<EOF
scene_name=${SCENE_NAME}
source_images=${ORIGINAL_DATA_DIR}/images
source_image_count=${source_image_count}
split_method=uniform_span_sparse_input_train_test_equal_n_views
span_image_count=${span_image_count}
test_count=${test_count}
train_candidate_count=${sampled_count}
p2_stride=manual
sampled_count=${sampled_count}
images_mode=${link_mode}
span_start_index=${span_start}
span_end_index=${span_end}
EOF

echo "======================================================="
echo "Done"
echo "Scene dir:     ${SCENE_DIR}"
echo "Span images:   ${span_image_count} -> ${TARGET_IMAGES} (indices ${span_start}..${span_end})"
echo "Sparse n_views:${sampled_count} (see train_sampled_files.txt)"
echo "Optional tests:${test_count} -> ${TARGET_TEST_IMAGES} (see test_files.txt)"
echo "Next: run instantsplat_gradio_no_metric.py with Input Directory = ${SCENE_DIR}"
echo "Note: sparse_${sampled_count}/1 will be created by init_geo.py (no --infer_video)"
echo "======================================================="
