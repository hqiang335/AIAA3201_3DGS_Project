#!/usr/bin/env python3
"""Prepare InstantSplat Part2 eval files.

This creates the official-style evaluation inputs expected by:
  python render.py --eval --images test_images
  python metrics.py -s <source_path> -m <model_path> --n_views <N>

For a prepared Part2 scene, it writes:
  <source_path>/sparse_<N>/1/cameras.txt
  <source_path>/sparse_<N>/1/images.txt
and links:
  <source_path>/sparse -> <original_dataset>/sparse

The test image list comes from prepare_instantsplat_part2_data.sh:
  <source_path>/test_files.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Part2 InstantSplat evaluation inputs.")
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--original-data-dir", type=Path, required=True)
    parser.add_argument("--n-views", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def quat_xyzw_to_rotmat(quat: list[float] | np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat, dtype=np.float64)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def rotmat_to_qvec_wxyz(rot: np.ndarray) -> np.ndarray:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rot.flat
    k = np.array(
        [
            [rxx - ryy - rzz, 0, 0, 0],
            [ryx + rxy, ryy - rxx - rzz, 0, 0],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ],
        dtype=np.float64,
    ) / 3.0
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def entry_to_w2c(entry: dict) -> np.ndarray:
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = quat_xyzw_to_rotmat(entry["cam_quat"])
    c2w[:3, 3] = np.asarray(entry["cam_trans"], dtype=np.float64)
    return np.linalg.inv(c2w)


def load_test_names(source_path: Path) -> list[str]:
    test_file = source_path / "test_files.txt"
    if not test_file.exists():
        raise FileNotFoundError(f"Missing {test_file}. Run prepare_instantsplat_part2_data.sh first.")
    names = []
    for line in test_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        names.append(parts[-1])
    return names


def pixel_intrinsics(entry: dict, width: int, height: int) -> tuple[float, float, float, float]:
    fx = float(entry.get("fx", 0.0))
    fy = float(entry.get("fy", fx))
    cx = float(entry.get("cx", 0.5))
    cy = float(entry.get("cy", 0.5))

    # RegGS-style json stores normalized intrinsics for these datasets.
    if abs(fx) <= 10:
        fx *= width
    if abs(fy) <= 10:
        fy *= height
    if abs(cx) <= 2:
        cx *= width
    if abs(cy) <= 2:
        cy *= height
    return fx, fy, cx, cy


def link_gt_sparse(source_path: Path, original_data_dir: Path) -> None:
    original_sparse = original_data_dir / "sparse"
    if not (original_sparse / "0" / "images.bin").exists():
        print(f"[WARN] Original sparse/0/images.bin not found at {original_sparse}; ATE may be unavailable.")
        return

    target_sparse = source_path / "sparse"
    if target_sparse.exists() or target_sparse.is_symlink():
        if target_sparse.resolve() == original_sparse.resolve():
            return
        if target_sparse.is_dir() and not target_sparse.is_symlink():
            print(f"[WARN] Keeping existing non-symlink sparse directory: {target_sparse}")
            return
        target_sparse.unlink()
    target_sparse.symlink_to(original_sparse, target_is_directory=True)


def main() -> None:
    args = parse_args()
    source_path = args.source_path
    original_data_dir = args.original_data_dir

    cameras_path = original_data_dir / "cameras.json"
    if not cameras_path.exists():
        raise FileNotFoundError(f"Missing {cameras_path}")
    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    cameras_by_name = {entry["image_name"]: entry for entry in cameras}

    test_names = load_test_names(source_path)
    test_images_dir = source_path / "test_images"
    if not test_images_dir.exists():
        raise FileNotFoundError(f"Missing {test_images_dir}")

    sparse_eval_dir = source_path / f"sparse_{args.n_views}" / "1"
    if sparse_eval_dir.exists() and args.overwrite:
        shutil.rmtree(sparse_eval_dir)
    sparse_eval_dir.mkdir(parents=True, exist_ok=True)

    cameras_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(test_names)}",
    ]
    images_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(test_names)}, mean observations per image: 0",
    ]

    missing = []
    for idx, name in enumerate(test_names, start=1):
        entry = cameras_by_name.get(name)
        if entry is None:
            missing.append(name)
            continue
        image_path = test_images_dir / name
        if not image_path.exists():
            missing.append(name)
            continue

        with Image.open(image_path) as image:
            width, height = image.size
        fx, fy, cx, cy = pixel_intrinsics(entry, width, height)

        w2c = entry_to_w2c(entry)
        qvec = rotmat_to_qvec_wxyz(w2c[:3, :3])
        tvec = w2c[:3, 3]

        cameras_lines.append(f"{idx} PINHOLE {width} {height} {fx} {fy} {cx} {cy}")
        images_lines.append(
            f"{idx} {' '.join(map(str, qvec.tolist()))} {' '.join(map(str, tvec.tolist()))} {idx} {name}"
        )
        images_lines.append("")

    if missing:
        raise RuntimeError(f"Missing camera entries or test images for: {missing[:10]}")

    (sparse_eval_dir / "cameras.txt").write_text("\n".join(cameras_lines) + "\n", encoding="utf-8")
    (sparse_eval_dir / "images.txt").write_text("\n".join(images_lines) + "\n", encoding="utf-8")
    link_gt_sparse(source_path, original_data_dir)

    print(f"Wrote eval cameras/images to {sparse_eval_dir}")
    print(f"Prepared {len(test_names)} test views from {test_images_dir}")


if __name__ == "__main__":
    main()
