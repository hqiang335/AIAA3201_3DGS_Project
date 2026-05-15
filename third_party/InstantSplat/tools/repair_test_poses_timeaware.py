#!/usr/bin/env python3
"""Rewrite sparse_<n>/1 test poses using frame-index-aware interpolation.

This fixes old eval splits whose test poses were sampled uniformly along the
training pose list rather than matched to each held-out frame's temporal index.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.colmap_loader import qvec2rotmat, read_extrinsics_text
from utils.camera_utils import viewmatrix
from utils.sfm_utils import save_extrinsic


def interpolate_pose_pair_by_alpha(left_pose, right_pose, alpha, rot_weight=0.1):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    poses = np.stack([left_pose[:3, :4], right_pose[:3, :4]], axis=0)
    pos = poses[:, :3, 3]
    lookat = pos - rot_weight * poses[:, :3, 2]
    up = pos + rot_weight * poses[:, :3, 1]

    p = (1.0 - alpha) * pos[0] + alpha * pos[1]
    l = (1.0 - alpha) * lookat[0] + alpha * lookat[1]
    u = (1.0 - alpha) * up[0] + alpha * up[1]
    return viewmatrix(p - l, u - p, p)


def interpolate_test_poses_by_frame_indices(train_poses, train_indices, test_indices):
    train_indices = np.asarray(train_indices, dtype=np.float64)
    test_indices = np.asarray(test_indices, dtype=np.float64)
    test_poses = []
    for test_idx in test_indices:
        right = int(np.searchsorted(train_indices, test_idx, side="left"))
        if right <= 0:
            test_poses.append(train_poses[0][:3, :4])
            continue
        if right >= len(train_indices):
            test_poses.append(train_poses[-1][:3, :4])
            continue
        left = right - 1
        denom = max(train_indices[right] - train_indices[left], 1.0)
        alpha = (test_idx - train_indices[left]) / denom
        test_poses.append(interpolate_pose_pair_by_alpha(train_poses[left], train_poses[right], alpha))
    return np.asarray(test_poses, dtype=np.float32).reshape(-1, 3, 4)


def _load_train_w2c_poses(images_txt: Path) -> tuple[list[str], np.ndarray]:
    extrinsics = read_extrinsics_text(str(images_txt))
    extrinsics = {k: v for k, v in sorted(extrinsics.items(), key=lambda item: item[1].name)}
    names = []
    poses = []
    for extr in extrinsics.values():
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = qvec2rotmat(extr.qvec).astype(np.float32)
        pose[:3, 3] = np.asarray(extr.tvec, dtype=np.float32)
        names.append(extr.name)
        poses.append(pose)
    return names, np.stack(poses, axis=0)


def _backup(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_name(f"{path.name}.before_timeaware_{stamp}")
    shutil.copy2(path, backup_path)
    print(f"[backup] {path} -> {backup_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--source_path", required=True, type=Path)
    parser.add_argument("-m", "--model_path", required=True, type=Path)
    parser.add_argument("--n_train", "--n_views", dest="n_train", required=True, type=int)
    parser.add_argument("--no_backup", action="store_true")
    args = parser.parse_args()

    source_path = args.source_path.resolve()
    model_path = args.model_path.resolve()
    split_path = model_path / "split_manifest.json"
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    with split_path.open("r", encoding="utf-8") as f:
        split = json.load(f)

    train_indices = split["train_indices"]
    test_indices = split["test_indices"]
    test_basenames = split["test_basenames"]
    sparse_train = source_path / f"sparse_{args.n_train}" / "0"
    sparse_test = source_path / f"sparse_{args.n_train}" / "1"
    train_names, train_poses = _load_train_w2c_poses(sparse_train / "images.txt")
    if train_names != split["train_basenames"]:
        print("[warn] sparse train names differ from split manifest:")
        print("       sparse:", train_names)
        print("       split: ", split["train_basenames"])

    test_poses = interpolate_test_poses_by_frame_indices(train_poses, train_indices, test_indices)
    if not args.no_backup:
        _backup(sparse_test / "images.txt")
        _backup(sparse_test / "images.bin")

    image_files = [str(source_path / "images" / name) for name in test_basenames]
    image_suffix = Path(test_basenames[0]).suffix
    save_extrinsic(sparse_test, test_poses, image_files, image_suffix)

    print(f"[ok] wrote time-aware test poses: {sparse_test}")
    for idx, name in zip(test_indices, test_basenames):
        right = int(np.searchsorted(np.asarray(train_indices), idx, side="left"))
        left = max(0, min(right - 1, len(train_indices) - 1))
        right = max(0, min(right, len(train_indices) - 1))
        if left == right:
            alpha = 0.0
        else:
            alpha = (idx - train_indices[left]) / float(train_indices[right] - train_indices[left])
        print(
            f"  {name}: frame_idx={idx}, "
            f"left={split['train_basenames'][left]}, right={split['train_basenames'][right]}, alpha={alpha:.4f}"
        )


if __name__ == "__main__":
    main()
