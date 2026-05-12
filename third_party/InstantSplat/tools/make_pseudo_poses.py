#!/usr/bin/env python3
"""Generate sparse pseudo camera poses between neighboring training views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.interpolate


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    return np.stack([vec0, vec1, vec2, position], axis=1)


def generate_interpolated_path(
    poses: np.ndarray,
    n_interp: int,
    spline_degree: int = 5,
    smoothness: float = 0.0,
    rot_weight: float = 0.1,
) -> np.ndarray:
    """Creates a smooth spline path between input keyframe camera poses."""

    def poses_to_points(input_poses: np.ndarray, dist: float) -> np.ndarray:
        pos = input_poses[:, :3, -1]
        lookat = input_poses[:, :3, -1] - dist * input_poses[:, :3, 2]
        up = input_poses[:, :3, -1] + dist * input_poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points: np.ndarray) -> np.ndarray:
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    points = poses_to_points(poses, dist=rot_weight)
    sh = points.shape
    flat_points = np.reshape(points, (sh[0], -1))
    k = min(spline_degree, sh[0] - 1)
    tck, _ = scipy.interpolate.splprep(flat_points.T, k=k, s=smoothness)
    u = np.linspace(0, 1, n_interp * (points.shape[0] - 1), endpoint=False)
    new_points = np.array(scipy.interpolate.splev(u, tck))
    new_points = np.reshape(new_points.T, (len(u), sh[1], sh[2]))
    return points_to_poses(new_points)


def visualize_poses(camera_poses: np.ndarray, save_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for pose in camera_poses:
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        camera_position = np.linalg.inv(rotation) @ -translation
        ax.scatter(camera_position[0], camera_position[1], camera_position[2], c="blue", s=10)
    fig.savefig(save_path)
    plt.close(fig)


def _load_split_manifest(model_path: Path) -> dict:
    manifest_path = model_path / "split_manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_4x4(pose: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float32)
    out[:3, :4] = pose[:3, :4]
    return out


def _resolve_pose_path(model_path: Path, iteration: int, pose_path: str) -> Path:
    if pose_path:
        path = Path(pose_path)
        return path if path.is_absolute() else model_path / path
    return model_path / "pose" / f"ours_{iteration}" / "pose_optimized.npy"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model_path", required=True, type=Path)
    parser.add_argument("--iteration", type=int, default=1000)
    parser.add_argument("--pose_path", type=str, default="")
    parser.add_argument("--pseudo_per_interval", type=int, default=1)
    parser.add_argument("--max_pseudo_views", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--smoothness", type=float, default=0.0)
    parser.add_argument("--rot_weight", type=float, default=0.1)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    if args.pseudo_per_interval < 1:
        raise ValueError("--pseudo_per_interval must be >= 1")

    model_path = args.model_path.resolve()
    output_dir = args.output_dir or (model_path / "pseudo_views")
    output_dir.mkdir(parents=True, exist_ok=True)

    pose_path = _resolve_pose_path(model_path, args.iteration, args.pose_path)
    poses = np.load(pose_path)
    if poses.ndim != 3 or poses.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(f"Expected poses with shape [N,3,4] or [N,4,4], got {poses.shape}")
    poses = np.stack([_as_4x4(p) for p in poses], axis=0)

    split_manifest = _load_split_manifest(model_path)
    train_basenames = split_manifest.get("train_basenames", [f"train_{i:05d}.png" for i in range(len(poses))])
    if len(train_basenames) != len(poses):
        print(
            "[WARN] split_manifest train_basenames count does not match pose count: "
            f"{len(train_basenames)} vs {len(poses)}. Falling back to generated names."
        )
        train_basenames = [f"train_{i:05d}.png" for i in range(len(poses))]

    pseudo_poses = []
    pseudo_views = []
    for left_idx in range(len(poses) - 1):
        # generate_interpolated_path returns the left endpoint first and excludes the right endpoint.
        interp = generate_interpolated_path(
            poses=poses[left_idx : left_idx + 2, :3, :4],
            n_interp=args.pseudo_per_interval + 1,
            smoothness=args.smoothness,
            rot_weight=args.rot_weight,
        )
        for local_idx, pose_3x4 in enumerate(interp[1:], start=1):
            pseudo_idx = len(pseudo_poses)
            pseudo_poses.append(_as_4x4(pose_3x4))
            t = local_idx / float(args.pseudo_per_interval + 1)
            pseudo_views.append(
                {
                    "pseudo_id": pseudo_idx,
                    "pose_index": pseudo_idx,
                    "image_name": f"pseudo_{pseudo_idx:05d}",
                    "left_train_index": left_idx,
                    "right_train_index": left_idx + 1,
                    "left_image": train_basenames[left_idx],
                    "right_image": train_basenames[left_idx + 1],
                    "interval_t": t,
                    "loss_weight": 1.0,
                }
            )

    if args.max_pseudo_views > 0 and len(pseudo_poses) > args.max_pseudo_views:
        keep = np.linspace(0, len(pseudo_poses) - 1, args.max_pseudo_views, dtype=int)
        pseudo_poses = [pseudo_poses[i] for i in keep]
        pseudo_views = [pseudo_views[i] for i in keep]
        for new_idx, view in enumerate(pseudo_views):
            view["pseudo_id"] = new_idx
            view["pose_index"] = new_idx
            view["image_name"] = f"pseudo_{new_idx:05d}"

    if not pseudo_poses:
        raise ValueError("No pseudo poses generated. Need at least two source poses.")

    pseudo_poses_np = np.stack(pseudo_poses, axis=0).astype(np.float32)
    pose_out = output_dir / "pseudo_poses.npy"
    np.save(pose_out, pseudo_poses_np)

    manifest = {
        "schema": "instantsplat_pseudo_views_v1",
        "source_model_path": str(model_path),
        "source_pose_path": str(pose_path),
        "iteration": args.iteration,
        "pseudo_per_interval": args.pseudo_per_interval,
        "pose_path": pose_out.name,
        "views": pseudo_views,
    }
    manifest_path = output_dir / "pseudo_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if args.visualize:
        visualize_poses(poses, output_dir / "real_train_poses.png")
        visualize_poses(pseudo_poses_np, output_dir / "pseudo_poses.png")

    print(f"[pseudo] source poses: {len(poses)}")
    print(f"[pseudo] generated poses: {len(pseudo_poses_np)}")
    print(f"[pseudo] wrote: {pose_out}")
    print(f"[pseudo] wrote: {manifest_path}")


if __name__ == "__main__":
    main()
