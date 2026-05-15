#!/usr/bin/env python3
"""Build an initialization point cloud from real MASt3R points plus pseudo RGB-D views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _as_4x4(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = pose
        return out
    raise ValueError(f"Unsupported pose shape: {pose.shape}")


def _read_simple_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)
    rgb = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype(np.uint8)
    return xyz, rgb


def _write_simple_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normals = np.zeros_like(xyz, dtype=np.float32)
    elements = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    attrs = np.concatenate([xyz.astype(np.float32), normals, rgb.astype(np.uint8)], axis=1)
    elements[:] = list(map(tuple, attrs))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def _read_intrinsics(cameras_txt: Path) -> tuple[float, float, float, float, int, int]:
    with cameras_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = [float(v) for v in parts[4:]]
            if model == "PINHOLE":
                fx, fy, cx, cy = params[:4]
            elif model == "SIMPLE_PINHOLE":
                fx = fy = params[0]
                cx, cy = params[1:3]
            else:
                raise ValueError(f"Unsupported camera model in {cameras_txt}: {model}")
            return fx, fy, cx, cy, width, height
    raise ValueError(f"No camera entry found in {cameras_txt}")


def _sample_indices(weights: np.ndarray, max_count: int, rng: np.random.Generator) -> np.ndarray:
    count = len(weights)
    if count <= max_count:
        return np.arange(count, dtype=np.int64)
    weights = weights.astype(np.float64)
    weights = np.maximum(weights, 1e-6)
    weights /= weights.sum()
    return rng.choice(count, size=max_count, replace=False, p=weights)


def _unproject_view(
    image_path: Path,
    mask_path: Path,
    depth_path: Path,
    w2c: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    min_mask: float,
    max_points: int,
    rng: np.random.Generator,
    depth_percentiles: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    depth = np.load(depth_path).astype(np.float32)
    depth = np.squeeze(depth)
    h, w = depth.shape
    if image.shape[:2] != (h, w):
        image = np.asarray(Image.fromarray(image).resize((w, h), Image.BILINEAR), dtype=np.uint8)
    if mask.shape != (h, w):
        mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.NEAREST), dtype=np.float32) / 255.0

    valid = np.isfinite(depth) & (depth > 0) & (mask >= min_mask)
    if valid.any() and (depth_percentiles[0] > 0 or depth_percentiles[1] < 100):
        valid_depths = depth[valid]
        lo, hi = np.percentile(valid_depths, depth_percentiles)
        valid &= (depth >= lo) & (depth <= hi)
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    chosen = _sample_indices(mask[ys, xs], max_points, rng)
    xs = xs[chosen].astype(np.float32)
    ys = ys[chosen].astype(np.float32)
    z = depth[ys.astype(np.int32), xs.astype(np.int32)].astype(np.float32)
    fx, fy, cx, cy = intrinsics
    x_cam = (xs - cx) / fx * z
    y_cam = (ys - cy) / fy * z
    cam = np.stack([x_cam, y_cam, z], axis=1)
    pose = _as_4x4(w2c)
    r = pose[:3, :3]
    t = pose[:3, 3]
    world = (cam - t[None, :]) @ r
    colors = image[ys.astype(np.int32), xs.astype(np.int32)]
    return world.astype(np.float32), colors.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_ply", type=Path, required=True)
    parser.add_argument("--pseudo_manifest", type=Path, required=True)
    parser.add_argument("--cameras_txt", type=Path, required=True)
    parser.add_argument("--output_ply", type=Path, required=True)
    parser.add_argument("--max_output_points", type=int, default=100000)
    parser.add_argument("--base_points", type=int, default=50000)
    parser.add_argument("--max_points_per_pseudo", type=int, default=3000)
    parser.add_argument("--min_mask", type=float, default=0.25)
    parser.add_argument("--depth_percentile_low", type=float, default=1.0)
    parser.add_argument("--depth_percentile_high", type=float, default=99.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    base_xyz, base_rgb = _read_simple_ply(args.base_ply.resolve())
    base_input_count = len(base_xyz)
    if args.base_points > 0 and len(base_xyz) > args.base_points:
        keep = rng.choice(len(base_xyz), size=args.base_points, replace=False)
        base_xyz = base_xyz[keep]
        base_rgb = base_rgb[keep]

    manifest_path = args.pseudo_manifest.resolve()
    manifest_dir = manifest_path.parent
    manifest = _load_json(manifest_path)
    poses = np.load(_resolve(manifest_dir, manifest["pose_path"]))
    intr = _read_intrinsics(args.cameras_txt.resolve())[:4]

    pseudo_xyz_all = []
    pseudo_rgb_all = []
    for view in manifest.get("views", []):
        depth_value = view.get("depth_path")
        if not depth_value:
            continue
        xyz, rgb = _unproject_view(
            _resolve(manifest_dir, view["image_path"]),
            _resolve(manifest_dir, view["mask_path"]),
            _resolve(manifest_dir, depth_value),
            poses[int(view["pose_index"])],
            intr,
            args.min_mask,
            args.max_points_per_pseudo,
            rng,
            (args.depth_percentile_low, args.depth_percentile_high),
        )
        if len(xyz):
            pseudo_xyz_all.append(xyz)
            pseudo_rgb_all.append(rgb)

    if pseudo_xyz_all:
        pseudo_xyz = np.concatenate(pseudo_xyz_all, axis=0)
        pseudo_rgb = np.concatenate(pseudo_rgb_all, axis=0)
    else:
        pseudo_xyz = np.empty((0, 3), dtype=np.float32)
        pseudo_rgb = np.empty((0, 3), dtype=np.uint8)

    max_pseudo = max(args.max_output_points - len(base_xyz), 0)
    if len(pseudo_xyz) > max_pseudo:
        keep = rng.choice(len(pseudo_xyz), size=max_pseudo, replace=False)
        pseudo_xyz = pseudo_xyz[keep]
        pseudo_rgb = pseudo_rgb[keep]

    out_xyz = np.concatenate([base_xyz, pseudo_xyz], axis=0)
    out_rgb = np.concatenate([base_rgb, pseudo_rgb], axis=0)
    if len(out_xyz) > args.max_output_points:
        keep = rng.choice(len(out_xyz), size=args.max_output_points, replace=False)
        out_xyz = out_xyz[keep]
        out_rgb = out_rgb[keep]

    _write_simple_ply(args.output_ply.resolve(), out_xyz, out_rgb)
    summary = {
        "base_input_points": int(base_input_count),
        "base_kept_points": int(len(base_xyz)),
        "pseudo_candidate_points": int(sum(len(x) for x in pseudo_xyz_all)),
        "pseudo_kept_points": int(len(pseudo_xyz)),
        "output_points": int(len(out_xyz)),
        "min_mask": args.min_mask,
        "max_points_per_pseudo": args.max_points_per_pseudo,
        "max_output_points": args.max_output_points,
    }
    summary_path = args.output_ply.resolve().with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[pseudo-init] wrote: {args.output_ply.resolve()}")


if __name__ == "__main__":
    main()
