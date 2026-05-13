#!/usr/bin/env python3
"""Build geometry-prefused pseudo views before diffusion enhancement.

This is a non-learned substitute for the pseudo-view deblur/pre-filter stage in
BRPO. For each rendered pseudo view, the script pulls colors from its left and
right real reference views through the pseudo depth and camera poses. Pixels
that are projection-valid and depth-consistent are blended back into the raw
Gaussian render, producing a less hallucination-prone input for Difix3D.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV availability depends on env.
    cv2 = None


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _copy_manifest_file(manifest_dir: Path, output_dir: Path, value: str, folder: str | None = None) -> str:
    src = _resolve(manifest_dir, value)
    rel = Path(value)
    if rel.is_absolute():
        rel = Path(folder or "assets") / rel.name
    elif folder is not None:
        rel = Path(folder) / rel.name
    dst = output_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists() and src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return str(rel)


def _copy_manifest_dir(manifest_dir: Path, output_dir: Path, value: str) -> str:
    src = _resolve(manifest_dir, value)
    rel = Path(value)
    if rel.is_absolute():
        rel = Path(src.name)
    dst = output_dir / rel
    if src.exists() and src.resolve() != dst.resolve():
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return str(rel)


def _as_4x4(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = pose
        return out
    raise ValueError(f"Unsupported pose shape: {pose.shape}")


def _load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _load_gray(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(path)


def _save_gray(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)


def _save_heatmap(path: Path, value: np.ndarray) -> None:
    value_u8 = np.clip(value * 255.0, 0, 255).astype(np.uint8)
    if cv2 is not None:
        heat = cv2.applyColorMap(value_u8, cv2.COLORMAP_TURBO)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        Image.fromarray(heat).save(path)
    else:
        Image.fromarray(value_u8, mode="L").save(path)


def _sample(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.remap(
            image.astype(np.float32),
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    h, w = image.shape[:2]
    x0 = np.floor(map_x).astype(np.int64)
    y0 = np.floor(map_y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)
    wa = (x1 - map_x) * (y1 - map_y)
    wb = (x1 - map_x) * (map_y - y0)
    wc = (map_x - x0) * (y1 - map_y)
    wd = (map_x - x0) * (map_y - y0)
    if image.ndim == 3:
        valid = valid[..., None]
        wa = wa[..., None]
        wb = wb[..., None]
        wc = wc[..., None]
        wd = wd[..., None]
    out = wa * image[y0c, x0c] + wb * image[y1c, x0c] + wc * image[y0c, x1c] + wd * image[y1c, x1c]
    return np.where(valid, out, 0.0)


def _find_source_image(source_images_dir: Path, image_name: str) -> Path:
    stem = Path(image_name).stem
    candidates = [
        source_images_dir / image_name,
        source_images_dir / "images" / image_name,
        source_images_dir / f"{stem}.png",
        source_images_dir / "images" / f"{stem}.png",
        source_images_dir / f"{stem}.jpg",
        source_images_dir / "images" / f"{stem}.jpg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find source image for {image_name} under {source_images_dir}")


def _load_cameras(model_path: Path) -> dict[str, dict]:
    cameras_path = model_path / "cameras.json"
    if not cameras_path.exists():
        return {}
    with cameras_path.open("r", encoding="utf-8") as f:
        cameras = json.load(f)
    return {Path(cam["img_name"]).stem: cam for cam in cameras}


def _project_pseudo_to_ref(
    pseudo_depth: np.ndarray,
    pseudo_w2c: np.ndarray,
    ref_w2c: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = pseudo_depth.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    z = pseudo_depth.astype(np.float32)
    valid = np.isfinite(z) & (z > 1e-5)

    x_cam = (xx - cx) / fx * z
    y_cam = (yy - cy) / fy * z
    points_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=-1).reshape(-1, 4).T
    points_world = np.linalg.inv(pseudo_w2c) @ points_cam
    points_ref = ref_w2c @ points_world

    x_ref = points_ref[0].reshape(h, w)
    y_ref = points_ref[1].reshape(h, w)
    z_ref = points_ref[2].reshape(h, w)
    map_x = fx * (x_ref / np.maximum(z_ref, 1e-6)) + cx
    map_y = fy * (y_ref / np.maximum(z_ref, 1e-6)) + cy
    valid &= np.isfinite(z_ref) & (z_ref > 1e-5)
    valid &= np.isfinite(map_x) & np.isfinite(map_y)
    valid &= (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
    return map_x.astype(np.float32), map_y.astype(np.float32), z_ref.astype(np.float32), valid


def _warp_ref_to_pseudo(
    ref_image: np.ndarray,
    pseudo_depth: np.ndarray,
    ref_depth: np.ndarray | None,
    alpha: np.ndarray,
    pseudo_w2c: np.ndarray,
    ref_w2c: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_tau: float,
    alpha_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    map_x, map_y, z_ref, valid = _project_pseudo_to_ref(pseudo_depth, pseudo_w2c, ref_w2c, fx, fy, cx, cy)
    warped = _sample(ref_image, map_x, map_y)
    depth_conf = np.ones_like(pseudo_depth, dtype=np.float32)
    depth_err = np.zeros_like(pseudo_depth, dtype=np.float32)
    if ref_depth is not None:
        sampled_depth = _sample(ref_depth.astype(np.float32), map_x, map_y)
        depth_valid = sampled_depth > 1e-5
        denom = 0.5 * (np.abs(z_ref) + np.abs(sampled_depth)) + 1e-6
        depth_err = np.abs(z_ref - sampled_depth) / denom
        depth_conf = np.exp(-depth_err / max(depth_tau, 1e-6)).astype(np.float32)
        valid &= depth_valid

    valid &= alpha >= alpha_threshold
    conf = np.where(valid, alpha * depth_conf, 0.0).astype(np.float32)
    stats = {
        "valid_ratio": float(valid.mean()),
        "depth_error_mean": float(depth_err[valid].mean()) if valid.any() and ref_depth is not None else 0.0,
        "conf_mean": float(conf.mean()),
    }
    return warped, conf, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo_manifest", type=Path, required=True)
    parser.add_argument("--source_images_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--alpha_threshold", type=float, default=0.2)
    parser.add_argument("--depth_tau", type=float, default=0.15)
    parser.add_argument("--confidence_threshold", type=float, default=0.03)
    parser.add_argument("--confidence_blur", type=float, default=1.0)
    parser.add_argument("--mix_strength", type=float, default=0.85)
    parser.add_argument("--min_confidence_mean", type=float, default=0.0)
    args = parser.parse_args()

    manifest_path = args.pseudo_manifest.resolve()
    manifest_dir = manifest_path.parent
    output_dir = (args.output_dir or (manifest_dir / "geometry_prefused")).resolve()
    images_dir = output_dir / "images"
    warped_left_dir = output_dir / "warped_left"
    warped_right_dir = output_dir / "warped_right"
    confidence_dir = output_dir / "confidence"
    confidence_vis_dir = output_dir / "confidence_vis"
    for folder in [images_dir, warped_left_dir, warped_right_dir, confidence_dir, confidence_vis_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    pose_src = _resolve(manifest_dir, manifest["pose_path"])
    source_pose_src = _resolve(manifest_dir, manifest["source_pose_path"])
    pseudo_poses = np.load(pose_src)
    source_poses = np.load(source_pose_src)

    pose_dst = output_dir / pose_src.name
    if pose_src.resolve() != pose_dst.resolve():
        shutil.copy2(pose_src, pose_dst)
    source_pose_dst = output_dir / source_pose_src.name
    if source_pose_src.resolve() != source_pose_dst.resolve():
        shutil.copy2(source_pose_src, source_pose_dst)
    manifest["pose_path"] = pose_dst.name
    manifest["source_pose_path"] = source_pose_dst.name

    model_path = _resolve(manifest_dir, manifest.get("source_model_path", ".."))
    cameras = _load_cameras(model_path)
    default_cam = next(iter(cameras.values())) if cameras else None
    ref_depth_dir_value = manifest.get("ref_depth_dir")
    ref_depth_dir = _resolve(manifest_dir, ref_depth_dir_value) if ref_depth_dir_value else None
    for dir_key in ("ref_depth_dir", "ref_alpha_dir", "ref_depth_vis_dir"):
        if manifest.get(dir_key):
            manifest[dir_key] = _copy_manifest_dir(manifest_dir, output_dir, manifest[dir_key])

    kept_views = []
    stats_all = []
    for view in manifest["views"]:
        image_name = view.get("image_name", f"pseudo_{int(view['pose_index']):05d}")
        raw = _load_rgb(_resolve(manifest_dir, view["raw_image_path"]))
        h, w = raw.shape[:2]
        size = (w, h)
        alpha = _load_gray(_resolve(manifest_dir, view["alpha_path"]), size=size)
        pseudo_depth = np.load(_resolve(manifest_dir, view["depth_path"])).astype(np.float32)

        left_image_name = view["left_image"]
        right_image_name = view["right_image"]
        left_ref = _load_rgb(_find_source_image(args.source_images_dir, left_image_name), size=size)
        right_ref = _load_rgb(_find_source_image(args.source_images_dir, right_image_name), size=size)

        left_stem = Path(left_image_name).stem
        right_stem = Path(right_image_name).stem
        cam_entry = cameras.get(left_stem) or cameras.get(right_stem) or default_cam
        if cam_entry is None:
            fx = fy = 0.5 * max(w, h)
            cx = 0.5 * w
            cy = 0.5 * h
        else:
            scale_x = w / float(cam_entry.get("width", w))
            scale_y = h / float(cam_entry.get("height", h))
            fx = float(cam_entry["fx"]) * scale_x
            fy = float(cam_entry["fy"]) * scale_y
            cx = 0.5 * w
            cy = 0.5 * h

        left_depth = None
        right_depth = None
        if ref_depth_dir is not None:
            left_depth_path = ref_depth_dir / f"{left_stem}.npy"
            right_depth_path = ref_depth_dir / f"{right_stem}.npy"
            if left_depth_path.exists():
                left_depth = np.load(left_depth_path).astype(np.float32)
            if right_depth_path.exists():
                right_depth = np.load(right_depth_path).astype(np.float32)

        pseudo_w2c = _as_4x4(pseudo_poses[int(view["pose_index"])])
        left_w2c = _as_4x4(source_poses[int(view["left_train_index"])])
        right_w2c = _as_4x4(source_poses[int(view["right_train_index"])])

        warped_left, left_conf, left_stats = _warp_ref_to_pseudo(
            left_ref, pseudo_depth, left_depth, alpha, pseudo_w2c, left_w2c,
            fx, fy, cx, cy, args.depth_tau, args.alpha_threshold,
        )
        warped_right, right_conf, right_stats = _warp_ref_to_pseudo(
            right_ref, pseudo_depth, right_depth, alpha, pseudo_w2c, right_w2c,
            fx, fy, cx, cy, args.depth_tau, args.alpha_threshold,
        )

        t = float(view.get("interval_t", 0.5))
        left_weight = left_conf * max(1.0 - t, 1e-3)
        right_weight = right_conf * max(t, 1e-3)
        denom = left_weight + right_weight
        ref_fused = np.where(
            denom[..., None] > 1e-8,
            (warped_left * left_weight[..., None] + warped_right * right_weight[..., None]) /
            np.maximum(denom[..., None], 1e-8),
            raw,
        )

        confidence = np.clip(denom, 0.0, 1.0)
        confidence = np.where(confidence >= args.confidence_threshold, confidence, 0.0)
        if args.confidence_blur > 0:
            pil = Image.fromarray(np.clip(confidence * 255.0, 0, 255).astype(np.uint8), mode="L")
            pil = pil.filter(ImageFilter.GaussianBlur(radius=args.confidence_blur))
            confidence = np.asarray(pil, dtype=np.float32) / 255.0
        confidence = np.clip(confidence * args.mix_strength, 0.0, 1.0)
        prefused = raw * (1.0 - confidence[..., None]) + ref_fused * confidence[..., None]

        confidence_mean = float(confidence.mean())
        view_stats = {
            "image_name": image_name,
            "confidence_mean": confidence_mean,
            "left_conf_mean": left_stats["conf_mean"],
            "right_conf_mean": right_stats["conf_mean"],
            "left_valid_ratio": left_stats["valid_ratio"],
            "right_valid_ratio": right_stats["valid_ratio"],
            "left_depth_error_mean": left_stats["depth_error_mean"],
            "right_depth_error_mean": right_stats["depth_error_mean"],
        }
        stats_all.append(view_stats)
        if confidence_mean < args.min_confidence_mean:
            continue

        image_rel = Path("images") / f"{image_name}.png"
        confidence_rel = Path("confidence") / f"{image_name}.png"
        confidence_vis_rel = Path("confidence_vis") / f"{image_name}.png"
        _save_rgb(output_dir / image_rel, prefused)
        _save_rgb(warped_left_dir / f"{image_name}.png", warped_left)
        _save_rgb(warped_right_dir / f"{image_name}.png", warped_right)
        _save_gray(output_dir / confidence_rel, confidence)
        _save_heatmap(output_dir / confidence_vis_rel, confidence)

        view["original_raw_image_path"] = _copy_manifest_file(
            manifest_dir, output_dir, view["raw_image_path"], folder="original_raw"
        )
        for aux_key in ("alpha_path", "depth_path", "depth_vis_path"):
            if view.get(aux_key):
                view[aux_key] = _copy_manifest_file(manifest_dir, output_dir, view[aux_key])
        view["raw_image_path"] = str(image_rel)
        view["geometry_confidence_path"] = str(confidence_rel)
        view["geometry_confidence_vis_path"] = str(confidence_vis_rel)
        view["geometry_prefusion"] = view_stats
        kept_views.append(view)

    manifest["views"] = kept_views
    manifest["geometry_prefusion"] = {
        "alpha_threshold": args.alpha_threshold,
        "depth_tau": args.depth_tau,
        "confidence_threshold": args.confidence_threshold,
        "confidence_blur": args.confidence_blur,
        "mix_strength": args.mix_strength,
        "min_confidence_mean": args.min_confidence_mean,
    }

    out_manifest = output_dir / "pseudo_manifest_geometry_prefused.json"
    with out_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    summary = {
        "num_input_views": len(stats_all),
        "num_kept_views": len(kept_views),
        "confidence_mean_min": min((v["confidence_mean"] for v in stats_all), default=0.0),
        "confidence_mean_mean": float(np.mean([v["confidence_mean"] for v in stats_all])) if stats_all else 0.0,
        "confidence_mean_max": max((v["confidence_mean"] for v in stats_all), default=0.0),
        "views": stats_all,
    }
    with (output_dir / "geometry_prefusion_stats.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[pseudo] geometry-prefused views: kept {len(kept_views)}/{len(stats_all)}")
    print(
        "[pseudo] geometry confidence mean: "
        f"min={summary['confidence_mean_min']:.4f}, "
        f"mean={summary['confidence_mean_mean']:.4f}, "
        f"max={summary['confidence_mean_max']:.4f}"
    )
    print(f"[pseudo] wrote: {out_manifest}")


if __name__ == "__main__":
    main()
