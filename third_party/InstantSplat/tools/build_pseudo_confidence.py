#!/usr/bin/env python3
"""Build BRPO-style fused pseudo images and soft confidence masks.

This script fuses left/right enhanced pseudo candidates with geometry-aware confidence.
It uses pseudo alpha, pseudo depth, source images, source poses, and optional rendered
reference depths to down-weight regions that do not reproject consistently.
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
except Exception:  # pragma: no cover - cv2 availability depends on env
    cv2 = None


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


def _load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _load_gray(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size is not None and img.size != size:
        img = img.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(path)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)


def _save_heatmap(path: Path, value: np.ndarray) -> None:
    value_u8 = np.clip(value * 255.0, 0, 255).astype(np.uint8)
    if cv2 is not None:
        heat = cv2.applyColorMap(value_u8, cv2.COLORMAP_TURBO)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        Image.fromarray(heat).save(path)
    else:
        Image.fromarray(value_u8, mode="L").save(path)


def _resize_np(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if image.shape[:2] == (h, w):
        return image
    pil_mode = "RGB" if image.ndim == 3 else "F"
    img = Image.fromarray(image.astype(np.float32), mode=pil_mode)
    img = img.resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32)


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

    # Bilinear fallback. Slower, but keeps the script usable without OpenCV.
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
    candidates = [
        source_images_dir / image_name,
        source_images_dir / "images" / image_name,
        source_images_dir / f"{Path(image_name).stem}.png",
        source_images_dir / "images" / f"{Path(image_name).stem}.png",
        source_images_dir / f"{Path(image_name).stem}.jpg",
        source_images_dir / "images" / f"{Path(image_name).stem}.jpg",
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
    pseudo_c2w = np.linalg.inv(pseudo_w2c)
    points_world = pseudo_c2w @ points_cam
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


def _side_confidence(
    candidate: np.ndarray,
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
    rgb_tau: float,
    depth_tau: float,
    alpha_threshold: float,
) -> tuple[np.ndarray, dict[str, float]]:
    map_x, map_y, z_ref, valid = _project_pseudo_to_ref(pseudo_depth, pseudo_w2c, ref_w2c, fx, fy, cx, cy)
    sampled_ref = _sample(ref_image, map_x, map_y)
    rgb_err = np.mean(np.abs(candidate - sampled_ref), axis=2)
    rgb_conf = np.exp(-rgb_err / max(rgb_tau, 1e-6))

    depth_conf = np.ones_like(rgb_conf, dtype=np.float32)
    depth_err = np.zeros_like(rgb_conf, dtype=np.float32)
    if ref_depth is not None:
        sampled_depth = _sample(ref_depth.astype(np.float32), map_x, map_y)
        depth_valid = sampled_depth > 1e-5
        denom = 0.5 * (np.abs(z_ref) + np.abs(sampled_depth)) + 1e-6
        depth_err = np.abs(z_ref - sampled_depth) / denom
        depth_conf = np.exp(-depth_err / max(depth_tau, 1e-6))
        valid &= depth_valid

    valid &= alpha >= alpha_threshold
    conf = alpha * rgb_conf * depth_conf
    conf = np.where(valid, conf, 0.0).astype(np.float32)
    stats = {
        "rgb_error_mean": float(rgb_err[valid].mean()) if valid.any() else 1.0,
        "depth_error_mean": float(depth_err[valid].mean()) if valid.any() and ref_depth is not None else 0.0,
        "valid_ratio": float(valid.mean()),
        "conf_mean": float(conf.mean()),
    }
    return conf, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo_manifest", type=Path, required=True)
    parser.add_argument("--source_images_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--enhanced_left_dir", type=Path, required=True)
    parser.add_argument("--enhanced_right_dir", type=Path, required=True)
    parser.add_argument("--alpha_threshold", type=float, default=0.2)
    parser.add_argument("--confidence_threshold", type=float, default=0.03)
    parser.add_argument("--agreement_tau", type=float, default=0.08)
    parser.add_argument("--rgb_tau", type=float, default=0.12)
    parser.add_argument("--depth_tau", type=float, default=0.15)
    parser.add_argument("--mask_blur", type=float, default=1.0)
    parser.add_argument("--min_mask_mean", type=float, default=0.0)
    parser.add_argument("--hard_mask", action="store_true")
    parser.add_argument(
        "--mask_mode",
        choices=("continuous", "trinary"),
        default="continuous",
        help="continuous keeps soft confidence; trinary writes BRPO-like 1/0.5/0 masks.",
    )
    parser.add_argument(
        "--side_conf_threshold",
        type=float,
        default=0.2,
        help="Per-side confidence threshold used by --mask_mode trinary.",
    )
    parser.add_argument(
        "--agreement_threshold",
        type=float,
        default=0.35,
        help="Left/right image agreement threshold used by --mask_mode trinary.",
    )
    parser.add_argument(
        "--set_loss_weight_from_mask",
        action="store_true",
        help="Set each pseudo view loss_weight from its average mask confidence.",
    )
    parser.add_argument("--loss_weight_scale", type=float, default=2.0)
    parser.add_argument("--min_loss_weight", type=float, default=0.2)
    parser.add_argument("--max_loss_weight", type=float, default=1.0)
    args = parser.parse_args()

    manifest_path = args.pseudo_manifest.resolve()
    manifest_dir = manifest_path.parent
    output_dir = (args.output_dir or manifest_dir).resolve()
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    confidence_dir = output_dir / "confidence_vis"
    for folder in [images_dir, masks_dir, confidence_dir]:
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

    kept_views = []
    stats_all = []
    input_view_count = 0
    ref_depth_dir_value = manifest.get("ref_depth_dir")
    ref_depth_dir = _resolve(manifest_dir, ref_depth_dir_value) if ref_depth_dir_value else None

    for view in manifest["views"]:
        input_view_count += 1
        image_name = view.get("image_name", f"pseudo_{int(view['pose_index']):05d}")
        left_path = args.enhanced_left_dir / f"{image_name}.png"
        right_path = args.enhanced_right_dir / f"{image_name}.png"
        if not left_path.exists() or not right_path.exists():
            raise FileNotFoundError(f"Missing enhanced pair for {image_name}")

        alpha = _load_gray(_resolve(manifest_dir, view["alpha_path"]))
        pseudo_depth = np.load(_resolve(manifest_dir, view["depth_path"])).astype(np.float32)
        h, w = pseudo_depth.shape
        size = (w, h)
        alpha = _resize_np(alpha, (h, w))
        left = _load_rgb(left_path, size=size)
        right = _load_rgb(right_path, size=size)

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

        pseudo_w2c = _as_4x4(pseudo_poses[int(view["pose_index"])])
        left_w2c = _as_4x4(source_poses[int(view["left_train_index"])])
        right_w2c = _as_4x4(source_poses[int(view["right_train_index"])])

        left_ref_depth = None
        right_ref_depth = None
        if ref_depth_dir is not None:
            left_depth_path = ref_depth_dir / f"{left_stem}.npy"
            right_depth_path = ref_depth_dir / f"{right_stem}.npy"
            if left_depth_path.exists():
                left_ref_depth = np.load(left_depth_path).astype(np.float32)
            if right_depth_path.exists():
                right_ref_depth = np.load(right_depth_path).astype(np.float32)

        left_conf, left_stats = _side_confidence(
            left, left_ref, pseudo_depth, left_ref_depth, alpha, pseudo_w2c, left_w2c,
            fx, fy, cx, cy, args.rgb_tau, args.depth_tau, args.alpha_threshold,
        )
        right_conf, right_stats = _side_confidence(
            right, right_ref, pseudo_depth, right_ref_depth, alpha, pseudo_w2c, right_w2c,
            fx, fy, cx, cy, args.rgb_tau, args.depth_tau, args.alpha_threshold,
        )

        agreement = np.exp(-np.mean(np.abs(left - right), axis=2) / max(args.agreement_tau, 1e-6))
        t = float(view.get("interval_t", 0.5))
        left_weight = left_conf * max(1.0 - t, 1e-3)
        right_weight = right_conf * max(t, 1e-3)
        denom = left_weight + right_weight
        avg = 0.5 * (left + right)
        fused = np.where(
            denom[..., None] > 1e-8,
            (left * left_weight[..., None] + right * right_weight[..., None]) / np.maximum(denom[..., None], 1e-8),
            avg,
        )

        if args.mask_mode == "trinary":
            agreement_ok = agreement >= args.agreement_threshold
            left_ok = (left_conf >= args.side_conf_threshold) & agreement_ok
            right_ok = (right_conf >= args.side_conf_threshold) & agreement_ok
            both_ok = left_ok & right_ok
            one_ok = left_ok ^ right_ok
            only_left = left_ok & ~right_ok
            only_right = right_ok & ~left_ok
            mask = np.zeros_like(agreement, dtype=np.float32)
            mask[both_ok] = 1.0
            mask[one_ok] = 0.5
            fused = np.where(
                only_left[..., None],
                left,
                np.where(only_right[..., None], right, fused),
            )
            mask_source = "brpo_trinary_reprojection_rgb_depth_agreement"
        else:
            mask = np.maximum(left_conf, right_conf) * agreement
            mask = np.where(mask >= args.confidence_threshold, mask, 0.0)
            mask_source = "alpha_x_reprojection_rgb_depth_x_bidirectional_agreement"
        if args.mask_blur > 0:
            pil = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L")
            pil = pil.filter(ImageFilter.GaussianBlur(radius=args.mask_blur))
            mask = np.asarray(pil, dtype=np.float32) / 255.0
        if args.hard_mask:
            mask = np.where(mask >= 0.5, 1.0, 0.0)

        mask_mean = float(mask.mean())
        loss_weight = float(view.get("loss_weight", 1.0))
        if args.set_loss_weight_from_mask:
            loss_weight = float(
                np.clip(mask_mean * args.loss_weight_scale, args.min_loss_weight, args.max_loss_weight)
            )
        if mask_mean < args.min_mask_mean:
            stats_all.append({
                **view,
                "mask_mean": mask_mean,
                "loss_weight": loss_weight,
                "filtered": True,
            })
            continue

        image_rel = Path("images") / f"{image_name}.png"
        mask_rel = Path("masks") / f"{image_name}.png"
        confidence_rel = Path("confidence_vis") / f"{image_name}.png"
        _save_rgb(output_dir / image_rel, fused)
        _save_mask(output_dir / mask_rel, mask)
        _save_heatmap(output_dir / confidence_rel, mask)

        view["image_path"] = str(image_rel)
        view["mask_path"] = str(mask_rel)
        view["confidence_vis_path"] = str(confidence_rel)
        view["mask_source"] = mask_source
        view["mask_mean"] = mask_mean
        view["loss_weight"] = loss_weight
        view["left_conf_mean"] = left_stats["conf_mean"]
        view["right_conf_mean"] = right_stats["conf_mean"]
        view["left_valid_ratio"] = left_stats["valid_ratio"]
        view["right_valid_ratio"] = right_stats["valid_ratio"]
        view["left_rgb_error_mean"] = left_stats["rgb_error_mean"]
        view["right_rgb_error_mean"] = right_stats["rgb_error_mean"]
        view["left_depth_error_mean"] = left_stats["depth_error_mean"]
        view["right_depth_error_mean"] = right_stats["depth_error_mean"]
        kept_views.append(view)
        stats_all.append(view.copy())

    manifest["views"] = kept_views
    manifest["confidence_fusion"] = {
        "alpha_threshold": args.alpha_threshold,
        "confidence_threshold": args.confidence_threshold,
        "agreement_tau": args.agreement_tau,
        "rgb_tau": args.rgb_tau,
        "depth_tau": args.depth_tau,
        "mask_blur": args.mask_blur,
        "min_mask_mean": args.min_mask_mean,
        "mask_mode": args.mask_mode,
        "side_conf_threshold": args.side_conf_threshold,
        "agreement_threshold": args.agreement_threshold,
        "set_loss_weight_from_mask": args.set_loss_weight_from_mask,
        "loss_weight_scale": args.loss_weight_scale,
        "min_loss_weight": args.min_loss_weight,
        "max_loss_weight": args.max_loss_weight,
    }

    train_manifest = output_dir / "pseudo_manifest_train.json"
    with train_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    summary = {
        "num_input_views": input_view_count,
        "num_kept_views": len(kept_views),
        "mask_mean_min": min((v["mask_mean"] for v in kept_views), default=0.0),
        "mask_mean_mean": float(np.mean([v["mask_mean"] for v in kept_views])) if kept_views else 0.0,
        "mask_mean_max": max((v["mask_mean"] for v in kept_views), default=0.0),
        "loss_weight_min": min((v["loss_weight"] for v in kept_views), default=0.0),
        "loss_weight_mean": float(np.mean([v["loss_weight"] for v in kept_views])) if kept_views else 0.0,
        "loss_weight_max": max((v["loss_weight"] for v in kept_views), default=0.0),
        "views": stats_all,
    }
    with (output_dir / "confidence_stats.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[pseudo] confidence fused pseudo views: kept {len(kept_views)}/{len(stats_all)}")
    print(
        "[pseudo] mask mean: "
        f"min={summary['mask_mean_min']:.4f}, "
        f"mean={summary['mask_mean_mean']:.4f}, "
        f"max={summary['mask_mean_max']:.4f}"
    )
    print(f"[pseudo] wrote: {train_manifest}")


if __name__ == "__main__":
    main()
