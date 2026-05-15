#!/usr/bin/env python3
"""Build MASt3R feature-correspondence confidence masks for pseudo views.

The existing pseudo confidence tool checks RGB/depth reprojection consistency.
This script adds the BRPO-style signal that matters most for pseudo-view
training: whether a pseudo pixel has reliable feature correspondences to the
left and/or right real reference view.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

# This InstantSplat fork keeps DUSt3R as a flat package at repo_root/dust3r,
# while MASt3R's matching helpers expect the original nested submodule layout
# and import mast3r.utils.path_to_dust3r only to adjust sys.path.  The repo root
# is already on sys.path above, so a tiny shim avoids requiring a fake symlink.
sys.modules.setdefault("mast3r.utils.path_to_dust3r", types.ModuleType("mast3r.utils.path_to_dust3r"))

from mast3r.cloud_opt.sparse_ga import extract_correspondences, symmetric_inference
from mast3r.model import AsymmetricMASt3R
from utils.sfm_utils import load_images


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _source_images_dir(source_path: Path) -> Path:
    return source_path / "images" if (source_path / "images").is_dir() else source_path


def _find_source_image(source_images_dir: Path, name: str) -> Path:
    images_dir = _source_images_dir(source_images_dir)
    direct = images_dir / name
    if direct.exists():
        return direct
    stem = Path(name).stem
    for suffix in (".png", ".jpg", ".jpeg", ".JPG", ".PNG"):
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find source image {name} under {images_dir}")


def _copy_relative(src: Path, dst_dir: Path) -> str:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst.name


def _load_gray(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _save_gray(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)


def _save_heatmap(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = np.clip(mask, 0.0, 1.0)
    rgb = np.zeros((*m.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(255.0 * m, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(255.0 * (1.0 - np.abs(m - 0.5) * 2.0), 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(255.0 * (1.0 - m), 0, 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _copy_pose_files(manifest: dict, manifest_dir: Path, output_dir: Path) -> None:
    for key in ("pose_path", "source_pose_path"):
        if key not in manifest:
            continue
        src = _resolve(manifest_dir, manifest[key])
        if src.exists():
            manifest[key] = _copy_relative(src, output_dir)


def _match_pseudo_to_ref(
    model,
    pseudo_path: Path,
    ref_path: Path,
    device: str,
    image_size: int,
    subsample: int,
    desc_conf_key: str,
    min_match_conf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    imgs, _ = load_images(
        [str(pseudo_path), str(ref_path)],
        size=image_size,
        square_ok=True,
        verbose=False,
    )
    res = symmetric_inference(model, imgs[0], imgs[1], device=device)
    descs = [r["desc"][0] for r in res]
    qonfs = [r[desc_conf_key][0] for r in res]
    xy_pseudo, xy_ref, conf = extract_correspondences(descs, qonfs, device=device, subsample=subsample)
    xy_pseudo = _to_numpy(xy_pseudo).astype(np.float32)
    xy_ref = _to_numpy(xy_ref).astype(np.float32)
    conf = _to_numpy(conf).astype(np.float32)
    if min_match_conf > 0.0 and len(conf):
        keep = conf >= min_match_conf
        xy_pseudo = xy_pseudo[keep]
        xy_ref = xy_ref[keep]
        conf = conf[keep]
    loaded_h, loaded_w = [int(v) for v in imgs[0]["true_shape"][0]]
    return xy_pseudo, xy_ref, conf, (loaded_w, loaded_h)


def _points_to_mask(
    xy: np.ndarray,
    conf: np.ndarray,
    loaded_size: tuple[int, int],
    output_size: tuple[int, int],
    dilation_px: int,
    blur_px: float,
    percentile_norm: float,
) -> np.ndarray:
    out_w, out_h = output_size
    mask = np.zeros((out_h, out_w), dtype=np.float32)
    if len(xy) == 0:
        return mask

    loaded_w, loaded_h = loaded_size
    xs = np.clip(np.rint(xy[:, 0] * out_w / max(loaded_w, 1)).astype(np.int32), 0, out_w - 1)
    ys = np.clip(np.rint(xy[:, 1] * out_h / max(loaded_h, 1)).astype(np.int32), 0, out_h - 1)
    norm = float(np.percentile(conf, percentile_norm)) if len(conf) else 1.0
    if not np.isfinite(norm) or norm <= 1e-6:
        norm = float(conf.max()) if len(conf) else 1.0
    weights = np.clip(conf / max(norm, 1e-6), 0.0, 1.0)
    for x, y, weight in zip(xs, ys, weights):
        if weight > mask[y, x]:
            mask[y, x] = weight

    pil = Image.fromarray((mask * 255.0).astype(np.uint8), mode="L")
    if dilation_px > 0:
        kernel = max(3, int(dilation_px) | 1)
        pil = pil.filter(ImageFilter.MaxFilter(kernel))
    if blur_px > 0:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=float(blur_px)))
    return np.asarray(pil, dtype=np.float32) / 255.0


def _combine_masks(existing: np.ndarray | None, feature: np.ndarray, mode: str) -> np.ndarray:
    if existing is None or mode == "replace":
        return feature
    if mode == "multiply":
        return existing * feature
    if mode == "min":
        return np.minimum(existing, feature)
    raise ValueError(f"Unsupported combine mode: {mode}")


def _candidate_image_path(view: dict, manifest_dir: Path, image_key: str) -> Path:
    for key in (image_key, "image_path", "raw_image_path"):
        value = view.get(key)
        if value:
            path = _resolve(manifest_dir, value)
            if path.exists():
                return path
    raise FileNotFoundError(f"Pseudo view {view.get('image_name', '<unnamed>')} has no usable image path.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo_manifest", type=Path, required=True)
    parser.add_argument("--source_images_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=Path("./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--subsample", type=int, default=8)
    parser.add_argument("--desc_conf_key", type=str, default="desc_conf")
    parser.add_argument("--min_match_conf", type=float, default=0.0)
    parser.add_argument("--side_threshold", type=float, default=0.5)
    parser.add_argument("--dilation_px", type=int, default=3)
    parser.add_argument("--blur_px", type=float, default=0.0)
    parser.add_argument("--percentile_norm", type=float, default=95.0)
    parser.add_argument("--candidate_image_key", type=str, default="image_path")
    parser.add_argument("--combine_with_existing", choices=["replace", "min", "multiply"], default="min")
    parser.add_argument("--min_combined_mask_mean", type=float, default=0.0)
    parser.add_argument("--set_loss_weight_from_mask", action="store_true")
    parser.add_argument("--loss_weight_scale", type=float, default=1.5)
    parser.add_argument("--min_loss_weight", type=float, default=0.05)
    parser.add_argument("--max_loss_weight", type=float, default=1.0)
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    ckpt_path = args.ckpt_path if args.ckpt_path.is_absolute() else repo_root / args.ckpt_path
    manifest_path = args.pseudo_manifest.resolve()
    manifest_dir = manifest_path.parent
    output_dir = args.output_dir.resolve()
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    feature_dir = output_dir / "feature_masks"
    feature_left_dir = output_dir / "feature_left"
    feature_right_dir = output_dir / "feature_right"
    vis_dir = output_dir / "feature_vis"
    for folder in (image_dir, mask_dir, feature_dir, feature_left_dir, feature_right_dir, vis_dir):
        folder.mkdir(parents=True, exist_ok=True)

    manifest = _load_json(manifest_path)
    _copy_pose_files(manifest, manifest_dir, output_dir)
    views = list(manifest.get("views", []))
    if args.max_views > 0:
        views = views[: args.max_views]

    model = AsymmetricMASt3R.from_pretrained(str(ckpt_path)).to(args.device)
    model.eval()

    kept_views = []
    stats_all = []
    with torch.no_grad():
        for view in tqdm(views, desc="Feature correspondence masks"):
            image_name = view.get("image_name", f"pseudo_{len(stats_all):05d}")
            out_mask_path = mask_dir / f"{image_name}.png"
            if args.skip_existing and out_mask_path.exists():
                continue

            pseudo_path = _candidate_image_path(view, manifest_dir, args.candidate_image_key)
            with Image.open(pseudo_path) as pseudo_img:
                output_size = pseudo_img.size

            left_ref = _find_source_image(args.source_images_dir.resolve(), view["left_image"])
            right_ref = _find_source_image(args.source_images_dir.resolve(), view["right_image"])

            xy_l, _, conf_l, loaded_size_l = _match_pseudo_to_ref(
                model,
                pseudo_path,
                left_ref,
                args.device,
                args.image_size,
                args.subsample,
                args.desc_conf_key,
                args.min_match_conf,
            )
            xy_r, _, conf_r, loaded_size_r = _match_pseudo_to_ref(
                model,
                pseudo_path,
                right_ref,
                args.device,
                args.image_size,
                args.subsample,
                args.desc_conf_key,
                args.min_match_conf,
            )
            left_mask = _points_to_mask(
                xy_l,
                conf_l,
                loaded_size_l,
                output_size,
                args.dilation_px,
                args.blur_px,
                args.percentile_norm,
            )
            right_mask = _points_to_mask(
                xy_r,
                conf_r,
                loaded_size_r,
                output_size,
                args.dilation_px,
                args.blur_px,
                args.percentile_norm,
            )

            left_ok = left_mask >= args.side_threshold
            right_ok = right_mask >= args.side_threshold
            feature_mask = np.zeros_like(left_mask, dtype=np.float32)
            feature_mask[left_ok & right_ok] = 1.0
            feature_mask[left_ok ^ right_ok] = 0.5

            existing_mask = None
            previous_mask = view.get("mask_path")
            if previous_mask:
                previous_mask_path = _resolve(manifest_dir, previous_mask)
                if previous_mask_path.exists():
                    existing_mask = _load_gray(previous_mask_path, output_size)
            combined = _combine_masks(existing_mask, feature_mask, args.combine_with_existing)
            combined_mean = float(combined.mean())

            stats = {
                **view,
                "feature_left_matches": int(len(conf_l)),
                "feature_right_matches": int(len(conf_r)),
                "feature_left_mask_mean": float(left_mask.mean()),
                "feature_right_mask_mean": float(right_mask.mean()),
                "feature_mask_mean": float(feature_mask.mean()),
                "combined_mask_mean": combined_mean,
            }
            if combined_mean < args.min_combined_mask_mean:
                stats["filtered"] = True
                stats_all.append(stats)
                continue

            image_rel = Path("images") / f"{image_name}{pseudo_path.suffix.lower()}"
            if image_rel.suffix not in (".png", ".jpg", ".jpeg"):
                image_rel = Path("images") / f"{image_name}.png"
            dst_image = output_dir / image_rel
            if dst_image.suffix.lower() == pseudo_path.suffix.lower():
                shutil.copy2(pseudo_path, dst_image)
            else:
                Image.open(pseudo_path).convert("RGB").save(dst_image)

            mask_rel = Path("masks") / f"{image_name}.png"
            feature_rel = Path("feature_masks") / f"{image_name}.png"
            left_rel = Path("feature_left") / f"{image_name}.png"
            right_rel = Path("feature_right") / f"{image_name}.png"
            vis_rel = Path("feature_vis") / f"{image_name}.png"
            _save_gray(output_dir / mask_rel, combined)
            _save_gray(output_dir / feature_rel, feature_mask)
            _save_gray(output_dir / left_rel, left_mask)
            _save_gray(output_dir / right_rel, right_mask)
            _save_heatmap(output_dir / vis_rel, combined)

            new_view = {
                **view,
                "image_path": str(image_rel),
                "mask_path": str(mask_rel),
                "previous_mask_path": previous_mask or "",
                "feature_mask_path": str(feature_rel),
                "feature_left_path": str(left_rel),
                "feature_right_path": str(right_rel),
                "confidence_vis_path": str(vis_rel),
                "mask_source": f"mast3r_feature_correspondence_{args.combine_with_existing}",
                "feature_left_matches": int(len(conf_l)),
                "feature_right_matches": int(len(conf_r)),
                "feature_left_mask_mean": float(left_mask.mean()),
                "feature_right_mask_mean": float(right_mask.mean()),
                "feature_mask_mean": float(feature_mask.mean()),
                "mask_mean": combined_mean,
            }
            if args.set_loss_weight_from_mask:
                new_view["loss_weight"] = float(
                    np.clip(combined_mean * args.loss_weight_scale, args.min_loss_weight, args.max_loss_weight)
                )
            kept_views.append(new_view)
            stats_all.append(new_view.copy())

    manifest["views"] = kept_views
    manifest["feature_correspondence_mask"] = {
        "ckpt_path": str(ckpt_path),
        "image_size": args.image_size,
        "subsample": args.subsample,
        "desc_conf_key": args.desc_conf_key,
        "min_match_conf": args.min_match_conf,
        "side_threshold": args.side_threshold,
        "dilation_px": args.dilation_px,
        "blur_px": args.blur_px,
        "combine_with_existing": args.combine_with_existing,
        "min_combined_mask_mean": args.min_combined_mask_mean,
    }
    train_manifest = output_dir / "pseudo_manifest_train.json"
    _write_json(train_manifest, manifest)

    means = [float(v.get("mask_mean", 0.0)) for v in kept_views]
    summary = {
        "num_input_views": len(views),
        "num_kept_views": len(kept_views),
        "mask_mean_min": min(means) if means else 0.0,
        "mask_mean_mean": float(np.mean(means)) if means else 0.0,
        "mask_mean_max": max(means) if means else 0.0,
        "views": stats_all,
    }
    _write_json(output_dir / "feature_mask_stats.json", summary)
    print(f"[feature-mask] kept {len(kept_views)}/{len(views)}")
    print(
        "[feature-mask] mask mean: "
        f"min={summary['mask_mean_min']:.4f}, "
        f"mean={summary['mask_mean_mean']:.4f}, "
        f"max={summary['mask_mean_max']:.4f}"
    )
    print(f"[feature-mask] wrote: {train_manifest}")


if __name__ == "__main__":
    main()
