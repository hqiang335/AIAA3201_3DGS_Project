#!/usr/bin/env python3
"""Build and compare pre-Difix pseudo-view deghosting variants.

The goal is to approximate BRPO's pseudo-view deblur/pre-filter stage before
reference-conditioned diffusion.  This script creates several non-learned
variants from the same raw Gaussian pseudo render and adjacent real frames,
then scores them by reprojection consistency and writes visual contact sheets.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_geometry_prefused_pseudo import (  # noqa: E402
    _as_4x4,
    _find_source_image,
    _load_cameras,
    _load_gray,
    _load_rgb,
    _resolve,
    _save_heatmap,
    _save_rgb,
    _warp_ref_to_pseudo,
)
from tools.build_pseudo_confidence import _side_confidence  # noqa: E402

try:
    import cv2
except Exception:  # pragma: no cover - depends on environment
    cv2 = None


def _save_gray(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)


def _smooth_conf(conf: np.ndarray, threshold: float, blur: float, strength: float) -> np.ndarray:
    conf = np.where(conf >= threshold, conf, 0.0).astype(np.float32)
    if blur > 0:
        pil = Image.fromarray(np.clip(conf * 255.0, 0, 255).astype(np.uint8), mode="L")
        pil = pil.filter(ImageFilter.GaussianBlur(radius=blur))
        conf = np.asarray(pil, dtype=np.float32) / 255.0
    return np.clip(conf * strength, 0.0, 1.0)


def _weighted_ref(
    raw: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    left_score: np.ndarray,
    right_score: np.ndarray,
) -> np.ndarray:
    denom = left_score + right_score
    return np.where(
        denom[..., None] > 1e-8,
        (left * left_score[..., None] + right * right_score[..., None]) / np.maximum(denom[..., None], 1e-8),
        raw,
    )


def _sharpness(image: np.ndarray) -> float:
    gray = (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(np.float32)
    if cv2 is not None:
        return float(cv2.Laplacian(gray, cv2.CV_32F).var())
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    return float((gx * gx + gy * gy).mean())


def _morph(mask: np.ndarray, dilate: int, close: int) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    if cv2 is not None:
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        if dilate > 0:
            kernel = np.ones((dilate, dilate), np.uint8)
            mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
    return (mask_u8 > 0).astype(np.float32)


def _inpaint(image: np.ndarray, mask: np.ndarray, radius: float) -> np.ndarray:
    if cv2 is None or mask.max() <= 0:
        return image
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    out = cv2.inpaint(image_u8, mask_u8, radius, cv2.INPAINT_TELEA)
    return out.astype(np.float32) / 255.0


def _build_variants(
    raw: np.ndarray,
    warped_left: np.ndarray,
    warped_right: np.ndarray,
    left_conf: np.ndarray,
    right_conf: np.ndarray,
    raw_left_conf: np.ndarray,
    raw_right_conf: np.ndarray,
    t: float,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    left_score = left_conf * max(1.0 - t, 1e-3)
    right_score = right_conf * max(t, 1e-3)
    soft_conf = _smooth_conf(left_score + right_score, args.confidence_threshold, args.confidence_blur, args.mix_strength)
    ref_fused = _weighted_ref(raw, warped_left, warped_right, left_score, right_score)

    best_is_left = left_score >= right_score
    best_score = np.maximum(left_score, right_score)
    best_warp = np.where(best_is_left[..., None], warped_left, warped_right)
    best_conf = _smooth_conf(best_score, args.confidence_threshold, args.confidence_blur, args.mix_strength)

    residual = np.clip(ref_fused - raw, -args.residual_clip, args.residual_clip)
    residual_conf = _smooth_conf(left_score + right_score, args.confidence_threshold, args.confidence_blur, args.residual_strength)

    agreement = np.exp(-np.mean(np.abs(warped_left - warped_right), axis=2) / max(args.agreement_tau, 1e-6))
    left_ok = left_conf >= args.side_conf_threshold
    right_ok = right_conf >= args.side_conf_threshold
    both_ok = left_ok & right_ok & (agreement >= args.agreement_threshold)
    only_left = left_ok & ~right_ok
    only_right = right_ok & ~left_ok
    one_or_both = both_ok | only_left | only_right
    agreement_select = np.where(
        both_ok[..., None],
        ref_fused,
        np.where(only_left[..., None], warped_left, np.where(only_right[..., None], warped_right, raw)),
    )

    hard_mask = best_score >= args.hard_replace_threshold
    raw_ref_score = np.maximum(
        raw_left_conf * max(1.0 - t, 1e-3),
        raw_right_conf * max(t, 1e-3),
    )
    suspicious = (
        (raw_ref_score < args.artifact_raw_conf_threshold)
        & (np.mean(np.abs(raw - ref_fused), axis=2) > args.artifact_delta_threshold)
        & ((left_conf > args.artifact_ref_conf_threshold) | (right_conf > args.artifact_ref_conf_threshold))
    )
    artifact_mask = _morph(suspicious.astype(np.float32), args.artifact_dilate, args.artifact_close)
    inpainted_raw = _inpaint(raw, artifact_mask, args.artifact_inpaint_radius)
    artifact_ref_fill = np.where(best_score[..., None] >= args.artifact_ref_conf_threshold, best_warp, inpainted_raw)
    artifact_inpaint = np.where(artifact_mask[..., None] > 0, artifact_ref_fill, raw)
    artifact_agreement = np.where(
        artifact_mask[..., None] > 0,
        np.where(one_or_both[..., None], agreement_select, artifact_ref_fill),
        np.where(one_or_both[..., None], agreement_select, raw),
    )
    variants = {
        "raw": raw,
        "soft_depth_blend": raw * (1.0 - soft_conf[..., None]) + ref_fused * soft_conf[..., None],
        "best_side_blend": raw * (1.0 - best_conf[..., None]) + best_warp * best_conf[..., None],
        "hard_best_side": np.where(hard_mask[..., None], best_warp, raw),
        "agreement_select": np.where(one_or_both[..., None], agreement_select, raw),
        "residual_clipped": np.clip(raw + residual_conf[..., None] * residual, 0.0, 1.0),
        "artifact_inpaint": artifact_inpaint,
        "artifact_agreement": artifact_agreement,
    }
    aux = {
        "soft_conf": soft_conf,
        "best_conf": best_conf,
        "hard_mask": hard_mask.astype(np.float32),
        "agreement_mask": one_or_both.astype(np.float32),
        "artifact_mask": artifact_mask,
        "raw_ref_score": raw_ref_score.astype(np.float32),
        "left_conf": left_conf,
        "right_conf": right_conf,
    }
    return variants, aux


def _metric_for_candidate(
    candidate: np.ndarray,
    raw: np.ndarray,
    left_ref: np.ndarray,
    right_ref: np.ndarray,
    pseudo_depth: np.ndarray,
    left_depth: np.ndarray | None,
    right_depth: np.ndarray | None,
    alpha: np.ndarray,
    pseudo_w2c: np.ndarray,
    left_w2c: np.ndarray,
    right_w2c: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    t: float,
    args: argparse.Namespace,
) -> dict[str, float]:
    left_conf, left_stats = _side_confidence(
        candidate, left_ref, pseudo_depth, left_depth, alpha, pseudo_w2c, left_w2c,
        fx, fy, cx, cy, args.rgb_tau, args.depth_tau, args.alpha_threshold,
    )
    right_conf, right_stats = _side_confidence(
        candidate, right_ref, pseudo_depth, right_depth, alpha, pseudo_w2c, right_w2c,
        fx, fy, cx, cy, args.rgb_tau, args.depth_tau, args.alpha_threshold,
    )
    wt_left = max(1.0 - t, 1e-3)
    wt_right = max(t, 1e-3)
    weighted_rgb_error = wt_left * left_stats["rgb_error_mean"] + wt_right * right_stats["rgb_error_mean"]
    weighted_conf = wt_left * left_stats["conf_mean"] + wt_right * right_stats["conf_mean"]
    return {
        "weighted_rgb_error": float(weighted_rgb_error),
        "weighted_conf": float(weighted_conf),
        "left_rgb_error": float(left_stats["rgb_error_mean"]),
        "right_rgb_error": float(right_stats["rgb_error_mean"]),
        "left_conf": float(left_stats["conf_mean"]),
        "right_conf": float(right_stats["conf_mean"]),
        "change_from_raw": float(np.mean(np.abs(candidate - raw))),
        "sharpness": _sharpness(candidate),
    }


def _write_manifest(template: dict, manifest_dir: Path, method_dir: Path, method: str, views: list[dict]) -> Path:
    manifest = copy.deepcopy(template)
    manifest["pre_difix_variant"] = method
    for key in ("pose_path", "source_pose_path", "ref_depth_dir", "ref_alpha_dir", "ref_depth_vis_dir"):
        if manifest.get(key):
            manifest[key] = str(_resolve(manifest_dir, manifest[key]))
    manifest["views"] = views
    out = method_dir / "pseudo_manifest_pre_difix.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out


def _make_contact_sheet(
    output_path: Path,
    rows: list[str],
    columns: list[tuple[str, Path]],
    thumb: tuple[int, int] = (180, 180),
    row_label_w: int = 110,
) -> None:
    label_h = 28
    pad = 8
    cell_w = thumb[0]
    cell_h = thumb[1] + label_h
    width = row_label_w + len(columns) * cell_w + (len(columns) + 1) * pad
    height = label_h + len(rows) * cell_h + (len(rows) + 1) * pad
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    x = row_label_w + pad
    for title, _ in columns:
        draw.text((x + 4, 7), title, fill=(0, 0, 0))
        x += cell_w + pad
    y = label_h + pad
    for row in rows:
        draw.text((8, y + thumb[1] // 2), row, fill=(0, 0, 0))
        x = row_label_w + pad
        for _, folder in columns:
            path = folder / f"{row}.png"
            if path.exists():
                img = Image.open(path).convert("RGB").resize(thumb, Image.Resampling.BILINEAR)
            else:
                img = Image.new("RGB", thumb, (230, 230, 230))
                ImageDraw.Draw(img).text((8, thumb[1] // 2), "missing", fill=(0, 0, 0))
            sheet.paste(img, (x, y + label_h))
            x += cell_w + pad
        y += cell_h + pad
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo_manifest", type=Path, required=True)
    parser.add_argument("--source_images_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--alpha_threshold", type=float, default=0.2)
    parser.add_argument("--depth_tau", type=float, default=0.15)
    parser.add_argument("--rgb_tau", type=float, default=0.12)
    parser.add_argument("--confidence_threshold", type=float, default=0.03)
    parser.add_argument("--confidence_blur", type=float, default=1.0)
    parser.add_argument("--mix_strength", type=float, default=0.85)
    parser.add_argument("--hard_replace_threshold", type=float, default=0.35)
    parser.add_argument("--side_conf_threshold", type=float, default=0.25)
    parser.add_argument("--agreement_tau", type=float, default=0.08)
    parser.add_argument("--agreement_threshold", type=float, default=0.35)
    parser.add_argument("--residual_clip", type=float, default=0.18)
    parser.add_argument("--residual_strength", type=float, default=0.85)
    parser.add_argument("--artifact_raw_conf_threshold", type=float, default=0.18)
    parser.add_argument("--artifact_ref_conf_threshold", type=float, default=0.18)
    parser.add_argument("--artifact_delta_threshold", type=float, default=0.08)
    parser.add_argument("--artifact_dilate", type=int, default=5)
    parser.add_argument("--artifact_close", type=int, default=5)
    parser.add_argument("--artifact_inpaint_radius", type=float, default=5.0)
    parser.add_argument("--max_sheet_views", type=int, default=8)
    args = parser.parse_args()

    manifest_path = args.pseudo_manifest.resolve()
    manifest_dir = manifest_path.parent
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    pose_src = _resolve(manifest_dir, manifest["pose_path"])
    source_pose_src = _resolve(manifest_dir, manifest["source_pose_path"])
    pseudo_poses = np.load(pose_src)
    source_poses = np.load(source_pose_src)
    model_path = _resolve(manifest_dir, manifest.get("source_model_path", ".."))
    cameras = _load_cameras(model_path)
    default_cam = next(iter(cameras.values())) if cameras else None
    ref_depth_dir_value = manifest.get("ref_depth_dir")
    ref_depth_dir = _resolve(manifest_dir, ref_depth_dir_value) if ref_depth_dir_value else None

    method_views: dict[str, list[dict]] = {}
    method_stats: dict[str, list[dict]] = {}
    method_dirs: dict[str, Path] = {}
    aux_dir = output_dir / "_aux"
    aux_dir.mkdir(parents=True, exist_ok=True)

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
        warped_left, left_conf, left_warp_stats = _warp_ref_to_pseudo(
            left_ref, pseudo_depth, left_depth, alpha, pseudo_w2c, left_w2c,
            fx, fy, cx, cy, args.depth_tau, args.alpha_threshold,
        )
        warped_right, right_conf, right_warp_stats = _warp_ref_to_pseudo(
            right_ref, pseudo_depth, right_depth, alpha, pseudo_w2c, right_w2c,
            fx, fy, cx, cy, args.depth_tau, args.alpha_threshold,
        )
        raw_left_conf, _ = _side_confidence(
            raw, left_ref, pseudo_depth, left_depth, alpha, pseudo_w2c, left_w2c,
            fx, fy, cx, cy, args.rgb_tau, args.depth_tau, args.alpha_threshold,
        )
        raw_right_conf, _ = _side_confidence(
            raw, right_ref, pseudo_depth, right_depth, alpha, pseudo_w2c, right_w2c,
            fx, fy, cx, cy, args.rgb_tau, args.depth_tau, args.alpha_threshold,
        )

        variants, aux = _build_variants(
            raw,
            warped_left,
            warped_right,
            left_conf,
            right_conf,
            raw_left_conf,
            raw_right_conf,
            float(view.get("interval_t", 0.5)),
            args,
        )

        if not (aux_dir / "warped_left").exists():
            for folder in ("warped_left", "warped_right", "left_conf", "right_conf", "artifact_mask", "raw_ref_score"):
                (aux_dir / folder).mkdir(parents=True, exist_ok=True)
        _save_rgb(aux_dir / "warped_left" / f"{image_name}.png", warped_left)
        _save_rgb(aux_dir / "warped_right" / f"{image_name}.png", warped_right)
        _save_heatmap(aux_dir / "left_conf" / f"{image_name}.png", aux["left_conf"])
        _save_heatmap(aux_dir / "right_conf" / f"{image_name}.png", aux["right_conf"])
        _save_heatmap(aux_dir / "artifact_mask" / f"{image_name}.png", aux["artifact_mask"])
        _save_heatmap(aux_dir / "raw_ref_score" / f"{image_name}.png", aux["raw_ref_score"])

        for method, candidate in variants.items():
            method_dir = output_dir / method
            image_dir = method_dir / "images"
            mask_dir = method_dir / "pre_difix_masks"
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            method_dirs[method] = image_dir
            _save_rgb(image_dir / f"{image_name}.png", candidate)
            if method == "best_side_blend":
                _save_gray(mask_dir / f"{image_name}.png", aux["best_conf"])
            elif method == "hard_best_side":
                _save_gray(mask_dir / f"{image_name}.png", aux["hard_mask"])
            elif method == "agreement_select":
                _save_gray(mask_dir / f"{image_name}.png", aux["agreement_mask"])
            elif method.startswith("artifact_"):
                _save_gray(mask_dir / f"{image_name}.png", aux["artifact_mask"])
            else:
                _save_gray(mask_dir / f"{image_name}.png", aux["soft_conf"])

            metrics = _metric_for_candidate(
                candidate, raw, left_ref, right_ref, pseudo_depth, left_depth, right_depth, alpha,
                pseudo_w2c, left_w2c, right_w2c, fx, fy, cx, cy, float(view.get("interval_t", 0.5)), args
            )
            metrics.update(
                {
                    "image_name": image_name,
                    "interval_t": float(view.get("interval_t", 0.5)),
                    "left_warp_conf": left_warp_stats["conf_mean"],
                    "right_warp_conf": right_warp_stats["conf_mean"],
                }
            )
            method_stats.setdefault(method, []).append(metrics)
            view_copy = copy.deepcopy(view)
            view_copy["raw_image_path"] = str(Path("images") / f"{image_name}.png")
            method_views.setdefault(method, []).append(view_copy)

    summary = {"methods": {}}
    for method, rows in sorted(method_stats.items()):
        method_dir = output_dir / method
        method_summary = {
            "num_views": len(rows),
            "weighted_rgb_error_mean": float(np.mean([r["weighted_rgb_error"] for r in rows])),
            "weighted_conf_mean": float(np.mean([r["weighted_conf"] for r in rows])),
            "change_from_raw_mean": float(np.mean([r["change_from_raw"] for r in rows])),
            "sharpness_mean": float(np.mean([r["sharpness"] for r in rows])),
            "rank_score": float(np.mean([r["weighted_conf"] - r["weighted_rgb_error"] for r in rows])),
            "views": rows,
        }
        summary["methods"][method] = method_summary
        with (method_dir / "pre_difix_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(method_summary, f, indent=2)
        _write_manifest(manifest, manifest_dir, method_dir, method, method_views[method])

    ranked = sorted(
        summary["methods"].items(),
        key=lambda kv: (kv[1]["rank_score"], -kv[1]["weighted_rgb_error_mean"]),
        reverse=True,
    )
    summary["ranking"] = [
        {
            "method": method,
            "rank_score": stats["rank_score"],
            "weighted_conf_mean": stats["weighted_conf_mean"],
            "weighted_rgb_error_mean": stats["weighted_rgb_error_mean"],
            "change_from_raw_mean": stats["change_from_raw_mean"],
            "sharpness_mean": stats["sharpness_mean"],
        }
        for method, stats in ranked
    ]
    with (output_dir / "pre_difix_variant_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Pick a balanced subset: fixed interesting examples plus highest/lowest confidence rows.
    all_names = [v.get("image_name", f"pseudo_{int(v['pose_index']):05d}") for v in manifest["views"]]
    fixed = [name for name in ("pseudo_00002", "pseudo_00016", "pseudo_00012", "pseudo_00024", "pseudo_00013", "pseudo_00000") if name in all_names]
    rows = fixed[: args.max_sheet_views]
    columns = [("raw", output_dir / "raw" / "images")]
    for method in ("soft_depth_blend", "hard_best_side", "agreement_select", "artifact_inpaint", "artifact_agreement", "residual_clipped"):
        columns.append((method, output_dir / method / "images"))
    columns.extend([
        ("artifact_mask", aux_dir / "artifact_mask"),
        ("warp_left", aux_dir / "warped_left"),
        ("warp_right", aux_dir / "warped_right"),
    ])
    _make_contact_sheet(output_dir / "pre_difix_variant_sheet.png", rows, columns)

    print("[pre-difix] ranking:")
    for item in summary["ranking"]:
        print(
            f"  {item['method']:>18s}  score={item['rank_score']:.4f} "
            f"conf={item['weighted_conf_mean']:.4f} rgb_err={item['weighted_rgb_error_mean']:.4f} "
            f"delta={item['change_from_raw_mean']:.4f}"
        )
    print(f"[pre-difix] wrote: {output_dir / 'pre_difix_variant_summary.json'}")
    print(f"[pre-difix] visual sheet: {output_dir / 'pre_difix_variant_sheet.png'}")


if __name__ == "__main__":
    main()
