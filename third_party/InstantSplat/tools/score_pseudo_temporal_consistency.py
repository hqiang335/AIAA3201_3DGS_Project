#!/usr/bin/env python3
"""Score pseudo views with no-GT temporal optical-flow consistency.

The score is view-level on purpose.  We use it to decide which pseudo views are
safe enough to include as training images or strong pseudo supervision.  It does
not need test/GT images: each pseudo frame is checked against its neighboring
frames inside the same real-view pair window.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base / path)


def _load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    rgb = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


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


def _candidate_image_path(view: dict[str, Any], manifest_dir: Path, image_key: str) -> Path:
    value = view.get(image_key) or view.get("image_path")
    if not value:
        raise KeyError(f"View is missing image path key {image_key}: {view}")
    path = Path(str(value))
    return path if path.is_absolute() else manifest_dir / path


def _flow_calc(method: str):
    if method == "dis" and hasattr(cv2, "DISOpticalFlow_create"):
        flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        return lambda a, b: flow.calc(a, b, None)

    def farneback(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return cv2.calcOpticalFlowFarneback(
            a,
            b,
            None,
            pyr_scale=0.5,
            levels=4,
            winsize=25,
            iterations=4,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )

    return farneback


def _remap(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        image.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _robust_mean(value: np.ndarray, valid: np.ndarray, clip: float) -> float:
    if not np.any(valid):
        return float(clip)
    selected = np.clip(value[valid], 0.0, clip)
    return float(np.mean(selected))


def _direction_score(
    current: np.ndarray,
    reference: np.ndarray,
    flow_fn,
    photo_tau: float,
    fb_tau: float,
    max_photo: float,
    max_fb: float,
) -> tuple[float, dict[str, float]]:
    h, w = current.shape[:2]
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")

    flow_cur_ref = flow_fn(_to_gray_u8(current), _to_gray_u8(reference)).astype(np.float32)
    map_x = xx + flow_cur_ref[..., 0]
    map_y = yy + flow_cur_ref[..., 1]
    valid = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)

    warped_reference = _remap(reference, map_x, map_y)
    photo_error = np.mean(np.abs(current - warped_reference), axis=2)

    flow_ref_cur = flow_fn(_to_gray_u8(reference), _to_gray_u8(current)).astype(np.float32)
    warped_back = _remap(flow_ref_cur, map_x, map_y)
    fb_error_px = np.linalg.norm(flow_cur_ref + warped_back, axis=2)
    fb_error = fb_error_px / max(math.sqrt(float(h * h + w * w)), 1.0)

    valid_ratio = float(np.mean(valid))
    photo = _robust_mean(photo_error, valid, max_photo)
    fb = _robust_mean(fb_error, valid, max_fb)
    score = math.exp(-photo / max(photo_tau, 1e-6)) * math.exp(-fb / max(fb_tau, 1e-6))
    score *= math.sqrt(max(valid_ratio, 0.0))
    stats = {
        "photo_error": photo,
        "fb_error": fb,
        "valid_ratio": valid_ratio,
        "score": float(score),
    }
    return float(score), stats


def _color_prior_score(current: np.ndarray, left: np.ndarray, right: np.ndarray, t: float, tau: float) -> tuple[float, float]:
    expected_mean = (1.0 - t) * left.mean(axis=(0, 1)) + t * right.mean(axis=(0, 1))
    current_mean = current.mean(axis=(0, 1))
    err = float(np.mean(np.abs(current_mean - expected_mean)))
    return float(math.exp(-err / max(tau, 1e-6))), err


def _copy_pose_files(manifest: dict[str, Any], manifest_dir: Path, output_manifest: Path) -> None:
    """Keep relative pose paths valid if a filtered/scored manifest is moved."""
    output_dir = output_manifest.parent
    for key in ("pose_path", "source_pose_path"):
        src = _resolve(manifest_dir, manifest.get(key))
        if src is None or not src.exists():
            continue
        dst = output_dir / src.name
        if src.resolve() != dst.resolve():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        manifest[key] = dst.name


def _absolutize_view_paths(view: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    result = dict(view)
    for key in (
        "image_path",
        "wide_image_path",
        "aspect_image_path",
        "mask_path",
        "feature_mask_path",
        "feature_left_path",
        "feature_right_path",
        "confidence_vis_path",
        "raw_image_path",
        "source_window_image_path",
    ):
        value = result.get(key)
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = (manifest_dir / path).resolve()
        result[key] = str(path)
    return result


def _proxy_scores(path: Path) -> dict[str, dict[str, float]]:
    data = _load_json(path)
    rows = data.get("rows", data if isinstance(data, list) else [])
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        image_name = row.get("image_name")
        if not image_name:
            continue
        out[str(image_name)] = {
            "proxy_full_psnr": float(row["full"]["psnr"]),
            "proxy_full_ssim": float(row["full"]["ssim"]),
            "proxy_masked_psnr": float(row["masked"]["psnr"]),
            "proxy_masked_ssim": float(row["masked"].get("ssim", row["full"]["ssim"])),
        }
    return out


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _save_contact_sheet(rows: list[dict[str, Any]], output: Path, source_images_dir: Path, manifest_dir: Path, image_key: str, max_rows: int) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: row["temporal_score"])[: max_rows // 2] + sorted(rows, key=lambda row: row["temporal_score"], reverse=True)[: max_rows // 2]
    tile_w, tile_h = 256, 256
    label_h = 34
    sheet = Image.new("RGB", (tile_w * 3, (tile_h + label_h) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(rows):
        y0 = r * (tile_h + label_h)
        view = row["view"]
        left = Image.open(_find_source_image(source_images_dir, view["left_image"])).convert("RGB").resize((tile_w, tile_h), Image.Resampling.BILINEAR)
        cur = Image.open(_candidate_image_path(view, manifest_dir, image_key)).convert("RGB").resize((tile_w, tile_h), Image.Resampling.BILINEAR)
        right = Image.open(_find_source_image(source_images_dir, view["right_image"])).convert("RGB").resize((tile_w, tile_h), Image.Resampling.BILINEAR)
        for c, img in enumerate((left, cur, right)):
            sheet.paste(img, (c * tile_w, y0 + label_h))
        label = (
            f"id={row['pseudo_id']} score={row['temporal_score']:.3f} "
            f"prev={row['prev_score']:.3f} next={row['next_score']:.3f} "
            f"photo={row['photo_error_mean']:.3f}"
        )
        draw.text((6, y0 + 8), label, fill=(0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo_manifest", type=Path, required=True)
    parser.add_argument("--source_images_dir", type=Path, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--image_key", default="image_path")
    parser.add_argument("--flow_method", choices=("dis", "farneback"), default="dis")
    parser.add_argument("--resize", type=int, default=192, help="Short side used for scoring; 0 keeps native size.")
    parser.add_argument("--photo_tau", type=float, default=0.08)
    parser.add_argument("--fb_tau", type=float, default=0.02)
    parser.add_argument("--color_tau", type=float, default=0.08)
    parser.add_argument("--max_photo", type=float, default=0.35)
    parser.add_argument("--max_fb", type=float, default=0.08)
    parser.add_argument("--set_loss_weight", action="store_true", help="Replace manifest loss_weight with temporal_score.")
    parser.add_argument("--min_loss_weight", type=float, default=0.25)
    parser.add_argument("--score_power", type=float, default=0.5, help="Use score**power when setting loss_weight.")
    parser.add_argument("--proxy_score_json", type=Path, default=None, help="Optional diagnostics only; not used for scoring.")
    parser.add_argument("--contact_sheet", type=Path, default=None)
    parser.add_argument("--contact_rows", type=int, default=16)
    args = parser.parse_args()

    manifest_path = args.pseudo_manifest.resolve()
    manifest_dir = manifest_path.parent
    manifest = _load_json(manifest_path)
    views = list(manifest.get("views", []))
    if not views:
        raise ValueError(f"No views in {manifest_path}")

    source_images_dir = args.source_images_dir.resolve()
    output_manifest = args.output_manifest.resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    first_path = _candidate_image_path(views[0], manifest_dir, args.image_key)
    first_img = Image.open(first_path).convert("RGB")
    if args.resize and args.resize > 0:
        w, h = first_img.size
        if min(w, h) != args.resize:
            scale = args.resize / float(min(w, h))
            size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        else:
            size = (w, h)
    else:
        size = first_img.size

    flow_fn = _flow_calc(args.flow_method)
    cache: dict[tuple[str, str], np.ndarray] = {}

    def load_frame(kind: str, name: str, view: dict[str, Any] | None = None) -> np.ndarray:
        key = (kind, name)
        if key in cache:
            return cache[key]
        if kind == "real":
            path = _find_source_image(source_images_dir, name)
        else:
            if view is None:
                raise ValueError("Pseudo view is required for pseudo frames")
            path = _candidate_image_path(view, manifest_dir, args.image_key)
        image = _load_rgb(path, size=size)
        cache[key] = image
        return image

    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for view in views:
        key = (
            int(view.get("viewcrafter_window_index", view.get("left_train_index", 0))),
            str(view["left_image"]),
            str(view["right_image"]),
        )
        groups.setdefault(key, []).append(view)

    scored_views: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for (_, left_image, right_image), group_views in sorted(groups.items(), key=lambda item: item[0]):
        group_views.sort(key=lambda view: float(view.get("interval_t", 0.5)))
        sequence: list[tuple[str, str, dict[str, Any] | None]] = [("real", left_image, None)]
        sequence += [("pseudo", str(view.get("image_name", f"pseudo_{int(view.get('pseudo_id', 0)):05d}")), view) for view in group_views]
        sequence += [("real", right_image, None)]

        left_real = load_frame("real", left_image)
        right_real = load_frame("real", right_image)
        for index in range(1, len(sequence) - 1):
            kind, name, view = sequence[index]
            if kind != "pseudo" or view is None:
                continue
            current = load_frame("pseudo", name, view)
            prev_kind, prev_name, prev_view = sequence[index - 1]
            next_kind, next_name, next_view = sequence[index + 1]
            previous = load_frame(prev_kind, prev_name, prev_view)
            next_frame = load_frame(next_kind, next_name, next_view)

            prev_score, prev_stats = _direction_score(
                current, previous, flow_fn, args.photo_tau, args.fb_tau, args.max_photo, args.max_fb
            )
            next_score, next_stats = _direction_score(
                current, next_frame, flow_fn, args.photo_tau, args.fb_tau, args.max_photo, args.max_fb
            )
            temporal_pair_score = math.sqrt(max(prev_score, 1e-8) * max(next_score, 1e-8))
            t = float(view.get("interval_t", 0.5))
            color_score, color_error = _color_prior_score(current, left_real, right_real, t, args.color_tau)
            temporal_score = float(temporal_pair_score * math.sqrt(max(color_score, 1e-8)))
            photo_error_mean = 0.5 * (prev_stats["photo_error"] + next_stats["photo_error"])
            fb_error_mean = 0.5 * (prev_stats["fb_error"] + next_stats["fb_error"])
            valid_ratio_mean = 0.5 * (prev_stats["valid_ratio"] + next_stats["valid_ratio"])

            scored = _absolutize_view_paths(view, manifest_dir)
            scored["temporal_score"] = temporal_score
            scored["temporal_prev_score"] = prev_score
            scored["temporal_next_score"] = next_score
            scored["temporal_photo_error_mean"] = photo_error_mean
            scored["temporal_fb_error_mean"] = fb_error_mean
            scored["temporal_valid_ratio_mean"] = valid_ratio_mean
            scored["temporal_color_score"] = color_score
            scored["temporal_color_error"] = color_error
            if args.set_loss_weight:
                scored["loss_weight"] = float(np.clip(temporal_score ** args.score_power, args.min_loss_weight, 1.0))
            scored_views.append(scored)
            rows.append(
                {
                    "pseudo_id": int(view.get("pseudo_id", len(rows))),
                    "image_name": view.get("image_name", name),
                    "left_image": left_image,
                    "right_image": right_image,
                    "interval_t": t,
                    "temporal_score": temporal_score,
                    "prev_score": prev_score,
                    "next_score": next_score,
                    "photo_error_mean": photo_error_mean,
                    "fb_error_mean": fb_error_mean,
                    "valid_ratio_mean": valid_ratio_mean,
                    "color_score": color_score,
                    "color_error": color_error,
                    "loss_weight": scored.get("loss_weight", view.get("loss_weight", 1.0)),
                    "view": scored,
                }
            )

    scored_views.sort(key=lambda view: int(view.get("pseudo_id", 0)))
    rows.sort(key=lambda row: row["pseudo_id"])
    manifest["views"] = scored_views
    manifest["temporal_consistency_scoring"] = {
        "method": "opencv_optical_flow_neighbor_warp",
        "flow_method": args.flow_method,
        "image_key": args.image_key,
        "resize": args.resize,
        "photo_tau": args.photo_tau,
        "fb_tau": args.fb_tau,
        "color_tau": args.color_tau,
        "max_photo": args.max_photo,
        "max_fb": args.max_fb,
        "set_loss_weight": bool(args.set_loss_weight),
        "min_loss_weight": args.min_loss_weight,
        "score_power": args.score_power,
    }
    _copy_pose_files(manifest, manifest_dir, output_manifest)
    with output_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    stats_path = output_manifest.with_name(output_manifest.stem + "_temporal_scores.json")
    serializable_rows = [{k: v for k, v in row.items() if k != "view"} for row in rows]
    summary = {
        "num_views": len(rows),
        "score_min": min(row["temporal_score"] for row in rows),
        "score_mean": float(np.mean([row["temporal_score"] for row in rows])),
        "score_max": max(row["temporal_score"] for row in rows),
        "rows": serializable_rows,
    }
    if args.proxy_score_json is not None:
        proxy = _proxy_scores(args.proxy_score_json)
        available = [row for row in rows if row["image_name"] in proxy]
        scores = [row["temporal_score"] for row in available]
        summary["proxy_correlation"] = {
            metric: _pearson(scores, [proxy[row["image_name"]][metric] for row in available])
            for metric in ("proxy_full_psnr", "proxy_full_ssim", "proxy_masked_psnr", "proxy_masked_ssim")
        }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.contact_sheet is not None:
        _save_contact_sheet(rows, args.contact_sheet, source_images_dir, manifest_dir, args.image_key, args.contact_rows)

    print(f"[temporal-score] wrote manifest: {output_manifest}")
    print(f"[temporal-score] wrote stats: {stats_path}")
    print(
        "[temporal-score] score "
        f"min={summary['score_min']:.4f} mean={summary['score_mean']:.4f} max={summary['score_max']:.4f}"
    )
    if "proxy_correlation" in summary:
        for metric, corr in summary["proxy_correlation"].items():
            print(f"[temporal-score] corr({metric})={corr:.4f}")


if __name__ == "__main__":
    main()
