#!/usr/bin/env python3
"""Build ReconX-style DUSt3R/MASt3R pseudo-view confidence maps.

The output manifest keeps the pseudo RGB images from an existing pseudo
manifest, but replaces/adds ``mask_path`` with soft confidence maps estimated
from MASt3R pairwise matching within each temporal window:

    real_left + pseudo_1 ... pseudo_k + real_right

For every pseudo frame, we collect the pseudo-side confidence maps from pairs
against the left/right real references, temporal neighbors, and optionally all
other frames in the same window. The aggregated soft map can then be used by
``train.py`` as a confidence-aware pseudo-view loss weight.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mast3r.model import AsymmetricMASt3R
from dust3r.inference import inference
from dust3r.utils.device import to_numpy
from utils.sfm_utils import load_images


def resolve_path(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_manifest(path: Path) -> tuple[dict, Path]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f), path.parent


def save_gray(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(value * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)


def save_heatmap(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value_u8 = np.clip(value * 255.0, 0, 255).astype(np.uint8)
    if cv2 is None:
        Image.fromarray(value_u8, mode="L").save(path)
        return
    heat = cv2.applyColorMap(value_u8, cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    Image.fromarray(heat).save(path)


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L")
    image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def robust_normalize(conf: np.ndarray, low: float, high: float) -> np.ndarray:
    conf = np.asarray(conf, dtype=np.float32)
    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)
    valid = conf[np.isfinite(conf)]
    if valid.size == 0:
        return np.zeros_like(conf, dtype=np.float32)
    lo = float(np.percentile(valid, low))
    hi = float(np.percentile(valid, high))
    if hi <= lo + 1e-6:
        hi = float(valid.max())
        lo = float(valid.min())
    if hi <= lo + 1e-6:
        return np.zeros_like(conf, dtype=np.float32)
    return np.clip((conf - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def view_image_path(manifest_dir: Path, view: dict) -> Path:
    path = resolve_path(manifest_dir, view.get("image_path"))
    if path is None:
        raise ValueError(f"Pseudo view is missing image_path: {view}")
    return path


def real_image_path(source_images_dir: Path, name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else source_images_dir / path


def group_windows(views: list[dict]) -> list[tuple[tuple[str, str], list[dict]]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for view in views:
        left = view.get("left_image")
        right = view.get("right_image")
        if not left or not right:
            raise ValueError(f"Pseudo view is missing left_image/right_image: {view}")
        grouped[(left, right)].append(view)
    windows = []
    for key, items in grouped.items():
        items.sort(key=lambda view: float(view.get("interval_t", view.get("pseudo_id", 0))))
        windows.append((key, items))
    windows.sort(key=lambda item: min(float(v.get("pseudo_id", 0)) for v in item[1]))
    return windows


def build_window_pairs(
    n_frames: int,
    pseudo_positions: list[int],
    left_right_weight: float,
    neighbor_weight: float,
    all_pair_weight: float,
    bidirectional: bool,
) -> tuple[list[tuple[int, int]], list[tuple[int, float]]]:
    pair_indices: list[tuple[int, int]] = []
    pair_targets: list[tuple[int, float]] = []

    def add_pair(a: int, b: int, pseudo_pos: int, weight: float) -> None:
        pair_indices.append((a, b))
        pair_targets.append((pseudo_pos, weight))

    for pos in pseudo_positions:
        weights: dict[int, float] = {}
        if all_pair_weight > 0:
            for ref in range(n_frames):
                if ref != pos:
                    weights[ref] = max(weights.get(ref, 0.0), all_pair_weight)
        if neighbor_weight > 0:
            for ref in (pos - 1, pos + 1):
                if 0 <= ref < n_frames and ref != pos:
                    weights[ref] = max(weights.get(ref, 0.0), neighbor_weight)
        if left_right_weight > 0:
            weights[0] = max(weights.get(0, 0.0), left_right_weight)
            weights[n_frames - 1] = max(weights.get(n_frames - 1, 0.0), left_right_weight)

        for ref, weight in sorted(weights.items()):
            add_pair(pos, ref, pos, weight)
            if bidirectional:
                add_pair(ref, pos, pos, weight)
    return pair_indices, pair_targets


def extract_pseudo_confidences(
    output: dict,
    pair_indices: list[tuple[int, int]],
    pair_targets: list[tuple[int, float]],
    pseudo_positions: list[int],
    normalize_low: float,
    normalize_high: float,
) -> dict[int, list[tuple[np.ndarray, float]]]:
    pred1_conf = to_numpy(output["pred1"]["conf"])
    pred2_conf = to_numpy(output["pred2"]["conf"])
    collected: dict[int, list[tuple[np.ndarray, float]]] = {pos: [] for pos in pseudo_positions}
    for edge_idx, ((a, b), (pseudo_pos, weight)) in enumerate(zip(pair_indices, pair_targets)):
        if a == pseudo_pos:
            conf = pred1_conf[edge_idx]
        elif b == pseudo_pos:
            conf = pred2_conf[edge_idx]
        else:
            continue
        collected[pseudo_pos].append((robust_normalize(conf, normalize_low, normalize_high), weight))
    return collected


def aggregate_confidences(items: list[tuple[np.ndarray, float]], mode: str) -> np.ndarray:
    if not items:
        raise ValueError("Cannot aggregate an empty confidence list")
    confs = np.stack([conf for conf, _ in items], axis=0)
    weights = np.asarray([weight for _, weight in items], dtype=np.float32)
    weights = weights / max(float(weights.sum()), 1e-6)
    if mode == "weighted_mean":
        return np.sum(confs * weights[:, None, None], axis=0)
    if mode == "max":
        return np.max(confs, axis=0)
    if mode == "noisy_or":
        return 1.0 - np.prod(1.0 - np.clip(confs * weights[:, None, None], 0.0, 1.0), axis=0)
    raise ValueError(f"Unknown aggregate mode: {mode}")


def postprocess_mask(mask: np.ndarray, blur: float, gamma: float, threshold: float, hard: bool) -> np.ndarray:
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    if gamma != 1.0:
        mask = np.power(mask, max(gamma, 1e-6))
    if threshold > 0:
        mask = np.where(mask >= threshold, mask, 0.0).astype(np.float32)
    if blur > 0:
        image = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L")
        image = image.filter(ImageFilter.GaussianBlur(radius=blur))
        mask = np.asarray(image, dtype=np.float32) / 255.0
    if hard:
        mask = np.where(mask >= 0.5, 1.0, 0.0).astype(np.float32)
    return np.clip(mask, 0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo_manifest", type=Path, required=True)
    parser.add_argument("--source_images_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=str, default="./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_windows", type=int, default=0, help="Debug limit. 0 processes all windows.")
    parser.add_argument("--bidirectional_pairs", action="store_true")
    parser.add_argument("--left_right_weight", type=float, default=1.0)
    parser.add_argument("--neighbor_weight", type=float, default=0.75)
    parser.add_argument("--all_pair_weight", type=float, default=0.5)
    parser.add_argument("--aggregate", choices=["weighted_mean", "max", "noisy_or"], default="weighted_mean")
    parser.add_argument("--normalize_low", type=float, default=5.0)
    parser.add_argument("--normalize_high", type=float, default=95.0)
    parser.add_argument("--mask_blur", type=float, default=1.0)
    parser.add_argument("--mask_gamma", type=float, default=1.0)
    parser.add_argument("--mask_threshold", type=float, default=0.0)
    parser.add_argument("--hard_mask", action="store_true")
    parser.add_argument("--set_loss_weight_from_mask", action="store_true")
    parser.add_argument("--loss_weight_scale", type=float, default=2.0)
    parser.add_argument("--min_loss_weight", type=float, default=0.2)
    parser.add_argument("--max_loss_weight", type=float, default=1.0)
    args = parser.parse_args()

    manifest, manifest_dir = load_manifest(args.pseudo_manifest)
    views = list(manifest.get("views", []))
    if not views:
        raise ValueError(f"{args.pseudo_manifest} does not contain views")
    source_images_dir = args.source_images_dir
    if source_images_dir is None:
        source_images_value = manifest.get("source_images_dir")
        if source_images_value:
            source_images_dir = Path(source_images_value)
    if source_images_dir is None:
        raise ValueError("--source_images_dir is required unless the pseudo manifest contains source_images_dir")

    output_dir = args.output_dir
    masks_dir = output_dir / "masks"
    confidence_dir = output_dir / "confidence_vis"
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    confidence_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ReconX-mask] Loading MASt3R: {args.ckpt_path}")
    model = AsymmetricMASt3R.from_pretrained(args.ckpt_path).to(args.device).eval()

    windows = group_windows(views)
    if args.max_windows > 0:
        windows = windows[: args.max_windows]

    updated_views = []
    stats = []
    processed_ids = set()

    for window_idx, ((left_name, right_name), window_views) in enumerate(windows):
        frame_paths = [real_image_path(source_images_dir, left_name)]
        frame_paths.extend(view_image_path(manifest_dir, view) for view in window_views)
        frame_paths.append(real_image_path(source_images_dir, right_name))
        for path in frame_paths:
            if not path.exists():
                raise FileNotFoundError(path)

        print(
            f"[ReconX-mask] Window {window_idx + 1}/{len(windows)}: "
            f"{left_name} -> {right_name}, pseudo={len(window_views)}"
        )
        images, _ = load_images([str(path) for path in frame_paths], size=args.image_size, verbose=False)
        pseudo_positions = list(range(1, len(frame_paths) - 1))
        pair_indices, pair_targets = build_window_pairs(
            len(frame_paths),
            pseudo_positions,
            args.left_right_weight,
            args.neighbor_weight,
            args.all_pair_weight,
            args.bidirectional_pairs,
        )
        pairs = [(images[i], images[j]) for i, j in pair_indices]
        output = inference(pairs, model, args.device, batch_size=args.batch_size, verbose=True)
        collected = extract_pseudo_confidences(
            output,
            pair_indices,
            pair_targets,
            pseudo_positions,
            args.normalize_low,
            args.normalize_high,
        )

        for local_idx, view in enumerate(window_views):
            pos = local_idx + 1
            image_path = view_image_path(manifest_dir, view)
            image_size = Image.open(image_path).size
            mask = aggregate_confidences(collected[pos], args.aggregate)
            mask = resize_mask(mask, image_size)
            mask = postprocess_mask(mask, args.mask_blur, args.mask_gamma, args.mask_threshold, args.hard_mask)

            image_name = view.get("image_name", f"pseudo_{int(view.get('pseudo_id', len(updated_views))):05d}")
            mask_rel = Path("masks") / f"{image_name}.png"
            confidence_rel = Path("confidence_vis") / f"{image_name}.png"
            save_gray(output_dir / mask_rel, mask)
            save_heatmap(output_dir / confidence_rel, mask)

            new_view = dict(view)
            new_view["image_path"] = str(image_path)
            new_view["mask_path"] = str(mask_rel)
            new_view["confidence_vis_path"] = str(confidence_rel)
            new_view["mask_source"] = "reconx_style_mast3r_pairwise_confidence"
            new_view["mask_mean"] = float(mask.mean())
            new_view["mask_pair_count"] = int(len(collected[pos]))
            if args.set_loss_weight_from_mask:
                new_view["loss_weight"] = float(
                    np.clip(mask.mean() * args.loss_weight_scale, args.min_loss_weight, args.max_loss_weight)
                )
            updated_views.append(new_view)
            processed_ids.add(id(view))
            stats.append({
                "image_name": image_name,
                "left_image": left_name,
                "right_image": right_name,
                "mask_mean": float(mask.mean()),
                "mask_min": float(mask.min()),
                "mask_max": float(mask.max()),
                "mask_pair_count": int(len(collected[pos])),
                "loss_weight": float(new_view.get("loss_weight", 1.0)),
            })

        del output
        torch.cuda.empty_cache()

    # Preserve any views not processed when --max_windows is used.
    for view in views:
        if id(view) not in processed_ids:
            copied = dict(view)
            path = view_image_path(manifest_dir, view)
            copied["image_path"] = str(path)
            updated_views.append(copied)

    pose_path = resolve_path(manifest_dir, manifest.get("pose_path"))
    if pose_path is not None:
        manifest["pose_path"] = str(pose_path)
    manifest["views"] = updated_views
    manifest["reconx_dust3r_confidence"] = {
        "source": str(args.pseudo_manifest),
        "source_images_dir": str(source_images_dir),
        "ckpt_path": args.ckpt_path,
        "image_size": args.image_size,
        "bidirectional_pairs": args.bidirectional_pairs,
        "left_right_weight": args.left_right_weight,
        "neighbor_weight": args.neighbor_weight,
        "all_pair_weight": args.all_pair_weight,
        "aggregate": args.aggregate,
        "normalize_low": args.normalize_low,
        "normalize_high": args.normalize_high,
        "mask_blur": args.mask_blur,
        "mask_gamma": args.mask_gamma,
        "mask_threshold": args.mask_threshold,
        "hard_mask": args.hard_mask,
        "set_loss_weight_from_mask": args.set_loss_weight_from_mask,
    }

    out_manifest = output_dir / "pseudo_manifest_reconx_dust3r_confidence.json"
    with out_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with (output_dir / "confidence_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    means = [item["mask_mean"] for item in stats]
    print(f"[ReconX-mask] wrote {out_manifest}")
    if means:
        print(
            "[ReconX-mask] mask mean: "
            f"min={min(means):.4f}, mean={float(np.mean(means)):.4f}, max={max(means):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
