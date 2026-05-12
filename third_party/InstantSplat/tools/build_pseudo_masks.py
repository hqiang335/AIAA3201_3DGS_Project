#!/usr/bin/env python3
"""Build pseudo training images and confidence masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _save_rgb(path: Path, image: np.ndarray) -> None:
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image_u8, mode="RGB").save(path)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(mask_u8, mode="L").save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--alpha_threshold", type=float, default=0.2)
    parser.add_argument("--agreement_tau", type=float, default=0.08)
    parser.add_argument("--enhanced_left_dir", type=Path, default=None)
    parser.add_argument("--enhanced_right_dir", type=Path, default=None)
    parser.add_argument("--hard_mask", action="store_true")
    args = parser.parse_args()

    manifest_path = args.pseudo_manifest.resolve()
    manifest_dir = manifest_path.parent
    output_dir = args.output_dir or manifest_dir
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    use_enhanced_pair = args.enhanced_left_dir is not None and args.enhanced_right_dir is not None
    mask_means = []
    for view in manifest["views"]:
        image_name = view.get("image_name", f"pseudo_{int(view['pose_index']):05d}")
        alpha_path = _resolve(manifest_dir, view["alpha_path"])
        alpha = _load_gray(alpha_path)

        if use_enhanced_pair:
            left = _load_rgb(args.enhanced_left_dir / f"{image_name}.png")
            right = _load_rgb(args.enhanced_right_dir / f"{image_name}.png")
            image = 0.5 * (left + right)
            diff = np.mean(np.abs(left - right), axis=2)
            agreement = np.exp(-diff / max(args.agreement_tau, 1e-6))
            mask = alpha * agreement
            view["mask_source"] = "alpha_x_left_right_agreement"
        else:
            raw_path = _resolve(manifest_dir, view["raw_image_path"])
            image = _load_rgb(raw_path)
            mask = alpha
            view["mask_source"] = "alpha_only"

        mask = np.where(alpha >= args.alpha_threshold, mask, 0.0)
        if args.hard_mask:
            mask = np.where(mask >= 0.5, 1.0, 0.0)

        image_rel = Path("images") / f"{image_name}.png"
        mask_rel = Path("masks") / f"{image_name}.png"
        _save_rgb(output_dir / image_rel, image)
        _save_mask(output_dir / mask_rel, mask)
        view["image_path"] = str(image_rel)
        view["mask_path"] = str(mask_rel)
        view["mask_mean"] = float(mask.mean())
        mask_means.append(float(mask.mean()))

    train_manifest = output_dir / "pseudo_manifest_train.json"
    with train_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[pseudo] built training pseudo views: {len(manifest['views'])}")
    print(f"[pseudo] mask mean: min={min(mask_means):.4f}, mean={np.mean(mask_means):.4f}, max={max(mask_means):.4f}")
    print(f"[pseudo] wrote: {train_manifest}")


if __name__ == "__main__":
    main()
