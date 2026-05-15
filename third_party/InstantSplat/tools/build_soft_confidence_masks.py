#!/usr/bin/env python3
"""Build softened pseudo-view confidence masks and a matching manifest.

The input manifest is typically ``fusion_stage0_v1/pseudo_manifest_fused_continuous.json``.
The script copies the manifest, writes softened masks under the output directory,
and rewrites each pseudo view's ``mask_path`` to point at the new masks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base / path


def _odd_kernel(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 1
    return value if value % 2 == 1 else value + 1


def soften_mask(mask: Image.Image, blur_radius: float, dilate_kernel: int, close_kernel: int) -> Image.Image:
    mask = mask.convert("L")
    if dilate_kernel > 1:
        mask = mask.filter(ImageFilter.MaxFilter(_odd_kernel(dilate_kernel)))
    if close_kernel > 1:
        size = _odd_kernel(close_kernel)
        mask = mask.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size))
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    arr = np.asarray(mask).astype(np.float32) / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="L")


def colorize_confidence(mask: Image.Image) -> Image.Image:
    arr = np.asarray(mask.convert("L")).astype(np.float32) / 255.0
    color = np.zeros((*arr.shape, 3), dtype=np.uint8)
    color[..., 0] = np.clip(255 * arr, 0, 255).astype(np.uint8)
    color[..., 1] = np.clip(180 * np.sqrt(arr), 0, 255).astype(np.uint8)
    color[..., 2] = np.clip(255 * (1.0 - arr), 0, 255).astype(np.uint8)
    return Image.fromarray(color, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--blur_radius", type=float, default=2.0)
    parser.add_argument("--dilate_kernel", type=int, default=3)
    parser.add_argument("--close_kernel", type=int, default=3)
    parser.add_argument("--output_manifest_name", default="pseudo_manifest_softconf_v1.json")
    args = parser.parse_args()

    input_manifest = args.input_manifest.resolve()
    manifest_dir = input_manifest.parent
    with input_manifest.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    output_dir = args.output_dir.resolve()
    mask_dir = output_dir / "masks"
    vis_dir = output_dir / "confidence_vis"
    mask_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    means = []
    for view in manifest.get("views", []):
        image_value = view.get("image_path")
        if image_value:
            view["image_path"] = str(_resolve(manifest_dir, image_value))
        mask_value = view.get("mask_path") or view.get("feature_mask_path")
        if not mask_value:
            raise ValueError(f"View {view.get('image_name')} has no mask_path or feature_mask_path")
        src = _resolve(manifest_dir, mask_value)
        if not src.exists():
            raise FileNotFoundError(src)

        name = Path(view.get("image_path", src.name)).name
        if Path(name).suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            name = f"{Path(name).stem}.png"
        dst = mask_dir / name
        vis = vis_dir / name

        softened = soften_mask(
            Image.open(src),
            blur_radius=args.blur_radius,
            dilate_kernel=args.dilate_kernel,
            close_kernel=args.close_kernel,
        )
        softened.save(dst)
        colorize_confidence(softened).save(vis)
        means.append(float(np.asarray(softened, dtype=np.float32).mean() / 255.0))

        view["previous_mask_path"] = mask_value
        view["mask_path"] = str(dst.relative_to(output_dir))
        view.pop("feature_mask_path", None)
        view.pop("feature_left_path", None)
        view.pop("feature_right_path", None)
        view["confidence_vis_path"] = str(vis.relative_to(output_dir))
        view["mask_source"] = "softened_stage0_confidence"
        view["soft_conf_mean"] = means[-1]

    manifest["soft_confidence"] = {
        "source_manifest": str(input_manifest),
        "blur_radius": args.blur_radius,
        "dilate_kernel": args.dilate_kernel,
        "close_kernel": args.close_kernel,
        "mask_mean_min": min(means) if means else None,
        "mask_mean_mean": sum(means) / len(means) if means else None,
        "mask_mean_max": max(means) if means else None,
    }

    output_manifest = output_dir / args.output_manifest_name
    with output_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote softened masks: {mask_dir}")
    print(f"Wrote confidence previews: {vis_dir}")
    print(f"Wrote manifest: {output_manifest}")
    if means:
        print(
            "Mask mean min/mean/max: "
            f"{min(means):.4f} / {sum(means) / len(means):.4f} / {max(means):.4f}"
        )


if __name__ == "__main__":
    main()
