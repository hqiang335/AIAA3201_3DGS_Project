#!/usr/bin/env python3
"""Create BRPO-style colored mask contact sheets for pseudo views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _source_images_dir(source_path: Path) -> Path:
    return source_path / "images" if (source_path / "images").is_dir() else source_path


def _find_image(images_dir: Path, name: str) -> Path:
    direct = images_dir / name
    if direct.exists():
        return direct
    stem = Path(name).stem
    for suffix in (".png", ".jpg", ".jpeg", ".JPG", ".PNG"):
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find image {name} in {images_dir}")


def _open_rgb(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.BILINEAR)
    return image


def _mask_to_color(mask_path: Path, size: tuple[int, int]) -> Image.Image:
    mask = Image.open(mask_path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    m = np.asarray(mask, dtype=np.float32) / 255.0
    color = np.zeros((*m.shape, 3), dtype=np.uint8)
    both = m >= 0.75
    one = (m >= 0.25) & (m < 0.75)
    none = m < 0.25
    color[none] = np.array([0, 0, 0], dtype=np.uint8)
    color[one] = np.array([255, 145, 0], dtype=np.uint8)
    color[both] = np.array([255, 0, 0], dtype=np.uint8)
    return Image.fromarray(color, mode="RGB")


def _overlay_mask(image: Image.Image, mask_path: Path, alpha: float = 0.45) -> Image.Image:
    color = _mask_to_color(mask_path, image.size)
    return Image.blend(image, color, alpha)


def _proxy_real_name(view: dict, base_split: dict, all_images: list[Path]) -> str:
    train_indices = base_split.get("train_indices")
    if train_indices:
        left_global = int(view["left_train_index"])
        right_global = int(view["right_train_index"])
        left_idx = int(train_indices[left_global])
        right_idx = int(train_indices[right_global])
        target_idx = int(round(left_idx + float(view.get("interval_t", 0.5)) * (right_idx - left_idx)))
        if all_images:
            target_idx = int(np.clip(target_idx, 0, len(all_images) - 1))
            return all_images[target_idx].name

    left_name = Path(view["left_image"]).stem
    right_name = Path(view["right_image"]).stem
    try:
        left_num = int(left_name.split("_")[-1])
        right_num = int(right_name.split("_")[-1])
        target_num = int(round(left_num + float(view.get("interval_t", 0.5)) * (right_num - left_num)))
        return f"frame_{target_num:05d}.png"
    except Exception:
        return view["left_image"]


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    x, y = xy
    draw.rectangle((x, y, x + 260, y + 24), fill=(255, 255, 255))
    draw.text((x + 6, y + 5), text, fill=(0, 0, 0))


def _make_sheet(
    manifest_path: Path,
    source_path: Path,
    base_split_path: Path,
    output_path: Path,
    max_views: int,
    thumb_width: int,
) -> None:
    manifest = _load_json(manifest_path)
    manifest_dir = manifest_path.parent
    base_split = _load_json(base_split_path)
    images_dir = _source_images_dir(source_path)
    all_images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])

    views = manifest.get("views", [])[:max_views]
    if not views:
        raise ValueError("No pseudo views to visualize.")

    first_image = Image.open(_resolve(manifest_dir, views[0]["image_path"])).convert("RGB")
    aspect = first_image.height / max(first_image.width, 1)
    thumb_size = (thumb_width, max(1, int(round(thumb_width * aspect))))
    labels = ["Left ref", "Pseudo", "Mask overlay", "Right ref", "Proxy real"]
    label_h = 30
    pad = 12
    row_h = thumb_size[1] + label_h + pad
    sheet_w = len(labels) * thumb_size[0] + (len(labels) + 1) * pad
    sheet_h = len(views) * row_h + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    for row, view in enumerate(views):
        y = pad + row * row_h
        pseudo = _open_rgb(_resolve(manifest_dir, view["image_path"]), thumb_size)
        mask_path = _resolve(manifest_dir, view["mask_path"])
        items = [
            _open_rgb(_find_image(images_dir, view["left_image"]), thumb_size),
            pseudo,
            _overlay_mask(pseudo, mask_path),
            _open_rgb(_find_image(images_dir, view["right_image"]), thumb_size),
            _open_rgb(_find_image(images_dir, _proxy_real_name(view, base_split, all_images)), thumb_size),
        ]
        for col, (label, image) in enumerate(zip(labels, items)):
            x = pad + col * (thumb_size[0] + pad)
            sheet.paste(image, (x, y + label_h))
            extra = ""
            if col == 2:
                extra = f" mean={float(view.get('mask_mean', 0.0)):.3f}"
            if col == 1:
                extra = f" t={float(view.get('interval_t', 0.5)):.2f}"
            _draw_label(draw, (x, y), label + extra)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo_manifest", type=Path, required=True)
    parser.add_argument("--source_path", type=Path, required=True)
    parser.add_argument("--base_split_manifest", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--max_views", type=int, default=9)
    parser.add_argument("--thumb_width", type=int, default=256)
    args = parser.parse_args()

    _make_sheet(
        args.pseudo_manifest.resolve(),
        args.source_path.resolve(),
        args.base_split_manifest.resolve(),
        args.output_path.resolve(),
        args.max_views,
        args.thumb_width,
    )
    print(f"[viz] wrote: {args.output_path.resolve()}")


if __name__ == "__main__":
    main()
