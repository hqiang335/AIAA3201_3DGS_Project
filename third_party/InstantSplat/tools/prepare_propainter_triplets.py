#!/usr/bin/env python3
"""Prepare three-frame ProPainter inputs for pseudo-view deghosting tests.

Each pseudo view is converted into a tiny video:
  0000.png = left real reference
  0001.png = raw pseudo render to be repaired
  0002.png = right real reference

Masks are black for the real references and mark suspicious pixels only on the
middle pseudo frame. ProPainter can then inpaint the middle frame using temporal
context from the two real references.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _resolve(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def _load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(path)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)


def _find_source_image(source_images_dir: Path, image_name: str) -> Path:
    stem = Path(image_name).stem
    candidates = [
        source_images_dir / image_name,
        source_images_dir / "images" / image_name,
        source_images_dir / f"{stem}.png",
        source_images_dir / "images" / f"{stem}.png",
        source_images_dir / f"{stem}.jpg",
        source_images_dir / "images" / f"{stem}.jpg",
        source_images_dir / f"{stem}.jpeg",
        source_images_dir / "images" / f"{stem}.jpeg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find source image for {image_name} under {source_images_dir}")


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


def _yellow_artifact_mask(raw: np.ndarray) -> np.ndarray:
    """A diagnostic mask for bright yellow floating artifacts.

    This is intentionally treated as a debug mask, not the final policy. It is
    useful for checking whether ProPainter can remove the obvious yellow blob in
    pseudo_00002 when given a reasonably localized artifact mask.
    """

    r, g, b = raw[..., 0], raw[..., 1], raw[..., 2]
    maxc = raw.max(axis=2)
    minc = raw.min(axis=2)
    sat = maxc - minc
    # Catch both saturated yellow splats and pale translucent yellow haze.
    yellow = (
        (r > 0.38)
        & (g > 0.38)
        & (b < 0.82)
        & (((r + g) * 0.5 - b) > 0.08)
        & (sat > 0.06)
    )
    return yellow.astype(np.float32)


def _temporal_outlier_mask(raw: np.ndarray, left: np.ndarray, right: np.ndarray, threshold: float) -> np.ndarray:
    err_left = np.mean(np.abs(raw - left), axis=2)
    err_right = np.mean(np.abs(raw - right), axis=2)
    return (np.minimum(err_left, err_right) > threshold).astype(np.float32)


def _load_good_mask(mask_path: Path | None, size: tuple[int, int]) -> np.ndarray | None:
    if mask_path is None or not mask_path.exists():
        return None
    image = Image.open(mask_path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.float32) / 255.0


def _build_masks(
    raw: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    good_mask: np.ndarray | None,
    artifact_mask: np.ndarray | None,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    h, w = raw.shape[:2]
    zero = np.zeros((h, w), dtype=np.float32)

    if good_mask is not None:
        masks["bad_conf0"] = (good_mask <= args.bad_conf0_threshold).astype(np.float32)
        masks["bad_conf_half"] = (good_mask < args.bad_conf_half_threshold).astype(np.float32)

    if artifact_mask is not None:
        masks["artifact_v2"] = (artifact_mask > 0.5).astype(np.float32)

    masks["yellow_debug"] = _yellow_artifact_mask(raw)
    masks["temporal_outlier"] = _temporal_outlier_mask(raw, left, right, args.temporal_delta_threshold)
    masks["temporal_yellow"] = masks["temporal_outlier"] * masks["yellow_debug"]

    if "bad_conf0" in masks:
        masks["bad_conf0_plus_yellow"] = np.maximum(masks["bad_conf0"], masks["yellow_debug"])
        masks["bad_conf0_plus_temporal"] = np.maximum(masks["bad_conf0"], masks["temporal_outlier"])
    if "artifact_v2" in masks:
        masks["artifact_v2_plus_yellow"] = np.maximum(masks["artifact_v2"], masks["yellow_debug"])
        masks["artifact_v2_plus_temporal"] = np.maximum(masks["artifact_v2"], masks["temporal_outlier"])
    if good_mask is not None:
        masks["temporal_lowconf"] = masks["temporal_outlier"] * (good_mask < args.bad_conf_half_threshold).astype(np.float32)

    selected = args.mask_sources or list(masks)
    out = {}
    for name in selected:
        mask = masks.get(name, zero)
        out[name] = _morph(mask, args.mask_dilate, args.mask_close)
    return out


def _read_manifest(path: Path) -> tuple[dict, Path]:
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest, path.parent


def _views_by_name(manifest: dict) -> dict[str, dict]:
    return {view["image_name"]: view for view in manifest.get("views", [])}


def _label(image: Image.Image, text: str) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = None
    draw.rectangle((0, 0, canvas.width, 22), fill=(0, 0, 0))
    draw.text((5, 4), text, fill=(255, 255, 255), font=font)
    return canvas


def _make_contact_sheet(rows: list[tuple[str, Path, Path, Path, Path]], out_path: Path) -> None:
    thumbs = []
    for mask_name, left_path, raw_path, right_path, mask_path in rows:
        left = _label(Image.open(left_path).convert("RGB"), f"{mask_name}: left")
        raw = _label(Image.open(raw_path).convert("RGB"), "raw pseudo")
        right = _label(Image.open(right_path).convert("RGB"), "right")
        mask = _label(Image.open(mask_path).convert("L").convert("RGB"), "inpaint mask")
        thumbs.append((left, raw, right, mask))
    if not thumbs:
        return
    w, h = thumbs[0][0].size
    gap = 8
    sheet = Image.new("RGB", (4 * w + 3 * gap, len(thumbs) * h + (len(thumbs) - 1) * gap), "white")
    for row_idx, row in enumerate(thumbs):
        y = row_idx * (h + gap)
        for col_idx, image in enumerate(row):
            sheet.paste(image, (col_idx * (w + gap), y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo_manifest", required=True)
    parser.add_argument("--source_images_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--confidence_manifest", default=None)
    parser.add_argument("--artifact_mask_dir", default=None)
    parser.add_argument("--warped_left_dir", default=None)
    parser.add_argument("--warped_right_dir", default=None)
    parser.add_argument("--pseudo_ids", nargs="*", type=int, default=None)
    parser.add_argument("--image_names", nargs="*", default=None)
    parser.add_argument("--mask_sources", nargs="*", default=None)
    parser.add_argument("--bad_conf0_threshold", type=float, default=0.01)
    parser.add_argument("--bad_conf_half_threshold", type=float, default=0.75)
    parser.add_argument("--temporal_delta_threshold", type=float, default=0.18)
    parser.add_argument("--mask_dilate", type=int, default=3)
    parser.add_argument("--mask_close", type=int, default=3)
    args = parser.parse_args()

    manifest_path = Path(args.pseudo_manifest)
    manifest, manifest_dir = _read_manifest(manifest_path)
    source_images_dir = Path(args.source_images_dir)
    output_dir = Path(args.output_dir)
    input_root = output_dir / "inputs"
    input_root.mkdir(parents=True, exist_ok=True)

    confidence_views = {}
    confidence_dir = None
    if args.confidence_manifest:
        confidence_manifest, confidence_dir = _read_manifest(Path(args.confidence_manifest))
        confidence_views = _views_by_name(confidence_manifest)

    views = manifest.get("views", [])
    if args.pseudo_ids is not None:
        wanted = set(args.pseudo_ids)
        views = [view for view in views if int(view["pseudo_id"]) in wanted]
    if args.image_names is not None:
        wanted_names = set(args.image_names)
        views = [view for view in views if view["image_name"] in wanted_names]

    commands = []
    metadata = {
        "pseudo_manifest": str(manifest_path),
        "source_images_dir": str(source_images_dir),
        "confidence_manifest": args.confidence_manifest,
        "artifact_mask_dir": args.artifact_mask_dir,
        "warped_left_dir": args.warped_left_dir,
        "warped_right_dir": args.warped_right_dir,
        "items": [],
    }
    sheet_rows: list[tuple[str, Path, Path, Path, Path]] = []

    for view in views:
        image_name = view["image_name"]
        raw_path = _resolve(manifest_dir, view.get("raw_image_path"))
        if raw_path is None or not raw_path.exists():
            raise FileNotFoundError(f"Missing raw pseudo image for {image_name}: {raw_path}")
        if args.warped_left_dir and args.warped_right_dir:
            left_path = Path(args.warped_left_dir) / f"{image_name}.png"
            right_path = Path(args.warped_right_dir) / f"{image_name}.png"
            if not left_path.exists() or not right_path.exists():
                raise FileNotFoundError(f"Missing warped references for {image_name}: {left_path}, {right_path}")
        else:
            left_path = _find_source_image(source_images_dir, view["left_image"])
            right_path = _find_source_image(source_images_dir, view["right_image"])

        raw = _load_rgb(raw_path)
        size = (raw.shape[1], raw.shape[0])
        left = _load_rgb(left_path, size)
        right = _load_rgb(right_path, size)

        good_mask = None
        confidence_view = confidence_views.get(image_name)
        if confidence_view and confidence_dir is not None:
            good_mask = _load_good_mask(_resolve(confidence_dir, confidence_view.get("mask_path")), size)

        artifact_mask = None
        if args.artifact_mask_dir:
            artifact_path = Path(args.artifact_mask_dir) / f"{image_name}.png"
            artifact_mask = _load_good_mask(artifact_path, size)

        masks = _build_masks(raw, left, right, good_mask, artifact_mask, args)
        for mask_name, mask in masks.items():
            video_dir = input_root / mask_name / image_name / "video"
            mask_dir = input_root / mask_name / image_name / "mask"
            _save_rgb(video_dir / "0000.png", left)
            _save_rgb(video_dir / "0001.png", raw)
            _save_rgb(video_dir / "0002.png", right)
            _save_mask(mask_dir / "0000.png", np.zeros_like(mask))
            _save_mask(mask_dir / "0001.png", mask)
            _save_mask(mask_dir / "0002.png", np.zeros_like(mask))
            mask_mean = float(mask.mean())
            commands.append(
                {
                    "mask_name": mask_name,
                    "image_name": image_name,
                    "video_dir": str(video_dir),
                    "mask_dir": str(mask_dir),
                    "mask_mean": mask_mean,
                }
            )
            metadata["items"].append(commands[-1])
            sheet_rows.append((mask_name, video_dir / "0000.png", video_dir / "0001.png", video_dir / "0002.png", mask_dir / "0001.png"))

    with (output_dir / "propainter_triplets.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with (output_dir / "run_propainter_commands.sh").open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        f.write("source /etc/network_turbo 2>/dev/null || true\n")
        f.write("source /root/miniconda3/etc/profile.d/conda.sh\n")
        f.write("conda activate instantsplat\n")
        f.write("cd /root/autodl-tmp/AIAA3201_3DGS_Project/third_party/ProPainter\n")
        for item in commands:
            out_dir = output_dir / "propainter_raw_outputs" / item["mask_name"] / item["image_name"]
            f.write(
                "python inference_propainter.py "
                f"--video '{item['video_dir']}' "
                f"--mask '{item['mask_dir']}' "
                f"--output '{out_dir}' "
                "--height 256 --width 256 --save_frames --fp16 "
                "--neighbor_length 3 --ref_stride 1 --subvideo_length 3 --raft_iter 10 --mask_dilation 0\n"
            )
    shutil.copymode(__file__, output_dir / "run_propainter_commands.sh")
    _make_contact_sheet(sheet_rows, output_dir / "propainter_inputs_sheet.png")
    print(f"Prepared {len(commands)} ProPainter triplets under {output_dir}")
    for item in commands:
        print(f"{item['image_name']} {item['mask_name']} mask_mean={item['mask_mean']:.4f}")


if __name__ == "__main__":
    main()
