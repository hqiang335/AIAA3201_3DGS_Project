#!/usr/bin/env python3
"""Compare global pseudo views against short-window pseudo views.

This is a proxy evaluation tool: pseudo poses usually do not have exact GT.  For
DL3DV-style ordered frames, it uses the temporally nearest real frame implied by
left/right frame names and interpolation t as a visual/metric proxy.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from skimage.metrics import structural_similarity
except Exception:  # pragma: no cover
    structural_similarity = None


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _frame_num(name: str) -> int:
    matches = re.findall(r"\d+", Path(name).stem)
    if not matches:
        raise ValueError(f"Could not parse frame number from {name}")
    return int(matches[-1])


def _key(view: dict) -> tuple[str, str, int]:
    return (
        Path(view["left_image"]).name,
        Path(view["right_image"]).name,
        int(round(float(view.get("interval_t", 0.0)) * 10000)),
    )


def _find_proxy_image(source_images_dir: Path, left_name: str, right_name: str, t: float) -> Path:
    left_num = _frame_num(left_name)
    right_num = _frame_num(right_name)
    target = int(round((1.0 - t) * left_num + t * right_num))
    candidates = sorted(source_images_dir.glob("*.png")) + sorted(source_images_dir.glob("*.jpg"))
    by_num = {}
    for path in candidates:
        try:
            by_num[_frame_num(path.name)] = path
        except ValueError:
            continue
    if not by_num:
        raise FileNotFoundError(f"No numbered images under {source_images_dir}")
    nearest = min(by_num, key=lambda n: abs(n - target))
    return by_num[nearest]


def _load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def _metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((pred - ref) ** 2))
    mae = float(np.mean(np.abs(pred - ref)))
    psnr = 99.0 if mse <= 1e-12 else float(-10.0 * math.log10(mse))
    out = {"psnr": psnr, "mae": mae}
    if structural_similarity is not None:
        out["ssim"] = float(structural_similarity(ref, pred, channel_axis=-1, data_range=1.0))
    return out


def _to_pil(image: np.ndarray, width: int) -> Image.Image:
    pil = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    if pil.width == width:
        return pil
    height = max(1, int(round(pil.height * (width / pil.width))))
    return pil.resize((width, height), Image.Resampling.BICUBIC)


def _label_tile(label: str, image: np.ndarray | None, width: int, height: int | None = None) -> Image.Image:
    if image is None:
        tile = Image.new("RGB", (width, height or int(width * 0.56)), (35, 35, 35))
    else:
        tile = _to_pil(image, width)
    label_h = 28
    out = Image.new("RGB", (tile.width, tile.height + label_h), (255, 255, 255))
    out.paste(tile, (0, label_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((6, 5), label, fill=(0, 0, 0), font=font)
    return out


def _save_contact_sheet(rows: list[dict], output_path: Path, thumb_width: int, max_rows: int) -> None:
    rows = rows[:max_rows] if max_rows > 0 else rows
    if not rows:
        return
    columns = [
        ("proxy", "proxy real"),
        ("baseline_raw", "global raw"),
        ("window_raw", "window raw"),
        ("baseline_final", "global Difix/fused"),
        ("window_final", "window Difix/fused"),
    ]
    rendered_rows = []
    for row in rows:
        tiles = []
        first_h = None
        for key, label in columns:
            img = row.get(key)
            tile = _label_tile(label, img, thumb_width, first_h)
            if first_h is None:
                first_h = tile.height - 28
            tiles.append(tile)
        row_w = sum(t.width for t in tiles)
        row_h = max(t.height for t in tiles) + 30
        row_img = Image.new("RGB", (row_w, row_h), (245, 245, 245))
        x = 0
        for tile in tiles:
            row_img.paste(tile, (x, 30))
            x += tile.width
        draw = ImageDraw.Draw(row_img)
        draw.text((6, 6), row["title"], fill=(0, 0, 0))
        rendered_rows.append(row_img)

    sheet_w = max(r.width for r in rendered_rows)
    sheet_h = sum(r.height for r in rendered_rows)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    y = 0
    for row_img in rendered_rows:
        sheet.paste(row_img, (0, y))
        y += row_img.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _aggregate(metric_rows: list[dict]) -> dict[str, dict[str, float]]:
    bucket: dict[str, list[dict]] = defaultdict(list)
    for row in metric_rows:
        for method, values in row["metrics"].items():
            bucket[method].append(values)
    summary = {}
    for method, values in bucket.items():
        keys = sorted({k for v in values for k in v})
        summary[method] = {
            k: float(np.mean([v[k] for v in values if k in v])) for k in keys
        }
        summary[method]["count"] = len(values)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window_validation_manifest", type=Path, required=True)
    parser.add_argument("--source_images_dir", type=Path, required=True)
    parser.add_argument("--baseline_manifest", type=Path, required=True)
    parser.add_argument("--baseline_final_dir", type=Path, default=None)
    parser.add_argument("--window_final_subdir", type=str, default="difix_trinary_s015_a025/images")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--thumb_width", type=int, default=320)
    parser.add_argument("--max_sheet_rows", type=int, default=40)
    args = parser.parse_args()

    window_manifest = _load_json(args.window_validation_manifest.resolve())
    baseline_manifest_path = args.baseline_manifest.resolve()
    baseline_manifest = _load_json(baseline_manifest_path)
    baseline_dir = baseline_manifest_path.parent
    baseline_by_key = {_key(v): v for v in baseline_manifest["views"]}

    metric_rows = []
    sheet_rows = []
    for window in window_manifest["windows"]:
        rendered_manifest_path = Path(window["pseudo_manifest_rendered"])
        if not rendered_manifest_path.exists():
            continue
        rendered = _load_json(rendered_manifest_path)
        rendered_dir = rendered_manifest_path.parent
        window_raw_dir = rendered_dir / "raw"
        window_final_dir = rendered_dir / args.window_final_subdir
        if not window_final_dir.exists():
            # Fall back to raw-only comparison when Difix/fusion has not been run.
            window_final_dir = None

        for view in rendered["views"]:
            key = _key(view)
            base_view = baseline_by_key.get(key)
            if base_view is None:
                continue
            image_name = view.get("image_name", f"pseudo_{int(view['pose_index']):05d}")
            base_image_name = base_view.get("image_name", f"pseudo_{int(base_view['pose_index']):05d}")
            proxy_path = _find_proxy_image(
                args.source_images_dir.resolve(),
                view["left_image"],
                view["right_image"],
                float(view.get("interval_t", 0.0)),
            )

            baseline_raw_path = _resolve(baseline_dir, base_view["raw_image_path"])
            window_raw_path = window_raw_dir / f"{image_name}.png"
            baseline_final_path = args.baseline_final_dir / f"{base_image_name}.png" if args.baseline_final_dir else None
            window_final_path = window_final_dir / f"{image_name}.png" if window_final_dir else None

            if not baseline_raw_path.exists() or not window_raw_path.exists():
                continue
            proxy = _load_rgb(proxy_path)
            size = (proxy.shape[1], proxy.shape[0])
            candidates = {
                "baseline_raw": _load_rgb(baseline_raw_path, size),
                "window_raw": _load_rgb(window_raw_path, size),
            }
            if baseline_final_path is not None and baseline_final_path.exists():
                candidates["baseline_final"] = _load_rgb(baseline_final_path, size)
            if window_final_path is not None and window_final_path.exists():
                candidates["window_final"] = _load_rgb(window_final_path, size)

            metrics = {name: _metrics(img, proxy) for name, img in candidates.items()}
            metric_rows.append(
                {
                    "window_tag": window["window_tag"],
                    "left_image": view["left_image"],
                    "right_image": view["right_image"],
                    "interval_t": float(view.get("interval_t", 0.0)),
                    "proxy_image": str(proxy_path),
                    "baseline_image_name": base_image_name,
                    "window_image_name": image_name,
                    "metrics": metrics,
                }
            )
            sheet_rows.append(
                {
                    "title": f"{window['window_tag']} | {view['left_image']} -> {view['right_image']} | t={float(view.get('interval_t', 0.0)):.2f} | proxy={proxy_path.name}",
                    "proxy": proxy,
                    "baseline_raw": candidates.get("baseline_raw"),
                    "window_raw": candidates.get("window_raw"),
                    "baseline_final": candidates.get("baseline_final"),
                    "window_final": candidates.get("window_final"),
                }
            )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _aggregate(metric_rows)
    payload = {"summary": summary, "rows": metric_rows}
    with (output_dir / "window_pseudo_proxy_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _save_contact_sheet(sheet_rows, output_dir / "window_pseudo_contact_sheet.png", args.thumb_width, args.max_sheet_rows)

    print(json.dumps(summary, indent=2))
    print(f"[compare] rows: {len(metric_rows)}")
    print(f"[compare] wrote: {output_dir / 'window_pseudo_proxy_metrics.json'}")
    print(f"[compare] wrote: {output_dir / 'window_pseudo_contact_sheet.png'}")


if __name__ == "__main__":
    main()
