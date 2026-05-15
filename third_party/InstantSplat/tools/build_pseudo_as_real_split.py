#!/usr/bin/env python3
"""Build a time-ordered pseudo-as-real dataset for InstantSplat init_geo.

The output source directory contains an ``images`` folder with symlinks to the
selected real training frames, ViewCrafter pseudo frames, and held-out real test
frames. The explicit split manifest is the source of truth for ordering.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path


def parse_frame_time(name: str) -> float:
    matches = re.findall(r"\d+", Path(name).stem)
    if not matches:
        raise ValueError(f"Cannot infer frame time from filename: {name}")
    return float(int(matches[-1]))


def link_or_copy(src: Path, dst: Path, copy: bool, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {dst}")
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def make_safe_pseudo_name(order: int, view: dict) -> str:
    left = Path(view["left_image"]).stem
    right = Path(view["right_image"]).stem
    t_tag = int(round(float(view["interval_t"]) * 1000))
    src_suffix = Path(view["image_path"]).suffix or ".png"
    return f"train_{order:05d}_pseudo_{left}_to_{right}_t{t_tag:04d}{src_suffix}"


def make_safe_real_name(order: int, basename: str) -> str:
    return f"train_{order:05d}_real_{basename}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real_source", type=Path, default=None, help="Real dataset root with an images directory.")
    parser.add_argument("--pseudo_manifest", type=Path, required=True, help="ViewCrafter pseudo_manifest_train.json.")
    parser.add_argument("--output_source", type=Path, required=True, help="Output source root for the pseudo-as-real run.")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking them.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output images/manifest.")
    args = parser.parse_args()

    with open(args.pseudo_manifest, "r", encoding="utf-8") as f:
        pseudo_manifest = json.load(f)

    real_images_dir = Path(pseudo_manifest.get("source_images_dir", ""))
    if args.real_source is not None:
        real_images_dir = args.real_source / "images"
    if not real_images_dir.exists():
        raise FileNotFoundError(f"Real images directory does not exist: {real_images_dir}")

    train_basenames = pseudo_manifest.get("train_basenames")
    if not train_basenames:
        raise ValueError("Pseudo manifest is missing train_basenames")

    base_split_manifest = pseudo_manifest.get("base_split_manifest")
    if not base_split_manifest:
        raise ValueError("Pseudo manifest is missing base_split_manifest; cannot recover held-out test frames.")
    with open(base_split_manifest, "r", encoding="utf-8") as f:
        base_split = json.load(f)
    base_train_basenames = base_split.get("train_basenames")
    base_train_times = base_split.get("train_indices")
    if not base_train_basenames or not base_train_times:
        raise ValueError(f"Base split manifest lacks train_basenames/train_indices: {base_split_manifest}")
    if len(base_train_basenames) != len(base_train_times):
        raise ValueError("train_basenames/train_indices length mismatch in base split manifest")

    test_basenames = base_split.get("test_basenames")
    test_times = base_split.get("test_indices")
    if not test_basenames or not test_times:
        raise ValueError(f"Base split manifest lacks test_basenames/test_indices: {base_split_manifest}")
    if len(test_basenames) != len(test_times):
        raise ValueError("test_basenames/test_indices length mismatch in base split manifest")

    pseudo_root = args.pseudo_manifest.parent
    output_images_dir = args.output_source / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    train_entries = []
    real_time_by_basename = {
        basename: float(time_value)
        for basename, time_value in zip(base_train_basenames, base_train_times)
    }
    for basename in train_basenames:
        time_value = real_time_by_basename.get(basename, parse_frame_time(basename))
        train_entries.append(
            {
                "kind": "real",
                "source_basename": basename,
                "source_path": str(real_images_dir / basename),
                "time": time_value,
            }
        )

    for view in pseudo_manifest.get("views", []):
        left_image = view["left_image"]
        right_image = view["right_image"]
        left_time = real_time_by_basename.get(left_image, parse_frame_time(left_image))
        right_time = real_time_by_basename.get(right_image, parse_frame_time(right_image))
        t = float(view["interval_t"])
        time_value = left_time + t * (right_time - left_time)
        train_entries.append(
            {
                "kind": "pseudo",
                "pseudo_id": view.get("pseudo_id"),
                "source_path": str((pseudo_root / view["image_path"]).resolve()),
                "time": time_value,
                "left_image": left_image,
                "right_image": right_image,
                "interval_t": t,
                "loss_weight": view.get("loss_weight"),
                "mask_path": view.get("feature_mask_path") or view.get("mask_path"),
            }
        )

    train_entries.sort(key=lambda item: item["time"])
    for order, entry in enumerate(train_entries):
        src = Path(entry["source_path"])
        if not src.exists():
            raise FileNotFoundError(f"Training image does not exist: {src}")
        if entry["kind"] == "real":
            image_name = make_safe_real_name(order, entry["source_basename"])
        else:
            image_name = make_safe_pseudo_name(
                order,
                {
                    "left_image": entry["left_image"],
                    "right_image": entry["right_image"],
                    "interval_t": entry["interval_t"],
                    "image_path": entry["source_path"],
                },
            )
        link_or_copy(src, output_images_dir / image_name, copy=args.copy, overwrite=args.overwrite)
        entry["image"] = image_name

    test_entries = []
    for basename, time_value in zip(test_basenames, test_times):
        src = real_images_dir / basename
        if not src.exists():
            raise FileNotFoundError(f"Test image does not exist: {src}")
        link_or_copy(src, output_images_dir / basename, copy=args.copy, overwrite=args.overwrite)
        test_entries.append(
            {
                "kind": "real_test",
                "image": basename,
                "source_path": str(src),
                "time": float(time_value),
            }
        )
    test_entries.sort(key=lambda item: item["time"])

    explicit_manifest = {
        "schema": "instantsplat_explicit_split_v1",
        "description": "10 real DL3DV-2 train views plus ViewCrafter pseudo views, sorted by temporal time.",
        "real_images_dir": str(real_images_dir),
        "pseudo_manifest": str(args.pseudo_manifest.resolve()),
        "base_split_manifest": str(Path(base_split_manifest).resolve()),
        "n_train": len(train_entries),
        "n_test": len(test_entries),
        "train": [
            {
                key: value
                for key, value in entry.items()
                if key in {
                    "image",
                    "kind",
                    "time",
                    "source_basename",
                    "pseudo_id",
                    "left_image",
                    "right_image",
                    "interval_t",
                    "loss_weight",
                    "mask_path",
                }
            }
            for entry in train_entries
        ],
        "test": [
            {
                "image": entry["image"],
                "kind": entry["kind"],
                "time": entry["time"],
            }
            for entry in test_entries
        ],
    }

    manifest_path = args.output_source / "explicit_split_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest already exists: {manifest_path}")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(explicit_manifest, f, indent=2)

    preview_path = args.output_source / "order_preview.txt"
    with open(preview_path, "w", encoding="utf-8") as f:
        f.write("TRAIN ORDER\n")
        for entry in explicit_manifest["train"]:
            f.write(f'{entry["time"]:10.4f}  {entry["kind"]:6s}  {entry["image"]}\n')
        f.write("\nTEST ORDER\n")
        for entry in explicit_manifest["test"]:
            f.write(f'{entry["time"]:10.4f}  {entry["kind"]:9s}  {entry["image"]}\n')

    print(f"Wrote source: {args.output_source}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote preview: {preview_path}")
    print(f"Train views: {len(train_entries)}")
    print(f"Test views: {len(test_entries)}")


if __name__ == "__main__":
    main()
