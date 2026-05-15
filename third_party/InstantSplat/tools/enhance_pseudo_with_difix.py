#!/usr/bin/env python3
"""Enhance raw pseudo views with Difix3D reference-guided diffusion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _load_rgb(path: Path, resolution: tuple[int, int] | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if resolution is not None and image.size != resolution:
        image = image.resize(resolution, Image.BICUBIC)
    return image


def _load_difix_pipeline(difix_repo: Path, model_id: str, dtype: torch.dtype):
    if difix_repo:
        sys.path.insert(0, str(difix_repo.resolve()))
    from pipeline_difix import DifixPipeline

    pipe = DifixPipeline.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
    pipe.to("cuda")
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return pipe


def _run_one(pipe, prompt: str, raw: Image.Image, ref: Image.Image, args) -> Image.Image:
    with torch.inference_mode():
        result = pipe(
            prompt,
            image=raw,
            ref_image=ref,
            num_inference_steps=args.num_inference_steps,
            timesteps=[args.timestep],
            guidance_scale=args.guidance_scale,
        )
    return result.images[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo_manifest", type=Path, required=True)
    parser.add_argument("--source_images_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--difix_repo", type=Path, default=Path("/root/autodl-tmp/AIAA3201_3DGS_Project/third_party/Difix3D/src"))
    parser.add_argument("--model_id", type=str, default="nvidia/difix_ref")
    parser.add_argument("--prompt", type=str, default="remove degradation")
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--timestep", type=int, default=199)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    manifest_path = args.pseudo_manifest.resolve()
    manifest_dir = manifest_path.parent
    output_dir = args.output_dir or manifest_dir
    left_dir = output_dir / "enhanced_left"
    right_dir = output_dir / "enhanced_right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    views = manifest.get("views", [])
    if args.max_views > 0:
        views = views[: args.max_views]
    if not views:
        raise ValueError(f"No pseudo views found in {manifest_path}")

    dtype = torch.float32 if args.fp32 else torch.float16
    pipe = _load_difix_pipeline(args.difix_repo, args.model_id, dtype)

    for view in tqdm(views, desc="Enhancing pseudo views with Difix3D"):
        image_name = view.get("image_name", f"pseudo_{int(view['pose_index']):05d}")
        left_out = left_dir / f"{image_name}.png"
        right_out = right_dir / f"{image_name}.png"
        if args.skip_existing and left_out.exists() and right_out.exists():
            continue

        raw = _load_rgb(_resolve(manifest_dir, view["raw_image_path"]))
        resolution = raw.size
        left_ref = _load_rgb(args.source_images_dir / view["left_image"], resolution)
        right_ref = _load_rgb(args.source_images_dir / view["right_image"], resolution)

        left_img = _run_one(pipe, args.prompt, raw, left_ref, args)
        right_img = _run_one(pipe, args.prompt, raw, right_ref, args)
        left_img.save(left_out)
        right_img.save(right_out)

    enhanced_manifest = output_dir / "pseudo_manifest_enhanced.json"
    manifest["enhanced_left_dir"] = str(left_dir.relative_to(output_dir))
    manifest["enhanced_right_dir"] = str(right_dir.relative_to(output_dir))
    with enhanced_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[pseudo] enhanced views: {len(views)}")
    print(f"[pseudo] wrote: {left_dir}")
    print(f"[pseudo] wrote: {right_dir}")
    print(f"[pseudo] wrote: {enhanced_manifest}")


if __name__ == "__main__":
    main()
