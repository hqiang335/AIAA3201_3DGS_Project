#!/usr/bin/env python3
"""Image-only InstantSplat metrics for rendered train/test folders.

This intentionally mirrors the image part of ``metrics.py`` (PSNR/SSIM/LPIPS),
but it does not require GT camera poses.  That makes it usable for FRONT-style
no-pose long-sequence experiments where image quality is the main comparable
signal.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ.pop("MKL_SERVICE_FORCE_INTEL", None)

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim


def _load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return tf.to_tensor(image).unsqueeze(0).to(device)


def evaluate_pair_dirs(renders_dir: Path, gt_dir: Path, output_dir: Path) -> dict:
    if not renders_dir.is_dir():
        raise FileNotFoundError(f"Missing renders dir: {renders_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Missing gt dir: {gt_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    render_files = sorted(
        path for path in renders_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not render_files:
        raise RuntimeError(f"No rendered images found in {renders_dir}")

    per_view = {}
    psnrs = []
    ssims = []
    lpipss = []

    metrics_txt = output_dir / "image_metrics.txt"
    with metrics_txt.open("w", encoding="utf-8") as handle:
        for render_path in tqdm(render_files, desc="Image metric evaluation"):
            gt_path = gt_dir / render_path.name
            if not gt_path.exists():
                raise FileNotFoundError(f"Missing GT image for {render_path.name}: {gt_path}")
            render = _load_rgb(render_path, device)
            gt = _load_rgb(gt_path, device)
            if render.shape != gt.shape:
                raise RuntimeError(
                    f"Shape mismatch for {render_path.name}: render={tuple(render.shape)} gt={tuple(gt.shape)}"
                )

            ssim_value = float(ssim(render, gt).detach().cpu())
            psnr_value = float(psnr(render, gt).detach().cpu())
            lpips_value = float(lpips(render, gt, net_type="vgg").detach().cpu())
            per_view[render_path.name] = {
                "PSNR": psnr_value,
                "SSIM": ssim_value,
                "LPIPS": lpips_value,
            }
            psnrs.append(psnr_value)
            ssims.append(ssim_value)
            lpipss.append(lpips_value)
            handle.write(
                f"{render_path.name}: PSNR {psnr_value:.6f}, "
                f"SSIM {ssim_value:.6f}, LPIPS {lpips_value:.6f}\n"
            )

    summary = {
        "num_images": len(render_files),
        "PSNR": float(torch.tensor(psnrs).mean()),
        "SSIM": float(torch.tensor(ssims).mean()),
        "LPIPS": float(torch.tensor(lpipss).mean()),
        "renders_dir": str(renders_dir),
        "gt_dir": str(gt_dir),
        "per_view": per_view,
    }
    (output_dir / "image_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("num_images", "PSNR", "SSIM", "LPIPS")}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model_path", type=Path)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--iteration", type=int, default=1000)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--renders_dir", type=Path)
    parser.add_argument("--gt_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()

    if args.renders_dir is None or args.gt_dir is None:
        if args.model_path is None:
            raise ValueError("Either --model_path or both --renders_dir/--gt_dir are required")
        method = args.method or f"ours_{args.iteration}"
        base = args.model_path / args.split / method
        renders_dir = base / "renders"
        gt_dir = base / "gt"
        output_dir = base
    else:
        renders_dir = args.renders_dir
        gt_dir = args.gt_dir
        output_dir = args.output_dir or renders_dir.parent

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluate_pair_dirs(renders_dir, gt_dir, output_dir)


if __name__ == "__main__":
    main()
