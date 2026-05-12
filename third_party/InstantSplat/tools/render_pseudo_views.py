#!/usr/bin/env python3
"""Render raw pseudo views from pseudo camera poses."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.graphics_utils import getWorld2View2
from utils.pose_utils import get_tensor_from_camera


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _set_camera_pose(template, pose: np.ndarray, idx: int):
    cam = copy.deepcopy(template)
    cam.uid = idx
    cam.colmap_id = 100000 + idx
    cam.image_name = f"pseudo_{idx:05d}"
    cam.R = pose[:3, :3].transpose()
    cam.T = pose[:3, 3]
    cam.world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T)).transpose(0, 1).cuda()
    cam.full_proj_transform = (
        cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    cam.camera_center = cam.world_view_transform.inverse()[3, :3]
    return cam


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=False)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--pseudo_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    model_args = model.extract(args)
    pipe = pipeline.extract(args)
    model_path = Path(model_args.model_path).resolve()
    manifest_path = args.pseudo_manifest.resolve()
    manifest_dir = manifest_path.parent
    manifest = _load_manifest(manifest_path)
    pose_path = _resolve(manifest_dir, manifest["pose_path"])
    poses = np.load(pose_path)

    output_dir = args.output_dir or manifest_dir
    raw_dir = output_dir / "raw"
    alpha_dir = output_dir / "alpha"
    raw_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)

    gaussians = GaussianModel(model_args.sh_degree)
    scene = Scene(model_args, gaussians, load_iteration=args.iteration, opt=args, shuffle=False)
    train_cameras = scene.getTrainCameras()
    if not train_cameras:
        raise ValueError("No training cameras available for pseudo render template.")
    template = train_cameras[0]

    bg_default = torch.tensor([1, 1, 1] if model_args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    bg_black = torch.zeros((3,), dtype=torch.float32, device="cuda")
    bg_white = torch.ones((3,), dtype=torch.float32, device="cuda")

    views = manifest["views"]
    with torch.no_grad():
        for view in tqdm(views, desc="Rendering pseudo views"):
            idx = int(view["pose_index"])
            cam = _set_camera_pose(template, poses[idx], idx)
            camera_pose = get_tensor_from_camera(cam.world_view_transform.transpose(0, 1))
            raw = render(cam, gaussians, pipe, bg_default, camera_pose=camera_pose)["render"].clamp(0.0, 1.0)
            black = render(cam, gaussians, pipe, bg_black, camera_pose=camera_pose)["render"].clamp(0.0, 1.0)
            white = render(cam, gaussians, pipe, bg_white, camera_pose=camera_pose)["render"].clamp(0.0, 1.0)
            alpha = (1.0 - (white - black).mean(dim=0, keepdim=True)).clamp(0.0, 1.0)

            image_name = view.get("image_name", f"pseudo_{idx:05d}")
            raw_rel = Path("raw") / f"{image_name}.png"
            alpha_rel = Path("alpha") / f"{image_name}.png"
            torchvision.utils.save_image(raw, output_dir / raw_rel)
            torchvision.utils.save_image(alpha, output_dir / alpha_rel)
            view["raw_image_path"] = str(raw_rel)
            view["alpha_path"] = str(alpha_rel)

    rendered_manifest = output_dir / "pseudo_manifest_rendered.json"
    with rendered_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[pseudo] rendered views: {len(views)}")
    print(f"[pseudo] wrote: {rendered_manifest}")


if __name__ == "__main__":
    main()
