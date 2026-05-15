#!/usr/bin/env python3
"""Render raw pseudo views and geometry auxiliaries from pseudo camera poses."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
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


def _copy_relative(src: Path, dst_dir: Path) -> str:
    dst = dst_dir / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst.name


def _as_4x4(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = pose
        return out
    raise ValueError(f"Unsupported pose shape: {pose.shape}")


def _set_camera_pose(template, pose: np.ndarray, idx: int, image_name: str | None = None):
    cam = copy.deepcopy(template)
    pose = _as_4x4(pose)
    cam.uid = idx
    cam.colmap_id = 100000 + idx
    cam.image_name = image_name or f"pseudo_{idx:05d}"
    cam.R = pose[:3, :3].transpose()
    cam.T = pose[:3, 3]
    cam.world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T)).transpose(0, 1).cuda()
    cam.full_proj_transform = (
        cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    cam.camera_center = cam.world_view_transform.inverse()[3, :3]
    return cam


def _save_depth(path: Path, depth: torch.Tensor) -> np.ndarray:
    depth_np = depth.detach().squeeze().float().cpu().numpy().astype(np.float32)
    np.save(path, depth_np)
    return depth_np


def _save_depth_vis(path: Path, depth: np.ndarray) -> None:
    valid = np.isfinite(depth) & (depth > 0)
    vis = np.zeros_like(depth, dtype=np.float32)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        if hi > lo:
            vis[valid] = np.clip((depth[valid] - lo) / (hi - lo), 0.0, 1.0)
    Image.fromarray((vis * 255.0).astype(np.uint8), mode="L").save(path)


def _render_camera(cam, gaussians, pipe, bg):
    camera_pose = get_tensor_from_camera(cam.world_view_transform.transpose(0, 1))
    pkg = render(
        cam,
        gaussians,
        pipe,
        bg,
        camera_pose=camera_pose,
        return_alpha=True,
        return_depth=True,
    )
    pkg["render"] = pkg["render"].clamp(0.0, 1.0)
    pkg["alpha"] = pkg["alpha"].clamp(0.0, 1.0)
    return pkg


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
    manifest_path = args.pseudo_manifest.resolve()
    manifest_dir = manifest_path.parent
    manifest = _load_manifest(manifest_path)

    output_dir = (args.output_dir or manifest_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    alpha_dir = output_dir / "alpha"
    depth_dir = output_dir / "depth"
    depth_vis_dir = output_dir / "depth_vis"
    ref_depth_dir = output_dir / "ref_depth"
    ref_alpha_dir = output_dir / "ref_alpha"
    ref_depth_vis_dir = output_dir / "ref_depth_vis"
    for folder in [raw_dir, alpha_dir, depth_dir, depth_vis_dir, ref_depth_dir, ref_alpha_dir, ref_depth_vis_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    pose_path = _resolve(manifest_dir, manifest["pose_path"])
    poses = np.load(pose_path)
    manifest["pose_path"] = _copy_relative(pose_path, output_dir)

    source_poses = None
    source_pose_path_value = manifest.get("source_pose_path")
    if source_pose_path_value:
        source_pose_path = _resolve(manifest_dir, source_pose_path_value)
        if source_pose_path.exists():
            source_poses = np.load(source_pose_path)
            manifest["source_pose_path"] = _copy_relative(source_pose_path, output_dir)

    gaussians = GaussianModel(model_args.sh_degree)
    scene = Scene(model_args, gaussians, load_iteration=args.iteration, opt=args, shuffle=False)
    train_cameras = scene.getTrainCameras()
    if not train_cameras:
        raise ValueError("No training cameras available for pseudo render template.")
    template = train_cameras[0]

    bg_default = torch.tensor(
        [1, 1, 1] if model_args.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    views = manifest["views"]
    train_refs: dict[int, str] = {}
    for view in views:
        if "left_train_index" in view and "left_image" in view:
            train_refs[int(view["left_train_index"])] = Path(view["left_image"]).stem
        if "right_train_index" in view and "right_image" in view:
            train_refs[int(view["right_train_index"])] = Path(view["right_image"]).stem

    with torch.no_grad():
        if source_poses is not None:
            for train_idx, stem in tqdm(sorted(train_refs.items()), desc="Rendering reference depths"):
                if train_idx >= len(source_poses):
                    continue
                ref_cam = _set_camera_pose(template, source_poses[train_idx], 200000 + train_idx, stem)
                ref_pkg = _render_camera(ref_cam, gaussians, pipe, bg_default)
                depth_np = _save_depth(ref_depth_dir / f"{stem}.npy", ref_pkg["depth"])
                _save_depth_vis(ref_depth_vis_dir / f"{stem}.png", depth_np)
                torchvision.utils.save_image(ref_pkg["alpha"], ref_alpha_dir / f"{stem}.png")

        for view in tqdm(views, desc="Rendering pseudo views"):
            idx = int(view["pose_index"])
            image_name = view.get("image_name", f"pseudo_{idx:05d}")
            cam = _set_camera_pose(template, poses[idx], idx, image_name)
            pkg = _render_camera(cam, gaussians, pipe, bg_default)

            raw_rel = Path("raw") / f"{image_name}.png"
            alpha_rel = Path("alpha") / f"{image_name}.png"
            depth_rel = Path("depth") / f"{image_name}.npy"
            depth_vis_rel = Path("depth_vis") / f"{image_name}.png"
            torchvision.utils.save_image(pkg["render"], output_dir / raw_rel)
            torchvision.utils.save_image(pkg["alpha"], output_dir / alpha_rel)
            depth_np = _save_depth(output_dir / depth_rel, pkg["depth"])
            _save_depth_vis(output_dir / depth_vis_rel, depth_np)
            view["raw_image_path"] = str(raw_rel)
            view["alpha_path"] = str(alpha_rel)
            view["depth_path"] = str(depth_rel)
            view["depth_vis_path"] = str(depth_vis_rel)

    if source_poses is not None:
        manifest["ref_depth_dir"] = "ref_depth"
        manifest["ref_alpha_dir"] = "ref_alpha"
        manifest["ref_depth_vis_dir"] = "ref_depth_vis"

    rendered_manifest = output_dir / "pseudo_manifest_rendered.json"
    with rendered_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[pseudo] rendered views: {len(views)}")
    if source_poses is not None:
        print(f"[pseudo] rendered reference depths: {len(train_refs)}")
    print(f"[pseudo] wrote: {rendered_manifest}")


if __name__ == "__main__":
    main()
