import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


@dataclass
class ImageRecord:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


@dataclass
class CameraRecord:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class WindowRecord:
    index: int
    start: int
    image_paths: List[str]
    scene_path: str
    init_model_path: str
    gaussian_model_path: str
    sparse_path: str
    transform: Dict
    alignment: Dict


def natural_key(path: Path) -> Tuple:
    parts = re.split(r"(\d+)", path.stem)
    return tuple(int(part) if part.isdigit() else part for part in parts) + (path.suffix.lower(),)


def resolve_image_dir(source: Path) -> Path:
    if source.name == "images" and source.is_dir():
        return source
    image_dir = source / "images"
    if image_dir.is_dir():
        return image_dir
    raise FileNotFoundError(f"Cannot find images directory under {source}")


def list_images(image_dir: Path) -> List[Path]:
    return sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )


def sample_images(images: Sequence[Path], sample_stride: int, n_views: int, offset: int) -> List[Path]:
    if sample_stride < 1:
        raise ValueError("sample_stride must be at least 1")
    sampled = list(images[offset::sample_stride])
    if n_views > 0:
        sampled = sampled[:n_views]
    if not sampled:
        raise ValueError("No images selected")
    return sampled


def make_window_starts(n_images: int, window_size: int, overlap: int) -> List[int]:
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if overlap < 1 or overlap >= window_size:
        raise ValueError("overlap must be in [1, window_size)")
    if n_images < window_size:
        raise ValueError(f"Need at least {window_size} images, got {n_images}")
    stride = window_size - overlap
    starts = list(range(0, n_images - window_size + 1, stride))
    tail = n_images - window_size
    if starts[-1] != tail:
        starts.append(tail)
    return starts


def safe_clean_dir(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def materialize_scene(scene_path: Path, images: Sequence[Path], link_mode: str, overwrite: bool) -> None:
    if overwrite:
        safe_clean_dir(scene_path)
    image_dir = scene_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        link_or_copy(image, image_dir / image.name, link_mode)


def load_colmap_helpers():
    helper_path = Path(__file__).resolve().parent / "scene" / "colmap_loader.py"
    spec = importlib.util.spec_from_file_location("instantsplat_colmap_loader", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load COLMAP helper from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_images_text(path: Path) -> Dict[str, ImageRecord]:
    colmap = load_colmap_helpers()
    images = {}
    for image in colmap.read_extrinsics_text(str(path)).values():
        images[image.name] = ImageRecord(
            image_id=int(image.id),
            qvec=np.asarray(image.qvec, dtype=float),
            tvec=np.asarray(image.tvec, dtype=float),
            camera_id=int(image.camera_id),
            name=image.name,
        )
    return images


def read_cameras_text(path: Path) -> Dict[int, CameraRecord]:
    colmap = load_colmap_helpers()
    cameras = {}
    for camera in colmap.read_intrinsics_text(str(path)).values():
        cameras[int(camera.id)] = CameraRecord(
            camera_id=int(camera.id),
            model=str(camera.model),
            width=int(camera.width),
            height=int(camera.height),
            params=np.asarray(camera.params, dtype=float),
        )
    return cameras


def image_w2c(record: ImageRecord) -> np.ndarray:
    colmap = load_colmap_helpers()
    w2c = np.eye(4, dtype=float)
    w2c[:3, :3] = colmap.qvec2rotmat(record.qvec)
    w2c[:3, 3] = record.tvec
    return w2c


def w2c_to_center(w2c: np.ndarray) -> np.ndarray:
    return -w2c[:3, :3].T @ w2c[:3, 3]


def camera_center(record: ImageRecord) -> np.ndarray:
    return w2c_to_center(image_w2c(record))


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or src.shape[0] < 3:
        raise ValueError("Need at least 3 paired 3D points for Sim(3) alignment")
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = (dst_centered.T @ src_centered) / src.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(src_centered**2, axis=1))
    if variance <= 1e-12:
        raise ValueError("Cannot align degenerate camera centers")
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = dst_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def average_rotations(rotations: np.ndarray) -> np.ndarray:
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3) or rotations.shape[0] == 0:
        raise ValueError("Need at least one 3x3 rotation matrix")
    u, _, vt = np.linalg.svd(np.sum(rotations, axis=0))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def fixed_rotation_similarity(src: np.ndarray, dst: np.ndarray, rotation: np.ndarray) -> Tuple[float, np.ndarray]:
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or src.shape[0] < 3:
        raise ValueError("Need at least 3 paired 3D points for fixed-rotation Sim(3) alignment")
    rotated_src = src @ rotation.T
    src_mean = rotated_src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = rotated_src - src_mean
    dst_centered = dst - dst_mean
    denominator = float(np.sum(src_centered**2))
    if denominator <= 1e-12:
        raise ValueError("Cannot align degenerate camera centers")
    scale = float(np.sum(src_centered * dst_centered) / denominator)
    if scale <= 1e-12:
        raise ValueError(f"Estimated non-positive scale: {scale}")
    translation = dst_mean - scale * src_mean
    return scale, translation


def apply_similarity(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (points @ rotation.T) + translation


def transform_w2c(w2c: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    c2w = np.linalg.inv(w2c)
    c2w_global = np.eye(4, dtype=float)
    c2w_global[:3, :3] = rotation @ c2w[:3, :3]
    c2w_global[:3, 3] = scale * rotation @ c2w[:3, 3] + translation
    return np.linalg.inv(c2w_global)


def compose_delta_similarity(transform: Dict, delta_scale: float, delta_rotation: np.ndarray, delta_translation: np.ndarray) -> Dict:
    base_scale = float(transform["scale"])
    base_rotation = np.asarray(transform["rotation"], dtype=float)
    base_translation = np.asarray(transform["translation"], dtype=float)
    return {
        "scale": float(delta_scale * base_scale),
        "rotation": delta_rotation @ base_rotation,
        "translation": delta_scale * (delta_rotation @ base_translation) + delta_translation,
    }


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (np.trace(rotation) - 1.0) / 2.0
    value = float(np.clip(value, -1.0, 1.0))
    return float(np.degrees(np.arccos(value)))


def point_set_diag(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    deltas = points[:, None, :] - points[None, :, :]
    return float(np.max(np.linalg.norm(deltas, axis=-1)))


def read_ply_points(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from plyfile import PlyData

        ply = PlyData.read(str(path))
        vertex = ply["vertex"]
        xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(float)
        if {"red", "green", "blue"}.issubset(set(vertex.data.dtype.names or [])):
            rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)
        else:
            rgb = np.full((xyz.shape[0], 3), 127, dtype=np.uint8)
        return xyz, rgb
    except Exception:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        header_end = next(i for i, line in enumerate(lines) if line.strip() == "end_header")
        data = np.loadtxt(lines[header_end + 1 :], dtype=float)
        if data.ndim == 1:
            data = data[None, :]
        xyz = data[:, :3].astype(float)
        rgb = data[:, 6:9].clip(0, 255).astype(np.uint8) if data.shape[1] >= 9 else np.full((xyz.shape[0], 3), 127, dtype=np.uint8)
        return xyz, rgb


def write_ply_points(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = rgb.clip(0, 255).astype(np.uint8)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {xyz.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property float nx\nproperty float ny\nproperty float nz\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(xyz, rgb):
            handle.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} 0 0 0 "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def sorted_numbered_names(names: Sequence[str], prefix: str) -> List[str]:
    return sorted([name for name in names if name.startswith(prefix)], key=lambda name: int(name.split("_")[-1]))


def qvec_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    w1, x1, y1, z1 = np.moveaxis(left, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(right, -1, 0)
    out = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.maximum(norm, 1e-12)


def read_gaussian_vertices(path: Path) -> np.ndarray:
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    return ply["vertex"].data.copy()


def write_gaussian_vertices(path: Path, vertices: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement

    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))


def transform_gaussian_vertices(
    vertices: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    transformed = vertices.copy()
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(float)
    xyz = apply_similarity(xyz, scale, rotation, translation)
    transformed["x"] = xyz[:, 0]
    transformed["y"] = xyz[:, 1]
    transformed["z"] = xyz[:, 2]

    names = vertices.dtype.names or ()
    log_scale = float(np.log(scale))
    for name in sorted_numbered_names(names, "scale_"):
        transformed[name] = np.asarray(vertices[name], dtype=float) + log_scale

    rot_names = sorted_numbered_names(names, "rot_")
    if len(rot_names) == 4:
        colmap = load_colmap_helpers()
        q_transform = colmap.rotmat2qvec(np.asarray(rotation, dtype=float))
        q_local = np.stack([vertices[name] for name in rot_names], axis=1).astype(float)
        q_local = q_local / np.maximum(np.linalg.norm(q_local, axis=1, keepdims=True), 1e-12)
        q_global = qvec_multiply(q_transform[None, :], q_local)
        q_global[q_global[:, 0] < 0] *= -1.0
        for channel, name in enumerate(rot_names):
            transformed[name] = q_global[:, channel]

    return transformed


def gaussian_log_scale_max(vertices: np.ndarray) -> np.ndarray:
    names = vertices.dtype.names or ()
    scale_names = sorted_numbered_names(names, "scale_")
    if not scale_names:
        return np.zeros(vertices.shape[0], dtype=float)
    scales = np.stack([vertices[name] for name in scale_names], axis=1).astype(float)
    return np.max(scales, axis=1)


def gaussian_vertex_scores(
    vertices: np.ndarray,
    score_mode: str = "opacity",
    scale_penalty: float = 1.0,
) -> np.ndarray:
    names = vertices.dtype.names or ()
    if "opacity" in names:
        scores = np.asarray(vertices["opacity"], dtype=float)
    else:
        scores = np.ones(vertices.shape[0], dtype=float)
    if score_mode == "opacity":
        return scores
    if score_mode == "opacity_scale_penalty":
        return scores - float(scale_penalty) * gaussian_log_scale_max(vertices)
    raise ValueError(f"Unknown Gaussian score mode: {score_mode}")


def filter_gaussian_scale_quantile(vertices: np.ndarray, max_quantile: float) -> np.ndarray:
    if vertices.shape[0] == 0 or max_quantile >= 1.0:
        return vertices
    if max_quantile <= 0.0:
        raise ValueError("--gaussian_scale_max_quantile must be > 0")
    log_scale = gaussian_log_scale_max(vertices)
    if not np.any(np.isfinite(log_scale)):
        return vertices
    threshold = float(np.quantile(log_scale, max_quantile))
    keep = log_scale <= threshold
    return vertices[keep]


def voxel_dedupe_gaussians(
    vertices: np.ndarray,
    voxel_size: float,
    score_mode: str,
    scale_penalty: float,
) -> np.ndarray:
    if voxel_size <= 0 or vertices.shape[0] == 0:
        return vertices
    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(float)
    keys = np.floor(points / voxel_size).astype(np.int64)
    order = np.argsort(-gaussian_vertex_scores(vertices, score_mode, scale_penalty))
    seen = set()
    keep = []
    for idx in order:
        key = tuple(keys[idx])
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)
    keep = np.asarray(sorted(keep), dtype=np.int64)
    return vertices[keep]


def prune_gaussian_count(
    vertices: np.ndarray,
    max_count: int,
    score_mode: str,
    scale_penalty: float,
) -> np.ndarray:
    if max_count <= 0 or vertices.shape[0] <= max_count:
        return vertices
    order = np.argsort(-gaussian_vertex_scores(vertices, score_mode, scale_penalty))[:max_count]
    return vertices[np.asarray(sorted(order), dtype=np.int64)]


def gaussian_xyz(vertices: np.ndarray) -> np.ndarray:
    return np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(float)


def camera_params(camera: CameraRecord) -> Tuple[float, float, float, float]:
    params = np.asarray(camera.params, dtype=float)
    if camera.model == "SIMPLE_PINHOLE":
        return float(params[0]), float(params[0]), float(params[1]), float(params[2])
    if camera.model == "PINHOLE":
        return float(params[0]), float(params[1]), float(params[2]), float(params[3])
    if len(params) >= 4:
        return float(params[0]), float(params[1]), float(params[2]), float(params[3])
    raise ValueError(f"Unsupported camera model for projection: {camera.model} {camera.params}")


def project_to_visibility_grid(
    points: np.ndarray,
    image_record: ImageRecord,
    camera: CameraRecord,
    cell_size: int,
    near: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    w2c = image_w2c(image_record)
    cam_points = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam_points[:, 2]
    fx, fy, cx, cy = camera_params(camera)
    valid_z = z > near
    u = np.zeros_like(z)
    v = np.zeros_like(z)
    u[valid_z] = fx * cam_points[valid_z, 0] / z[valid_z] + cx
    v[valid_z] = fy * cam_points[valid_z, 1] / z[valid_z] + cy
    valid = (
        valid_z
        & (u >= 0.0)
        & (u < camera.width)
        & (v >= 0.0)
        & (v < camera.height)
    )
    grid_w = int(np.ceil(camera.width / cell_size))
    grid_x = np.floor(u[valid] / cell_size).astype(np.int64)
    grid_y = np.floor(v[valid] / cell_size).astype(np.int64)
    grid_key = grid_y * grid_w + grid_x
    valid_indices = np.flatnonzero(valid)
    return valid_indices, grid_key, z[valid]


def mark_visibility_duplicates(
    main_vertices: np.ndarray,
    sub_vertices: np.ndarray,
    shared_names: Sequence[str],
    final_records_by_name: Dict[str, ImageRecord],
    camera: CameraRecord,
    cell_size: int,
    depth_margin: float,
    min_shared: int,
) -> Tuple[np.ndarray, Dict]:
    duplicate_hits = np.zeros(sub_vertices.shape[0], dtype=np.int16)
    visible_hits = np.zeros(sub_vertices.shape[0], dtype=np.int16)
    shared_used = []
    if main_vertices.shape[0] == 0 or sub_vertices.shape[0] == 0:
        return np.zeros(sub_vertices.shape[0], dtype=bool), {
            "shared_used": shared_used,
            "visible_hits": 0,
            "duplicate_hits": 0,
        }

    main_xyz = gaussian_xyz(main_vertices)
    sub_xyz = gaussian_xyz(sub_vertices)
    grid_w = int(np.ceil(camera.width / cell_size))
    grid_h = int(np.ceil(camera.height / cell_size))
    grid_size = grid_w * grid_h

    for name in shared_names:
        image_record = final_records_by_name.get(name)
        if image_record is None:
            continue
        main_idx, main_key, main_depth = project_to_visibility_grid(main_xyz, image_record, camera, cell_size)
        if main_idx.size == 0:
            continue
        depth_buffer = np.full(grid_size, np.inf, dtype=np.float32)
        np.minimum.at(depth_buffer, main_key, main_depth.astype(np.float32))

        sub_idx, sub_key, sub_depth = project_to_visibility_grid(sub_xyz, image_record, camera, cell_size)
        if sub_idx.size == 0:
            continue
        main_nearest = depth_buffer[sub_key]
        covered = np.isfinite(main_nearest)
        if not np.any(covered):
            continue
        # If sub is not meaningfully in front of main at the same image cell, treat it as already explained.
        duplicate = covered & (sub_depth >= (main_nearest.astype(float) - depth_margin))
        visible_hits[sub_idx] += covered.astype(np.int16)
        duplicate_hits[sub_idx] += duplicate.astype(np.int16)
        shared_used.append(name)

    threshold = max(1, int(min_shared))
    duplicate_mask = duplicate_hits >= threshold
    return duplicate_mask, {
        "shared_used": shared_used,
        "visible_hits": int(np.count_nonzero(visible_hits)),
        "duplicate_hits": int(np.count_nonzero(duplicate_mask)),
        "min_shared": threshold,
        "cell_size": int(cell_size),
        "depth_margin": float(depth_margin),
    }


def window_gaussian_ply_path(record: WindowRecord, args: argparse.Namespace) -> Path:
    return (
        Path(record.gaussian_model_path)
        / "point_cloud"
        / f"iteration_{args.window_gaussian_iterations}"
        / "point_cloud.ply"
    )


def select_gaussian_refine_points(vertices: np.ndarray, max_points: int, score_mode: str, scale_penalty: float) -> np.ndarray:
    xyz = gaussian_xyz(vertices)
    if max_points <= 0 or xyz.shape[0] <= max_points:
        return xyz
    scores = gaussian_vertex_scores(vertices, score_mode, scale_penalty)
    indices = np.argsort(-scores)[:max_points]
    return xyz[np.asarray(sorted(indices), dtype=np.int64)]


def cap_point_pool(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    indices = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
    return points[indices]


def torch_rotvec_to_matrix(rotvec):
    import torch

    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
    x, y, z = rotvec
    zeros = torch.zeros((), device=rotvec.device, dtype=rotvec.dtype)
    k = torch.stack(
        [
            torch.stack([zeros, -z, y]),
            torch.stack([z, zeros, -x]),
            torch.stack([-y, x, zeros]),
        ]
    )
    theta2 = torch.dot(rotvec, rotvec)
    theta = torch.sqrt(torch.clamp(theta2, min=1e-12))
    theta4 = theta2 * theta2
    small = theta2 < 1e-8
    a_taylor = 1.0 - theta2 / 6.0 + theta4 / 120.0
    b_taylor = 0.5 - theta2 / 24.0 + theta4 / 720.0
    a = torch.where(small, a_taylor, torch.sin(theta) / theta)
    b = torch.where(small, b_taylor, (1.0 - torch.cos(theta)) / torch.clamp(theta2, min=1e-12))
    return eye + a * k + b * (k @ k)


def refine_similarity_nn_overlap(
    args: argparse.Namespace,
    main_points_global: np.ndarray,
    sub_points_local: np.ndarray,
    transform: Dict,
    scene_diag: float,
) -> Tuple[Dict, Dict]:
    info = {"enabled": True, "accepted": False}
    if main_points_global.shape[0] < args.gaussian_refine_min_overlap or sub_points_local.shape[0] < args.gaussian_refine_min_overlap:
        info["reason"] = "not enough main/sub points"
        return transform, info

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    radius = max(float(scene_diag) * args.gaussian_refine_nn_radius_ratio, 1e-6)
    base_scale = float(transform["scale"])
    base_rotation = np.asarray(transform["rotation"], dtype=float)
    base_translation = np.asarray(transform["translation"], dtype=float)
    sub_init_np = apply_similarity(sub_points_local, base_scale, base_rotation, base_translation)

    main = torch.as_tensor(main_points_global, dtype=torch.float32, device=device)
    sub_init = torch.as_tensor(sub_init_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        initial_dist = torch.cdist(sub_init, main).min(dim=1).values
        overlap_mask = initial_dist < radius
        overlap_indices = torch.nonzero(overlap_mask, as_tuple=False).flatten()
        if overlap_indices.numel() > args.gaussian_refine_max_overlap_points:
            pick = torch.linspace(
                0,
                overlap_indices.numel() - 1,
                args.gaussian_refine_max_overlap_points,
                device=device,
            ).long()
            overlap_indices = overlap_indices[pick]
        initial_overlap_rmse = float(torch.sqrt(torch.mean(initial_dist[overlap_indices] ** 2)).detach().cpu()) if overlap_indices.numel() else float("inf")

    info["overlap_points"] = int(overlap_indices.numel())
    info["radius"] = float(radius)
    info["initial_overlap_rmse"] = initial_overlap_rmse
    if overlap_indices.numel() < args.gaussian_refine_min_overlap:
        info["reason"] = f"not enough overlap points: {overlap_indices.numel()} < {args.gaussian_refine_min_overlap}"
        return transform, info

    sub_overlap = sub_init[overlap_indices].detach()
    log_scale = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
    rotvec = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=bool(args.gaussian_refine_optimize_rotation))
    translation = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)
    params = [log_scale, translation]
    if args.gaussian_refine_optimize_rotation:
        params.append(rotvec)
    optimizer = torch.optim.Adam(params, lr=args.gaussian_refine_lr)

    best = None
    for _ in range(args.gaussian_refine_iters):
        optimizer.zero_grad(set_to_none=True)
        rotation = torch_rotvec_to_matrix(rotvec)
        scale = torch.exp(log_scale)
        moved = scale * (sub_overlap @ rotation.T) + translation
        dists = torch.cdist(moved, main).min(dim=1).values
        data_loss = torch.mean(torch.sqrt(dists * dists + 1e-8))
        reg = (
            args.gaussian_refine_scale_reg * log_scale.square()
            + args.gaussian_refine_trans_reg * translation.square().sum() / (radius * radius)
        )
        if args.gaussian_refine_optimize_rotation:
            reg = reg + args.gaussian_refine_rot_reg * rotvec.square().sum()
        loss = data_loss + reg
        loss.backward()
        optimizer.step()
        current = float(data_loss.detach().cpu())
        if best is None or current < best["rmse"]:
            best = {
                "rmse": current,
                "log_scale": float(log_scale.detach().cpu()),
                "rotvec": rotvec.detach().cpu().numpy().astype(float),
                "translation": translation.detach().cpu().numpy().astype(float),
            }

    if best is None:
        info["reason"] = "optimizer produced no state"
        return transform, info

    delta_scale = float(np.exp(best["log_scale"]))
    delta_rotation = axis_angle_to_matrix_np(best["rotvec"])
    delta_translation = best["translation"]
    delta_rotation_deg = rotation_angle_deg(delta_rotation)
    delta_translation_norm = float(np.linalg.norm(delta_translation))
    improvement = initial_overlap_rmse - float(best["rmse"])
    info.update(
        {
            "final_overlap_rmse": float(best["rmse"]),
            "improvement": float(improvement),
            "delta_scale": delta_scale,
            "delta_rotation_deg": delta_rotation_deg,
            "delta_translation_norm": delta_translation_norm,
            "rotation_optimized": bool(args.gaussian_refine_optimize_rotation),
        }
    )
    guard_reasons = []
    if improvement <= max(initial_overlap_rmse * args.gaussian_refine_min_improvement_ratio, 0.0):
        guard_reasons.append("insufficient overlap improvement")
    if abs(np.log(delta_scale)) > args.gaussian_refine_max_log_scale:
        guard_reasons.append("scale delta too large")
    if delta_rotation_deg > args.gaussian_refine_max_rotation_deg:
        guard_reasons.append("rotation delta too large")
    if delta_translation_norm > float(scene_diag) * args.gaussian_refine_max_translation_ratio:
        guard_reasons.append("translation delta too large")
    if guard_reasons:
        info["reason"] = "; ".join(guard_reasons)
        return transform, info

    info["accepted"] = True
    info["reason"] = "accepted"
    return compose_delta_similarity(transform, delta_scale, delta_rotation, delta_translation), info


def axis_angle_to_matrix_np(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3, dtype=float)
    axis = np.asarray(rotvec, dtype=float) / theta
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    return np.eye(3, dtype=float) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def refresh_window_transformed_records(record: WindowRecord) -> None:
    local_records = read_images_text(Path(record.sparse_path) / "images.txt")
    transform = record.transform
    transformed_records = {}
    for image_path in record.image_paths:
        name = Path(image_path).name
        if name not in local_records:
            continue
        w2c = transform_w2c(
            image_w2c(local_records[name]),
            float(transform["scale"]),
            np.asarray(transform["rotation"], dtype=float),
            np.asarray(transform["translation"], dtype=float),
        )
        colmap = load_colmap_helpers()
        transformed_records[name] = {
            "qvec": colmap.rotmat2qvec(w2c[:3, :3]).tolist(),
            "tvec": w2c[:3, 3].tolist(),
        }
    record.alignment["transformed_records"] = transformed_records


def estimate_scene_diag_from_windows(window_records: Sequence[WindowRecord]) -> float:
    centers = []
    for record in window_records:
        for payload in record.alignment.get("transformed_records", {}).values():
            image_record = ImageRecord(
                image_id=-1,
                qvec=np.asarray(payload["qvec"], dtype=float),
                tvec=np.asarray(payload["tvec"], dtype=float),
                camera_id=-1,
                name="",
            )
            centers.append(camera_center(image_record))
    if len(centers) < 2:
        return 1.0
    return max(point_set_diag(np.stack(centers)), 1e-6)


def refine_gaussian_submap_transforms(args: argparse.Namespace, window_records: Sequence[WindowRecord], output_dir: Path) -> None:
    if args.gaussian_refine_mode == "none":
        return
    if args.global_init_mode != "gaussian_submaps":
        raise RuntimeError("Gaussian transform refinement requires --global_init_mode gaussian_submaps")

    scene_diag = estimate_scene_diag_from_windows(window_records)
    main_points = np.empty((0, 3), dtype=float)
    report = []
    for record in window_records:
        gaussian_path = window_gaussian_ply_path(record, args)
        vertices = read_gaussian_vertices(gaussian_path)
        local_points = select_gaussian_refine_points(
            vertices,
            args.gaussian_refine_sample_points,
            args.gaussian_score_mode,
            args.gaussian_scale_penalty,
        )
        refine_info = {"enabled": False, "accepted": False, "reason": "reference or disabled"}
        if record.index > 0 and args.gaussian_refine_mode == "nn_overlap":
            new_transform, refine_info = refine_similarity_nn_overlap(
                args,
                main_points,
                local_points,
                record.transform,
                scene_diag,
            )
            if refine_info.get("accepted"):
                record.transform = new_transform
                refresh_window_transformed_records(record)
        elif args.gaussian_refine_mode != "nn_overlap":
            raise ValueError(f"Unknown Gaussian refine mode: {args.gaussian_refine_mode}")

        transformed_points = apply_similarity(
            local_points,
            float(record.transform["scale"]),
            np.asarray(record.transform["rotation"], dtype=float),
            np.asarray(record.transform["translation"], dtype=float),
        )
        main_points = cap_point_pool(np.concatenate([main_points, transformed_points], axis=0), args.gaussian_refine_main_points)
        record.alignment["gaussian_refine"] = refine_info
        report.append({"window": record.index, **refine_info})
        print(
            f"[gaussian-refine] window {record.index:03d}: accepted={refine_info.get('accepted')} "
            f"overlap={refine_info.get('overlap_points', 0)} "
            f"improve={refine_info.get('improvement', 0):.6f} "
            f"reason={refine_info.get('reason')}"
        )

    (output_dir / "gaussian_refine_report.json").write_text(
        json.dumps(report, indent=2, default=lambda value: value.tolist() if hasattr(value, "tolist") else value),
        encoding="utf-8",
    )


def load_confidence(sparse_path: Path, n_points: int) -> np.ndarray:
    for name in ("confidence_dsp.npy", "confidence.npy"):
        path = sparse_path / name
        if path.exists():
            conf = np.load(path).reshape(-1)
            if conf.shape[0] == n_points:
                return conf.astype(float)
    return np.ones(n_points, dtype=float)


def load_keep_mask(mask_dir: Path, image_name: str, shape: Tuple[int, int]) -> np.ndarray:
    path = mask_dir / image_name
    if not path.exists():
        return np.ones(shape, dtype=bool)
    try:
        from PIL import Image

        mask = np.asarray(Image.open(path).convert("L")) > 127
        if mask.shape == shape:
            return mask
    except Exception as exc:
        print(f"[warn] failed to read keep mask {path}: {exc}")
    return np.ones(shape, dtype=bool)


def load_owner_view_points(
    sparse_path: Path,
    window: WindowRecord,
    pose_sources: Dict[str, Dict],
    use_window_masks: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_path = sparse_path / "points3D_all.npy"
    colors_path = sparse_path / "pointsColor_all.npy"
    confidence_path = sparse_path / "confidence.npy"
    if not points_path.exists() or not colors_path.exists() or not confidence_path.exists():
        raise FileNotFoundError(
            f"owner_views merge requires points3D_all.npy, pointsColor_all.npy, and confidence.npy in {sparse_path}"
        )

    points_all = np.load(points_path)
    colors_all = np.load(colors_path)
    confidence_all = np.load(confidence_path)
    if points_all.ndim != 4 or points_all.shape[-1] != 3:
        raise ValueError(f"Unexpected points3D_all shape: {points_all.shape}")
    if colors_all.shape != points_all.shape:
        raise ValueError(f"points/colors shape mismatch: {points_all.shape} vs {colors_all.shape}")

    n_views, height, width, _ = points_all.shape
    confidence_all = confidence_all.reshape(n_views, height, width)
    local_names = [Path(path).name for path in window.image_paths]
    selected_indices = [
        idx for idx, name in enumerate(local_names)
        if pose_sources.get(name, {}).get("window") == window.index
    ]
    if not selected_indices:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.uint8), np.empty((0,), dtype=float)

    mask_dir = sparse_path / f"overlapping_masks_{n_views}"
    points = []
    colors = []
    confidence = []
    for idx in selected_indices:
        mask = load_keep_mask(mask_dir, local_names[idx], (height, width)) if use_window_masks else np.ones((height, width), dtype=bool)
        points.append(points_all[idx][mask].reshape(-1, 3).astype(float))
        colors.append((colors_all[idx][mask].reshape(-1, 3).clip(0.0, 1.0) * 255.0).astype(np.uint8))
        confidence.append(confidence_all[idx][mask].reshape(-1).astype(float))

    return np.concatenate(points, axis=0), np.concatenate(colors, axis=0), np.concatenate(confidence, axis=0)


def load_dense_pointmaps(sparse_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    points_path = sparse_path / "points3D_all.npy"
    confidence_path = sparse_path / "confidence.npy"
    if not points_path.exists() or not confidence_path.exists():
        raise FileNotFoundError(f"Missing dense pointmap files in {sparse_path}")
    points = np.load(points_path).astype(float)
    confidence = np.load(confidence_path).astype(float)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"Unexpected points3D_all shape: {points.shape}")
    n_views, height, width, _ = points.shape
    confidence = confidence.reshape(n_views, height, width)
    return points, confidence


def collect_shared_dense_pairs(
    args: argparse.Namespace,
    shared_names: Sequence[str],
    current_sparse: Path,
    current_image_paths: Sequence[Path],
    previous_window: WindowRecord,
) -> Tuple[np.ndarray, np.ndarray]:
    current_points, current_conf = load_dense_pointmaps(current_sparse)
    previous_sparse = Path(previous_window.sparse_path)
    previous_points, previous_conf = load_dense_pointmaps(previous_sparse)
    current_names = [path.name for path in current_image_paths]
    previous_names = [Path(path).name for path in previous_window.image_paths]
    current_index = {name: idx for idx, name in enumerate(current_names)}
    previous_index = {name: idx for idx, name in enumerate(previous_names)}

    src_chunks = []
    dst_chunks = []
    for name in shared_names:
        if name not in current_index or name not in previous_index:
            continue
        cur_idx = current_index[name]
        prev_idx = previous_index[name]
        cur_pts = current_points[cur_idx]
        prev_pts = previous_points[prev_idx]
        cur_conf = current_conf[cur_idx]
        prev_conf = previous_conf[prev_idx]
        height, width = cur_conf.shape
        valid = np.isfinite(cur_pts).all(axis=-1) & np.isfinite(prev_pts).all(axis=-1)
        valid &= np.linalg.norm(cur_pts, axis=-1) > 1e-8
        valid &= np.linalg.norm(prev_pts, axis=-1) > 1e-8
        if args.dense_refine_use_masks:
            cur_mask = load_keep_mask(current_sparse / f"overlapping_masks_{current_points.shape[0]}", name, (height, width))
            prev_mask = load_keep_mask(previous_sparse / f"overlapping_masks_{previous_points.shape[0]}", name, (height, width))
            valid &= cur_mask & prev_mask
        if not np.any(valid):
            continue
        pair_conf = np.minimum(cur_conf, prev_conf)
        conf_values = pair_conf[valid]
        if conf_values.size == 0:
            continue
        threshold = float(np.quantile(conf_values, args.dense_refine_conf_quantile))
        valid &= pair_conf >= threshold
        if not np.any(valid):
            continue
        src_chunks.append(cur_pts[valid].reshape(-1, 3))
        dst_chunks.append(prev_pts[valid].reshape(-1, 3))

    if not src_chunks:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)

    src = np.concatenate(src_chunks, axis=0)
    dst = np.concatenate(dst_chunks, axis=0)
    if src.shape[0] > args.dense_refine_max_points:
        rng = np.random.default_rng(args.dense_refine_seed)
        indices = rng.choice(src.shape[0], args.dense_refine_max_points, replace=False)
        src = src[indices]
        dst = dst[indices]
    return src, dst


def dense_refine_alignment(
    args: argparse.Namespace,
    shared_names: Sequence[str],
    current_sparse: Path,
    current_image_paths: Sequence[Path],
    previous_window: WindowRecord,
    transform: Dict,
) -> Tuple[Dict, Dict]:
    info = {"enabled": True, "accepted": False}
    src, previous_local = collect_shared_dense_pairs(
        args,
        shared_names,
        current_sparse,
        current_image_paths,
        previous_window,
    )
    info["num_pairs"] = int(src.shape[0])
    if src.shape[0] < args.dense_refine_min_points:
        info["reason"] = f"not enough dense pairs: {src.shape[0]} < {args.dense_refine_min_points}"
        return transform, info

    previous_transform = previous_window.transform
    dst = apply_similarity(
        previous_local,
        float(previous_transform["scale"]),
        np.asarray(previous_transform["rotation"], dtype=float),
        np.asarray(previous_transform["translation"], dtype=float),
    )
    rotation = np.asarray(transform["rotation"], dtype=float)
    before = apply_similarity(
        src,
        float(transform["scale"]),
        rotation,
        np.asarray(transform["translation"], dtype=float),
    )
    before_errors = np.linalg.norm(before - dst, axis=1)
    before_rmse = float(np.sqrt(np.mean(before_errors**2)))
    refined_scale, refined_translation = fixed_rotation_similarity(src, dst, rotation)
    after = apply_similarity(src, refined_scale, rotation, refined_translation)
    after_errors = np.linalg.norm(after - dst, axis=1)
    after_rmse = float(np.sqrt(np.mean(after_errors**2)))
    info.update(
        {
            "scale_before": float(transform["scale"]),
            "scale_after": float(refined_scale),
            "translation_norm_delta": float(
                np.linalg.norm(refined_translation - np.asarray(transform["translation"], dtype=float))
            ),
            "dense_rmse_before": before_rmse,
            "dense_rmse_after": after_rmse,
            "dense_median_before": float(np.median(before_errors)),
            "dense_median_after": float(np.median(after_errors)),
        }
    )
    if refined_scale < args.min_scale or refined_scale > args.max_scale:
        info["reason"] = f"refined scale {refined_scale:.6f} outside [{args.min_scale}, {args.max_scale}]"
        return transform, info
    if after_rmse > before_rmse * args.dense_refine_accept_ratio:
        info["reason"] = (
            f"dense rmse did not improve enough: {after_rmse:.6f} > "
            f"{before_rmse * args.dense_refine_accept_ratio:.6f}"
        )
        return transform, info

    info["accepted"] = True
    info["reason"] = "accepted"
    return {"scale": refined_scale, "rotation": rotation, "translation": refined_translation}, info


def voxel_dedupe(points: np.ndarray, colors: np.ndarray, confidence: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if voxel_size <= 0 or points.shape[0] == 0:
        return points, colors, confidence
    keys = np.floor(points / voxel_size).astype(np.int64)
    order = np.argsort(-confidence)
    seen = set()
    keep = []
    for idx in order:
        key = tuple(keys[idx])
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)
    keep = np.asarray(sorted(keep), dtype=np.int64)
    return points[keep], colors[keep], confidence[keep]


def write_cameras_text(path: Path, cameras: Sequence[CameraRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        handle.write(f"# Number of cameras: {len(cameras)}\n")
        for camera in cameras:
            params = " ".join(f"{float(v):.12g}" for v in camera.params)
            handle.write(f"{camera.camera_id} {camera.model} {camera.width} {camera.height} {params}\n")


def write_images_text(path: Path, records: Sequence[ImageRecord]) -> None:
    colmap = load_colmap_helpers()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        handle.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        handle.write(f"# Number of images: {len(records)}, mean observations per image: 0\n")
        for record in records:
            qvec = np.asarray(record.qvec, dtype=float)
            qvec = qvec / max(np.linalg.norm(qvec), 1e-12)
            line = [
                str(record.image_id),
                *(f"{float(v):.17g}" for v in qvec),
                *(f"{float(v):.17g}" for v in record.tvec),
                str(record.camera_id),
                record.name,
            ]
            handle.write(" ".join(line) + "\n\n")


def median_camera(cameras: Sequence[CameraRecord], camera_id: int) -> CameraRecord:
    if not cameras:
        raise ValueError("No cameras available")
    widths = [camera.width for camera in cameras]
    heights = [camera.height for camera in cameras]
    params = np.stack([camera.params for camera in cameras])
    width = int(round(float(np.median(widths))))
    height = int(round(float(np.median(heights))))
    model = cameras[0].model
    return CameraRecord(camera_id, model, width, height, np.median(params, axis=0))


def run_command(command: Sequence[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("PYTHONUNBUFFERED", "1")
    print("$ " + " ".join(str(part) for part in command), flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(str(part) for part in command) + "\n\n")
        log_file.flush()
        process = subprocess.Popen(
            list(map(str, command)),
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return process.wait()


def append_train_lr_overrides(command: List[str], args: argparse.Namespace, prefix: str) -> None:
    mappings = [
        (f"{prefix}_position_lr_init", "--position_lr_init"),
        (f"{prefix}_position_lr_final", "--position_lr_final"),
        (f"{prefix}_scaling_lr", "--scaling_lr"),
        (f"{prefix}_rotation_lr", "--rotation_lr"),
    ]
    for attr, flag in mappings:
        value = float(getattr(args, attr, 0.0))
        if value > 0:
            command.extend([flag, f"{value:g}"])


def run_window_init(
    instant_splat_root: Path,
    python_bin: str,
    scene_path: Path,
    model_path: Path,
    n_views: int,
    scene_graph: str,
    skip_existing: bool,
) -> None:
    sparse_path = scene_path / f"sparse_{n_views}" / "0"
    if skip_existing and (sparse_path / "images.txt").exists() and (sparse_path / "points3D.ply").exists():
        print(f"[skip] existing window init: {scene_path.name}")
        return
    command = [
        python_bin,
        "init_geo.py",
        "--source_path",
        str(scene_path),
        "--model_path",
        str(model_path),
        "--n_views",
        str(n_views),
        "--focal_avg",
        "--co_vis_dsp",
        "--conf_aware_ranking",
        "--infer_video",
        "--scene_graph",
        scene_graph,
    ]
    code = run_command(command, instant_splat_root, model_path / "logs" / "init.log")
    if code != 0:
        raise RuntimeError(f"Window init failed for {scene_path}: exit {code}")


def run_window_gaussian_train(
    instant_splat_root: Path,
    args: argparse.Namespace,
    scene_path: Path,
    model_path: Path,
    n_views: int,
) -> Path:
    iteration = int(args.window_gaussian_iterations)
    if iteration <= 0:
        raise ValueError("--window_gaussian_iterations must be positive for gaussian_submaps mode")
    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if args.skip_existing and ply_path.exists():
        print(f"[skip] existing window Gaussian train: {model_path.name}")
        return ply_path

    pose_freeze = args.window_gaussian_pose_freeze_iters
    if pose_freeze < 0:
        pose_freeze = iteration
    command = [
        args.python,
        "train.py",
        "-s",
        str(scene_path),
        "-m",
        str(model_path),
        "--n_views",
        str(n_views),
        "--iterations",
        str(iteration),
        "--test_iterations",
        str(iteration),
        "--pp_optimizer",
        "--optim_pose",
        "--pose_freeze_iters",
        str(pose_freeze),
        "--pose_lr_scale",
        str(args.window_gaussian_pose_lr_scale),
    ]
    append_train_lr_overrides(command, args, "window")
    code = run_command(command, instant_splat_root, model_path / "logs" / "train.log")
    if code != 0:
        raise RuntimeError(f"Window Gaussian train failed for {scene_path}: exit {code}")
    if not ply_path.exists():
        raise FileNotFoundError(f"Window Gaussian train did not produce {ply_path}")
    return ply_path


def pose_candidate_score(name: str, window_start: int, window_len: int, sampled_name_to_index: Dict[str, int]) -> float:
    sampled_index = sampled_name_to_index[name]
    window_center = window_start + (window_len - 1) / 2.0
    return abs(sampled_index - window_center)


def update_best_records(
    global_records: Dict[str, ImageRecord],
    global_record_scores: Dict[str, float],
    transformed: Dict[str, ImageRecord],
    window_start: int,
    window_len: int,
    sampled_name_to_index: Dict[str, int],
) -> None:
    for name, rec in transformed.items():
        score = pose_candidate_score(name, window_start, window_len, sampled_name_to_index)
        if name not in global_records or score < global_record_scores[name]:
            global_records[name] = rec
            global_record_scores[name] = score


def transformed_payload_to_records(payload: Dict[str, Dict]) -> Dict[str, ImageRecord]:
    return {
        name: ImageRecord(
            image_id=-1,
            qvec=np.asarray(record["qvec"], dtype=float),
            tvec=np.asarray(record["tvec"], dtype=float),
            camera_id=-1,
            name=name,
        )
        for name, record in payload.items()
    }


def compute_alignment_residuals(
    local_records: Dict[str, ImageRecord],
    global_image_records: Dict[str, ImageRecord],
    shared_names: Sequence[str],
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Dict:
    src_centers = np.stack([camera_center(local_records[name]) for name in shared_names])
    dst_centers = np.stack([camera_center(global_image_records[name]) for name in shared_names])
    transformed = apply_similarity(src_centers, scale, rotation, translation)
    translation_errors = np.linalg.norm(transformed - dst_centers, axis=1)
    translation_rmse = float(np.sqrt(np.mean(translation_errors**2)))
    shared_diag = point_set_diag(dst_centers)
    translation_ratio = float(translation_rmse / max(shared_diag, 1e-9))

    rot_errors = []
    shared_residuals = []
    for name, trans_error in zip(shared_names, translation_errors):
        local_c2w = np.linalg.inv(image_w2c(local_records[name]))
        global_c2w = np.linalg.inv(image_w2c(global_image_records[name]))
        predicted_rot = rotation @ local_c2w[:3, :3]
        rot_error = rotation_angle_deg(predicted_rot.T @ global_c2w[:3, :3])
        rot_errors.append(rot_error)
        shared_residuals.append(
            {
                "image": name,
                "translation_error": float(trans_error),
                "rotation_deg": float(rot_error),
            }
        )

    return {
        "translation_rmse": translation_rmse,
        "translation_max": float(np.max(translation_errors)),
        "translation_ratio": translation_ratio,
        "shared_path_diag": shared_diag,
        "rotation_mean_deg": float(np.mean(rot_errors)),
        "rotation_max_deg": float(np.max(rot_errors)),
        "shared_residuals": shared_residuals,
    }


def estimate_alignment_candidate(
    method: str,
    local_records: Dict[str, ImageRecord],
    global_image_records: Dict[str, ImageRecord],
    shared_names: Sequence[str],
) -> Tuple[Dict, Dict]:
    src_centers = np.stack([camera_center(local_records[name]) for name in shared_names])
    dst_centers = np.stack([camera_center(global_image_records[name]) for name in shared_names])

    if method == "center_umeyama":
        scale, rotation, translation = umeyama_similarity(src_centers, dst_centers)
    elif method == "rotation_center":
        relative_rotations = []
        for name in shared_names:
            local_c2w = np.linalg.inv(image_w2c(local_records[name]))
            global_c2w = np.linalg.inv(image_w2c(global_image_records[name]))
            relative_rotations.append(global_c2w[:3, :3] @ local_c2w[:3, :3].T)
        rotation = average_rotations(np.stack(relative_rotations))
        scale, translation = fixed_rotation_similarity(src_centers, dst_centers, rotation)
    else:
        raise ValueError(f"Unknown alignment method: {method}")

    transform = {"scale": scale, "rotation": rotation, "translation": translation}
    residuals = compute_alignment_residuals(local_records, global_image_records, shared_names, scale, rotation, translation)
    alignment = {
        "method": method,
        "shared_images": list(shared_names),
        "scale": scale,
        **residuals,
    }
    return transform, alignment


def choose_alignment_candidate(candidates: Sequence[Tuple[Dict, Dict]], args: argparse.Namespace) -> Tuple[Dict, Dict]:
    if not candidates:
        raise RuntimeError("No valid alignment candidates were produced")

    def score(candidate: Tuple[Dict, Dict]) -> Tuple[float, float, float, float]:
        _, alignment = candidate
        scale = float(alignment["scale"])
        scale_penalty = 1.0 if scale < args.min_scale or scale > args.max_scale else 0.0
        return (
            scale_penalty,
            float(alignment["rotation_mean_deg"]),
            float(alignment["translation_ratio"]),
            float(alignment["translation_rmse"]),
        )

    transform, alignment = min(candidates, key=score)
    summary = {}
    for _, candidate_alignment in candidates:
        summary[candidate_alignment["method"]] = {
            "scale": float(candidate_alignment["scale"]),
            "translation_rmse": float(candidate_alignment["translation_rmse"]),
            "translation_ratio": float(candidate_alignment["translation_ratio"]),
            "rotation_mean_deg": float(candidate_alignment["rotation_mean_deg"]),
            "rotation_max_deg": float(candidate_alignment["rotation_max_deg"]),
        }
    alignment["candidate_summary"] = summary
    return transform, alignment


def mark_alignment_quality(args: argparse.Namespace, alignment: Dict) -> None:
    bad_reasons = []
    if alignment.get("status") != "reference":
        scale = float(alignment.get("scale", 0.0))
        rotation_mean = float(alignment.get("rotation_mean_deg", 0.0))
        translation_ratio = float(alignment.get("translation_ratio", 0.0))
        if scale < args.min_scale or scale > args.max_scale:
            bad_reasons.append(f"scale {scale:.6f} outside [{args.min_scale}, {args.max_scale}]")
        if rotation_mean > args.max_rotation_residual_deg:
            bad_reasons.append(
                f"rotation_mean_deg {rotation_mean:.4f} > {args.max_rotation_residual_deg:.4f}"
            )
        if translation_ratio > args.max_translation_residual_ratio:
            bad_reasons.append(
                f"translation_ratio {translation_ratio:.6f} > {args.max_translation_residual_ratio:.6f}"
            )
    alignment["is_bad"] = bool(bad_reasons)
    alignment["bad_reasons"] = bad_reasons


def build_alignment_for_window(
    args: argparse.Namespace,
    window_images: Sequence[Path],
    window_sparse: Path,
    global_image_records: Dict[str, ImageRecord],
    previous_transform: Optional[Dict],
    previous_window: Optional[WindowRecord],
) -> Tuple[Dict, Dict[str, ImageRecord]]:
    local_records = read_images_text(window_sparse / "images.txt")
    shared_names = sorted(set(local_records) & set(global_image_records), key=lambda name: natural_key(Path(name)))
    if previous_transform is None:
        transform = {"scale": 1.0, "rotation": np.eye(3), "translation": np.zeros(3)}
        alignment = {
            "status": "reference",
            "method": "reference",
            "shared_images": shared_names,
            "scale": 1.0,
            "translation_rmse": 0.0,
            "translation_max": 0.0,
            "translation_ratio": 0.0,
            "shared_path_diag": 0.0,
            "rotation_mean_deg": 0.0,
            "rotation_max_deg": 0.0,
            "shared_residuals": [],
        }
    else:
        if len(shared_names) < 3:
            raise RuntimeError(
                f"Need at least 3 shared frames for cross-window alignment, got {len(shared_names)}: {shared_names}"
            )
        if args.alignment_mode == "hybrid":
            methods = ("rotation_center", "center_umeyama")
        else:
            methods = (args.alignment_mode,)
        candidates = []
        candidate_errors = {}
        for method in methods:
            try:
                candidates.append(estimate_alignment_candidate(method, local_records, global_image_records, shared_names))
            except Exception as exc:
                candidate_errors[method] = str(exc)
        transform, alignment = choose_alignment_candidate(candidates, args)
        alignment["status"] = "aligned"
        if candidate_errors:
            alignment["candidate_errors"] = candidate_errors
        if args.dense_refine and previous_window is not None:
            refined_transform, dense_info = dense_refine_alignment(
                args,
                shared_names,
                window_sparse,
                window_images,
                previous_window,
                transform,
            )
            alignment["dense_refine"] = dense_info
            if dense_info.get("accepted"):
                refined_residuals = compute_alignment_residuals(
                    local_records,
                    global_image_records,
                    shared_names,
                    float(refined_transform["scale"]),
                    np.asarray(refined_transform["rotation"], dtype=float),
                    np.asarray(refined_transform["translation"], dtype=float),
                )
                pose_guard_reasons = []
                if refined_residuals["translation_ratio"] > args.max_translation_residual_ratio:
                    pose_guard_reasons.append(
                        f"translation_ratio {refined_residuals['translation_ratio']:.6f} "
                        f"> {args.max_translation_residual_ratio:.6f}"
                    )
                if refined_residuals["rotation_mean_deg"] > args.max_rotation_residual_deg:
                    pose_guard_reasons.append(
                        f"rotation_mean_deg {refined_residuals['rotation_mean_deg']:.4f} "
                        f"> {args.max_rotation_residual_deg:.4f}"
                    )
                if refined_residuals["translation_ratio"] > alignment["translation_ratio"] * args.dense_refine_pose_ratio:
                    pose_guard_reasons.append(
                        f"translation_ratio worsened too much: {refined_residuals['translation_ratio']:.6f} "
                        f"> {alignment['translation_ratio'] * args.dense_refine_pose_ratio:.6f}"
                    )
                dense_info["candidate_pose_residuals"] = {
                    "translation_rmse": float(refined_residuals["translation_rmse"]),
                    "translation_ratio": float(refined_residuals["translation_ratio"]),
                    "rotation_mean_deg": float(refined_residuals["rotation_mean_deg"]),
                    "rotation_max_deg": float(refined_residuals["rotation_max_deg"]),
                }
                if pose_guard_reasons:
                    dense_info["accepted"] = False
                    dense_info["reason"] = "rejected by pose guard"
                    dense_info["pose_guard_reasons"] = pose_guard_reasons
                else:
                    transform = refined_transform
                    for key, value in refined_residuals.items():
                        alignment[key] = value
                    alignment["scale"] = float(transform["scale"])
                    alignment["method"] = f"{alignment['method']}+dense"

    mark_alignment_quality(args, alignment)

    transformed_records = {}
    for image_path in window_images:
        name = image_path.name
        record = local_records[name]
        w2c = transform_w2c(
            image_w2c(record),
            float(transform["scale"]),
            np.asarray(transform["rotation"], dtype=float),
            np.asarray(transform["translation"], dtype=float),
        )
        colmap = load_colmap_helpers()
        transformed_records[name] = ImageRecord(
            image_id=-1,
            qvec=colmap.rotmat2qvec(w2c[:3, :3]),
            tvec=w2c[:3, 3],
            camera_id=-1,
            name=name,
        )
    return {"transform": transform, "alignment": alignment}, transformed_records


def merge_gaussian_submaps(
    args: argparse.Namespace,
    window_records: Sequence[WindowRecord],
    output_path: Path,
    fallback_voxel_size: float,
    final_records_by_name: Dict[str, ImageRecord],
    camera_template: CameraRecord,
    camera_path_diag: float,
) -> Dict:
    transformed_vertices: List[np.ndarray] = []
    per_window_counts = []
    for record in window_records:
        if not record.gaussian_model_path:
            raise RuntimeError(
                "Gaussian submap merge requested, but a window has no gaussian_model_path. "
                "Run with --global_init_mode gaussian_submaps from the start."
            )
        gaussian_path = (
            Path(record.gaussian_model_path)
            / "point_cloud"
            / f"iteration_{args.window_gaussian_iterations}"
            / "point_cloud.ply"
        )
        if not gaussian_path.exists():
            raise FileNotFoundError(f"Missing window Gaussian submap: {gaussian_path}")
        vertices = read_gaussian_vertices(gaussian_path)
        transform = record.transform
        vertices = transform_gaussian_vertices(
            vertices,
            float(transform["scale"]),
            np.asarray(transform["rotation"], dtype=float),
            np.asarray(transform["translation"], dtype=float),
        )
        count_after_transform = int(vertices.shape[0])
        vertices = filter_gaussian_scale_quantile(vertices, args.gaussian_scale_max_quantile)
        count_after_scale_filter = int(vertices.shape[0])
        vertices = prune_gaussian_count(
            vertices,
            args.gaussian_per_window_max_count,
            args.gaussian_score_mode,
            args.gaussian_scale_penalty,
        )
        transformed_vertices.append(vertices)
        per_window_counts.append(
            {
                "window": record.index,
                "path": str(gaussian_path),
                "count_before_window_prune": count_after_transform,
                "count_after_scale_filter": count_after_scale_filter,
                "count_after_window_prune": int(vertices.shape[0]),
            }
        )

    if not transformed_vertices:
        raise RuntimeError("No Gaussian submaps were available for global merge")

    voxel_size = args.gaussian_voxel_size if args.gaussian_voxel_size > 0 else fallback_voxel_size
    before = int(sum(vertices.shape[0] for vertices in transformed_vertices))
    visibility_report = {"enabled": False}
    if args.gaussian_fusion_mode == "incremental_visibility":
        depth_margin = max(camera_path_diag * args.gaussian_visibility_depth_ratio, 1e-6)
        main_vertices = transformed_vertices[0][:0]
        visibility_steps = []
        for record, vertices, stats in zip(window_records, transformed_vertices, per_window_counts):
            before_visibility = int(vertices.shape[0])
            if main_vertices.shape[0] == 0:
                kept = vertices
                visibility_info = {
                    "shared_used": [],
                    "visible_hits": 0,
                    "duplicate_hits": 0,
                    "min_shared": int(args.gaussian_visibility_min_shared),
                    "cell_size": int(args.gaussian_visibility_cell_size),
                    "depth_margin": float(depth_margin),
                }
            else:
                shared_names = record.alignment.get("shared_images", [])
                duplicate_mask, visibility_info = mark_visibility_duplicates(
                    main_vertices,
                    vertices,
                    shared_names,
                    final_records_by_name,
                    camera_template,
                    args.gaussian_visibility_cell_size,
                    depth_margin,
                    args.gaussian_visibility_min_shared,
                )
                kept = vertices[~duplicate_mask]
            main_vertices = np.concatenate([main_vertices, kept])
            stats["count_before_visibility_fusion"] = before_visibility
            stats["count_removed_by_visibility"] = int(before_visibility - kept.shape[0])
            stats["count_after_visibility_fusion"] = int(kept.shape[0])
            stats["visibility"] = visibility_info
            visibility_steps.append(
                {
                    "window": record.index,
                    "before": before_visibility,
                    "kept": int(kept.shape[0]),
                    "removed": int(before_visibility - kept.shape[0]),
                    **visibility_info,
                }
            )
        merged = main_vertices
        visibility_report = {
            "enabled": True,
            "steps": visibility_steps,
            "count_after_visibility": int(merged.shape[0]),
        }
    elif args.gaussian_fusion_mode == "batch_voxel":
        merged = np.concatenate(transformed_vertices)
    else:
        raise ValueError(f"Unknown Gaussian fusion mode: {args.gaussian_fusion_mode}")

    before_dedupe_after_fusion = int(merged.shape[0])
    merged = voxel_dedupe_gaussians(
        merged,
        voxel_size,
        args.gaussian_score_mode,
        args.gaussian_scale_penalty,
    )
    after_dedupe = int(merged.shape[0])
    merged = prune_gaussian_count(
        merged,
        args.gaussian_merge_max_count,
        args.gaussian_score_mode,
        args.gaussian_scale_penalty,
    )
    after_prune = int(merged.shape[0])
    write_gaussian_vertices(output_path, merged)
    return {
        "enabled": True,
        "path": str(output_path),
        "window_gaussian_iterations": int(args.window_gaussian_iterations),
        "fusion_mode": args.gaussian_fusion_mode,
        "voxel_size": float(voxel_size),
        "count_before_dedupe": before,
        "count_before_dedupe_after_fusion": before_dedupe_after_fusion,
        "count_after_dedupe": after_dedupe,
        "count_after_prune": after_prune,
        "max_count": int(args.gaussian_merge_max_count),
        "score_mode": args.gaussian_score_mode,
        "scale_penalty": float(args.gaussian_scale_penalty),
        "scale_max_quantile": float(args.gaussian_scale_max_quantile),
        "visibility_fusion": visibility_report,
        "per_window_counts": per_window_counts,
    }


def merge_global_scene(args: argparse.Namespace, sampled_images: Sequence[Path], window_records: List[WindowRecord]) -> Path:
    output_dir = Path(args.output_dir).resolve()
    workspace = output_dir / "longseq_workspace"
    global_scene = workspace / "global_scene"
    materialize_scene(global_scene, sampled_images, args.link_mode, args.overwrite_global)
    sparse_out = global_scene / f"sparse_{len(sampled_images)}" / "0"
    sparse_out.mkdir(parents=True, exist_ok=True)

    sampled_name_to_index = {image.name: idx for idx, image in enumerate(sampled_images)}
    final_records_by_name: Dict[str, ImageRecord] = {}
    final_record_scores: Dict[str, float] = {}
    pose_sources: Dict[str, Dict] = {}
    all_cameras: List[CameraRecord] = []
    all_points = []
    all_colors = []
    all_conf = []

    for record in window_records:
        sparse = Path(record.sparse_path)
        local_cameras = read_cameras_text(sparse / "cameras.txt")
        all_cameras.extend(local_cameras.values())
        transformed_records = record.alignment.get("transformed_records", {})
        for name, rec in transformed_records.items():
            if name not in sampled_name_to_index:
                continue
            score = pose_candidate_score(name, record.start, len(record.image_paths), sampled_name_to_index)
            if name not in final_records_by_name or score < final_record_scores[name]:
                final_records_by_name[name] = ImageRecord(
                    image_id=-1,
                    qvec=np.asarray(rec["qvec"], dtype=float),
                    tvec=np.asarray(rec["tvec"], dtype=float),
                    camera_id=-1,
                    name=name,
                )
                final_record_scores[name] = score
                pose_sources[name] = {"window": record.index, "score": score}

    missing_records = [image.name for image in sampled_images if image.name not in final_records_by_name]
    if missing_records:
        raise RuntimeError(f"Missing final camera poses for images: {missing_records}")

    for record in window_records:
        sparse = Path(record.sparse_path)
        transform = record.transform
        if args.point_merge_mode == "owner_views":
            points, colors, conf = load_owner_view_points(
                sparse,
                record,
                pose_sources,
                not args.no_window_masks,
            )
        else:
            points, colors = read_ply_points(sparse / "points3D.ply")
            conf = load_confidence(sparse, points.shape[0])
        if points.shape[0] == 0:
            continue
        points = apply_similarity(
            points,
            float(transform["scale"]),
            np.asarray(transform["rotation"], dtype=float),
            np.asarray(transform["translation"], dtype=float),
        )
        all_points.append(points)
        all_colors.append(colors)
        all_conf.append(conf)

    if not all_points:
        raise RuntimeError("No points were selected for global merge")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    confidence = np.concatenate(all_conf, axis=0)
    centers = np.stack([camera_center(final_records_by_name[image.name]) for image in sampled_images])
    diag = float(np.max(np.linalg.norm(centers - centers.mean(axis=0, keepdims=True), axis=1)) * 2.0)
    voxel_size = args.voxel_size if args.voxel_size > 0 else max(diag * args.voxel_diag_ratio, 1e-6)
    before = points.shape[0]
    points, colors, confidence = voxel_dedupe(points, colors, confidence, voxel_size)

    camera_template = median_camera(all_cameras, 1)
    cameras = []
    image_records = []
    for idx, image_path in enumerate(sampled_images, start=1):
        name = image_path.name
        base = final_records_by_name[name]
        cameras.append(CameraRecord(idx, camera_template.model, camera_template.width, camera_template.height, camera_template.params))
        image_records.append(ImageRecord(idx, base.qvec, base.tvec, idx, name))

    write_cameras_text(sparse_out / "cameras.txt", cameras)
    write_images_text(sparse_out / "images.txt", image_records)
    write_ply_points(sparse_out / "points3D.ply", points, colors)
    np.save(sparse_out / "confidence_dsp.npy", confidence.reshape(-1, 1))

    gaussian_report = {"enabled": False}
    if args.global_init_mode == "gaussian_submaps":
        gaussian_report = merge_gaussian_submaps(
            args,
            window_records,
            global_scene / "initial_gaussians.ply",
            args.gaussian_voxel_size if args.gaussian_voxel_size > 0 else max(diag * args.gaussian_voxel_diag_ratio, 1e-6),
            final_records_by_name,
            camera_template,
            diag,
        )

    report = {
        "global_scene": str(global_scene),
        "n_views": len(sampled_images),
        "point_count_before_dedupe": int(before),
        "point_count_after_dedupe": int(points.shape[0]),
        "voxel_size": voxel_size,
        "camera_path_diag": diag,
        "median_camera": asdict(camera_template),
        "pose_sources": pose_sources,
        "point_merge_mode": args.point_merge_mode,
        "use_window_masks": not args.no_window_masks,
        "gaussian_submaps": gaussian_report,
    }
    (output_dir / "longseq_merge_report.json").write_text(
        json.dumps(report, indent=2, default=lambda value: value.tolist() if hasattr(value, "tolist") else value),
        encoding="utf-8",
    )
    return global_scene


def run_train_render(args: argparse.Namespace, global_scene: Path) -> None:
    output_dir = Path(args.output_dir).resolve()
    train_out = output_dir / args.train_name
    instant_splat_root = Path(__file__).resolve().parent
    n_views = len(list((global_scene / "images").iterdir()))
    train_command = [
        args.python,
        "train.py",
        "-s",
        str(global_scene),
        "-m",
        str(train_out),
        "--n_views",
        str(n_views),
        "--iterations",
        str(args.iterations),
        "--test_iterations",
        str(args.iterations),
        "--pp_optimizer",
        "--optim_pose",
        "--pose_freeze_iters",
        str(args.pose_freeze_iters),
        "--pose_lr_scale",
        str(args.pose_lr_scale),
    ]
    append_train_lr_overrides(train_command, args, "global")
    initial_gaussians_path = global_scene / "initial_gaussians.ply"
    if args.global_init_mode == "gaussian_submaps":
        if not initial_gaussians_path.exists():
            raise FileNotFoundError(f"Missing merged initial Gaussians: {initial_gaussians_path}")
        train_command.extend(["--initial_gaussians_path", str(initial_gaussians_path)])
    code = run_command(train_command, instant_splat_root, train_out / "logs" / "train.log")
    if code != 0:
        raise RuntimeError(f"Global train failed: exit {code}")

    if args.skip_render:
        return

    render_command = [
        args.python,
        "render.py",
        "-s",
        str(global_scene),
        "-m",
        str(train_out),
        "--n_views",
        str(n_views),
        "--iterations",
        str(args.iterations),
        "--infer_video",
    ]
    code = run_command(render_command, instant_splat_root, train_out / "logs" / "render.log")
    if code != 0:
        raise RuntimeError(f"Global render failed: exit {code}")


def write_alignment_reports(output_dir: Path, windows: Sequence[WindowRecord]) -> None:
    records = []
    for window in windows:
        row = asdict(window)
        transform = row["transform"]
        transform["rotation"] = np.asarray(transform["rotation"]).tolist()
        transform["translation"] = np.asarray(transform["translation"]).tolist()
        records.append(row)
    (output_dir / "longseq_alignment.json").write_text(
        json.dumps(records, indent=2, default=lambda value: value.tolist() if hasattr(value, "tolist") else value),
        encoding="utf-8",
    )
    with (output_dir / "longseq_alignment.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "window",
                "start",
                "status",
                "method",
                "shared_images",
                "scale",
                "translation_rmse",
                "translation_ratio",
                "rotation_mean_deg",
                "rotation_max_deg",
                "is_bad",
                "bad_reasons",
            ]
        )
        for window in windows:
            alignment = window.alignment
            writer.writerow(
                [
                    window.index,
                    window.start,
                    alignment.get("status"),
                    alignment.get("method"),
                    " ".join(alignment.get("shared_images", [])),
                    alignment.get("scale"),
                    alignment.get("translation_rmse"),
                    alignment.get("translation_ratio"),
                    alignment.get("rotation_mean_deg"),
                    alignment.get("rotation_max_deg"),
                    alignment.get("is_bad"),
                    " | ".join(alignment.get("bad_reasons", [])),
                ]
            )


def write_window_alignment_debug(output_dir: Path, window: WindowRecord) -> None:
    debug_dir = output_dir / "alignment_debug" / f"window_{window.index:03d}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in window.alignment.items() if key != "transformed_records"}
    (debug_dir / "residuals.json").write_text(
        json.dumps(payload, indent=2, default=lambda value: value.tolist() if hasattr(value, "tolist") else value),
        encoding="utf-8",
    )
    with (debug_dir / "shared_residuals.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "translation_error", "rotation_deg"])
        for residual in window.alignment.get("shared_residuals", []):
            writer.writerow([residual["image"], residual["translation_error"], residual["rotation_deg"]])


def run_pipeline(args: argparse.Namespace) -> Path:
    instant_splat_root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).resolve()
    workspace = output_dir / "longseq_workspace"
    image_dir = resolve_image_dir(Path(args.source_path).resolve())
    all_images = list_images(image_dir)
    sampled = sample_images(all_images, args.sample_stride, args.n_views, args.offset)
    starts = make_window_starts(len(sampled), args.window_size, args.overlap)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_images.txt").write_text("\n".join(str(p) for p in sampled) + "\n", encoding="utf-8")
    (output_dir / "window_starts.json").write_text(json.dumps(starts, indent=2), encoding="utf-8")

    global_records: Dict[str, ImageRecord] = {}
    global_record_scores: Dict[str, float] = {}
    window_records: List[WindowRecord] = []
    previous_transform = None
    sampled_name_to_index = {image.name: idx for idx, image in enumerate(sampled)}

    for win_idx, start in enumerate(starts):
        window_images = sampled[start : start + args.window_size]
        scene_path = workspace / "window_scenes" / f"window_{win_idx:03d}"
        model_path = workspace / "window_inits" / f"window_{win_idx:03d}"
        materialize_scene(scene_path, window_images, args.link_mode, args.overwrite_windows)
        run_window_init(
            instant_splat_root,
            args.python,
            scene_path,
            model_path,
            len(window_images),
            args.window_scene_graph,
            args.skip_existing,
        )
        sparse_path = scene_path / f"sparse_{len(window_images)}" / "0"
        alignment_bundle, transformed = build_alignment_for_window(
            args,
            window_images,
            sparse_path,
            global_records,
            previous_transform,
            window_records[-1] if window_records else None,
        )
        transform = alignment_bundle["transform"]
        alignment = alignment_bundle["alignment"]
        alignment["transformed_records"] = {
            name: {"qvec": rec.qvec.tolist(), "tvec": rec.tvec.tolist()}
            for name, rec in transformed.items()
        }
        window_record = WindowRecord(
            index=win_idx,
            start=start,
            image_paths=[str(p) for p in window_images],
            scene_path=str(scene_path),
            init_model_path=str(model_path),
            gaussian_model_path="",
            sparse_path=str(sparse_path),
            transform=transform,
            alignment=alignment,
        )
        window_records.append(window_record)
        write_alignment_reports(output_dir, window_records)
        write_window_alignment_debug(output_dir, window_record)
        print(
            f"[align] window {win_idx:03d}: {alignment['status']} method={alignment.get('method')} "
            f"shared={len(alignment.get('shared_images', []))} "
            f"trans_rmse={alignment.get('translation_rmse', 0):.6f} "
            f"trans_ratio={alignment.get('translation_ratio', 0):.6f} "
            f"rot_mean={alignment.get('rotation_mean_deg', 0):.4f} "
            f"bad={alignment.get('is_bad', False)}"
        )
        if alignment.get("is_bad") and not args.allow_bad_alignment:
            reasons = "; ".join(alignment.get("bad_reasons", []))
            raise RuntimeError(
                f"Bad cross-window alignment at window {win_idx:03d}. "
                f"{reasons}. Re-run with --allow_bad_alignment only for debugging."
            )
        gaussian_model_path = ""
        if args.global_init_mode == "gaussian_submaps":
            gaussian_model = workspace / "window_gaussians" / f"window_{win_idx:03d}"
            run_window_gaussian_train(
                instant_splat_root,
                args,
                scene_path,
                gaussian_model,
                len(window_images),
            )
            gaussian_model_path = str(gaussian_model)
            window_record.gaussian_model_path = gaussian_model_path
        update_best_records(
            global_records,
            global_record_scores,
            transformed,
            start,
            len(window_images),
            sampled_name_to_index,
        )
        previous_transform = transform

    if args.gaussian_refine_mode != "none":
        refine_gaussian_submap_transforms(args, window_records, output_dir)
        global_records = {}
        global_record_scores = {}
        for record in window_records:
            transformed_records = transformed_payload_to_records(record.alignment.get("transformed_records", {}))
            update_best_records(
                global_records,
                global_record_scores,
                transformed_records,
                record.start,
                len(record.image_paths),
                sampled_name_to_index,
            )
            write_window_alignment_debug(output_dir, record)

    write_alignment_reports(output_dir, window_records)
    global_scene = merge_global_scene(args, sampled, window_records)
    if not args.init_only:
        run_train_render(args, global_scene)
    return global_scene


def self_test() -> None:
    starts = make_window_starts(20, 6, 3)
    assert starts == [0, 3, 6, 9, 12, 14], starts
    rng = np.random.default_rng(7)
    src = rng.normal(size=(8, 3))
    angle = 0.37
    rot = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    scale = 2.4
    trans = np.array([0.5, -1.2, 3.0])
    dst = apply_similarity(src, scale, rot, trans)
    est_scale, est_rot, est_trans = umeyama_similarity(src, dst)
    assert abs(est_scale - scale) < 1e-10
    assert np.max(np.abs(est_rot - rot)) < 1e-10
    assert np.max(np.abs(est_trans - trans)) < 1e-10
    est_scale_fixed, est_trans_fixed = fixed_rotation_similarity(src, dst, rot)
    assert abs(est_scale_fixed - scale) < 1e-10
    assert np.max(np.abs(est_trans_fixed - trans)) < 1e-10
    est_avg_rot = average_rotations(np.stack([rot, rot, rot]))
    assert np.max(np.abs(est_avg_rot - rot)) < 1e-10
    round_trip = (apply_similarity(dst - trans, 1.0 / scale, rot.T, np.zeros(3)))
    assert np.max(np.abs(round_trip - src)) < 1e-10
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]
    vertices = np.zeros(1, dtype=dtype)
    vertices["x"] = 1.0
    vertices["rot_0"] = 1.0
    transformed = transform_gaussian_vertices(vertices, scale, rot, trans)
    expected_xyz = apply_similarity(np.array([[1.0, 0.0, 0.0]]), scale, rot, trans)
    assert np.max(np.abs(np.stack([transformed["x"], transformed["y"], transformed["z"]], axis=1) - expected_xyz)) < 1e-6
    assert abs(float(transformed["scale_0"][0]) - np.log(scale)) < 1e-6
    import torch

    rotvec = torch.zeros(3, dtype=torch.float32, requires_grad=True)
    point = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([[1.0, 0.2, 0.0]], dtype=torch.float32)
    moved = point @ torch_rotvec_to_matrix(rotvec).T
    loss = torch.mean((moved - target) ** 2)
    loss.backward()
    assert torch.isfinite(rotvec.grad).all()
    assert abs(float(rotvec.grad[2])) > 1e-4
    print("long_sequence_align self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InstantSplat long-sequence cross-window alignment")
    parser.add_argument("--source_path", type=Path, default=Path("/root/autodl-fs/405841/FRONT"))
    parser.add_argument("--output_dir", type=Path, default=Path("output_infer/long_sequence_align_front"))
    parser.add_argument("--n_views", type=int, default=20)
    parser.add_argument("--sample_stride", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--window_size", type=int, default=6)
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--python", default="python")
    parser.add_argument("--window_scene_graph", default="complete")
    parser.add_argument("--alignment_mode", choices=["center_umeyama", "rotation_center", "hybrid"], default="hybrid")
    parser.add_argument("--max_rotation_residual_deg", type=float, default=10.0)
    parser.add_argument("--max_translation_residual_ratio", type=float, default=0.03)
    parser.add_argument("--min_scale", type=float, default=0.3)
    parser.add_argument("--max_scale", type=float, default=3.0)
    parser.add_argument("--allow_bad_alignment", action="store_true")
    parser.add_argument("--link_mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--voxel_size", type=float, default=0.0)
    parser.add_argument("--voxel_diag_ratio", type=float, default=0.003)
    parser.add_argument("--point_merge_mode", choices=["ply", "owner_views"], default="ply")
    parser.add_argument(
        "--global_init_mode",
        choices=["point_cloud", "gaussian_submaps"],
        default="gaussian_submaps",
        help="Global train init: point_cloud=from merged points3D only; gaussian_submaps=from merged aligned window Gaussians (default).",
    )
    parser.add_argument("--window_gaussian_iterations", type=int, default=300)
    parser.add_argument("--window_gaussian_pose_freeze_iters", type=int, default=-1)
    parser.add_argument("--window_gaussian_pose_lr_scale", type=float, default=0.1)
    parser.add_argument("--window_position_lr_init", type=float, default=0.0)
    parser.add_argument("--window_position_lr_final", type=float, default=0.0)
    parser.add_argument("--window_scaling_lr", type=float, default=0.0)
    parser.add_argument("--window_rotation_lr", type=float, default=0.0)
    parser.add_argument("--gaussian_voxel_size", type=float, default=0.0)
    parser.add_argument("--gaussian_voxel_diag_ratio", type=float, default=0.003)
    parser.add_argument("--gaussian_per_window_max_count", type=int, default=0)
    parser.add_argument("--gaussian_merge_max_count", type=int, default=800000)
    parser.add_argument("--gaussian_fusion_mode", choices=["batch_voxel", "incremental_visibility"], default="batch_voxel")
    parser.add_argument("--gaussian_visibility_cell_size", type=int, default=8)
    parser.add_argument("--gaussian_visibility_depth_ratio", type=float, default=0.01)
    parser.add_argument("--gaussian_visibility_min_shared", type=int, default=2)
    parser.add_argument("--gaussian_score_mode", choices=["opacity", "opacity_scale_penalty"], default="opacity")
    parser.add_argument("--gaussian_scale_penalty", type=float, default=1.0)
    parser.add_argument("--gaussian_scale_max_quantile", type=float, default=1.0)
    parser.add_argument("--gaussian_refine_mode", choices=["none", "nn_overlap"], default="none")
    parser.add_argument("--gaussian_refine_sample_points", type=int, default=5000)
    parser.add_argument("--gaussian_refine_main_points", type=int, default=12000)
    parser.add_argument("--gaussian_refine_max_overlap_points", type=int, default=4000)
    parser.add_argument("--gaussian_refine_min_overlap", type=int, default=500)
    parser.add_argument("--gaussian_refine_nn_radius_ratio", type=float, default=0.02)
    parser.add_argument("--gaussian_refine_iters", type=int, default=80)
    parser.add_argument("--gaussian_refine_lr", type=float, default=0.02)
    parser.add_argument("--gaussian_refine_optimize_rotation", action="store_true")
    parser.add_argument("--gaussian_refine_min_improvement_ratio", type=float, default=0.005)
    parser.add_argument("--gaussian_refine_max_log_scale", type=float, default=0.03)
    parser.add_argument("--gaussian_refine_max_rotation_deg", type=float, default=2.0)
    parser.add_argument("--gaussian_refine_max_translation_ratio", type=float, default=0.01)
    parser.add_argument("--gaussian_refine_scale_reg", type=float, default=0.1)
    parser.add_argument("--gaussian_refine_rot_reg", type=float, default=0.1)
    parser.add_argument("--gaussian_refine_trans_reg", type=float, default=0.1)
    parser.add_argument("--no_window_masks", action="store_true")
    parser.add_argument("--dense_refine", action="store_true")
    parser.add_argument("--dense_refine_max_points", type=int, default=50000)
    parser.add_argument("--dense_refine_min_points", type=int, default=2000)
    parser.add_argument("--dense_refine_conf_quantile", type=float, default=0.75)
    parser.add_argument("--dense_refine_accept_ratio", type=float, default=0.98)
    parser.add_argument("--dense_refine_pose_ratio", type=float, default=1.1)
    parser.add_argument("--dense_refine_use_masks", action="store_true")
    parser.add_argument("--dense_refine_seed", type=int, default=13)
    parser.add_argument("--train_name", default="global_train")
    parser.add_argument("--pose_freeze_iters", type=int, default=300)
    parser.add_argument("--pose_lr_scale", type=float, default=0.25)
    parser.add_argument("--global_position_lr_init", type=float, default=0.0)
    parser.add_argument("--global_position_lr_final", type=float, default=0.0)
    parser.add_argument("--global_scaling_lr", type=float, default=0.0)
    parser.add_argument("--global_rotation_lr", type=float, default=0.0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--overwrite_windows", action="store_true")
    parser.add_argument("--overwrite_global", action="store_true")
    parser.add_argument("--init_only", action="store_true")
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.self_test:
        self_test()
    else:
        scene = run_pipeline(parsed)
        print(f"Global scene ready: {scene}")
