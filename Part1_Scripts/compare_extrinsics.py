#!/usr/bin/env python3
"""Compare COLMAP/VGGT camera extrinsics after Sim(3) alignment.

Example:
    python compare_extrinsics.py \
        --ref-model /root/autodl-fs/DL3DV-2-200images/sparse/0 \
        --query-model /root/autodl-fs/VGGT-colmap/DL3DV-2_200images-300000points-conf1/sparse/0 \
        --plot-path compare_vggt_vs_colmap.png \
        --json-out compare_vggt_vs_colmap.json
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass
class ImageRecord:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


@dataclass
class Pose:
    name: str
    c2w: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two COLMAP-format extrinsic sets.")
    parser.add_argument("--ref-model", type=Path, required=True, help="Reference COLMAP model directory.")
    parser.add_argument("--query-model", type=Path, required=True, help="Query COLMAP model directory.")
    parser.add_argument("--ref-label", default="COLMAP", help="Label for the reference trajectory.")
    parser.add_argument("--query-label", default="VGGT", help="Label for the query trajectory.")
    parser.add_argument("--match-mode", choices=["exact", "basename"], default="basename", help="How to match image names across two models.")
    parser.add_argument("--json-out", type=Path, help="Optional JSON file for summary metrics.")
    parser.add_argument("--csv-out", type=Path, help="Optional CSV file with per-image errors.")
    parser.add_argument("--plot-path", type=Path, help="Optional image path to save a trajectory plot.")
    return parser.parse_args()


def read_bytes(fid, num_bytes: int, fmt: str):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("Unexpected end of file while reading COLMAP model.")
    return struct.unpack("<" + fmt, data)


def read_c_string(fid) -> str:
    buffer = bytearray()
    while True:
        char = fid.read(1)
        if not char:
            raise EOFError("Unexpected EOF while reading a null-terminated string.")
        if char == b"\x00":
            return buffer.decode("utf-8")
        buffer.extend(char)


def read_images_binary(path: Path) -> dict[int, ImageRecord]:
    images = {}
    with path.open("rb") as fid:
        num_images = read_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_bytes(fid, 4, "i")[0]
            qvec = np.array(read_bytes(fid, 8 * 4, "dddd"), dtype=np.float64)
            tvec = np.array(read_bytes(fid, 8 * 3, "ddd"), dtype=np.float64)
            camera_id = read_bytes(fid, 4, "i")[0]
            name = read_c_string(fid)
            num_points2d = read_bytes(fid, 8, "Q")[0]
            fid.seek(num_points2d * 24, 1)
            images[image_id] = ImageRecord(
                image_id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
            )
    return images


def read_images_text(path: Path) -> dict[int, ImageRecord]:
    images = {}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line or line.startswith("#"):
            i += 1
            continue
        tokens = line.split()
        image_id = int(tokens[0])
        qvec = np.array([float(x) for x in tokens[1:5]], dtype=np.float64)
        tvec = np.array([float(x) for x in tokens[5:8]], dtype=np.float64)
        camera_id = int(tokens[8])
        name = tokens[9]
        images[image_id] = ImageRecord(
            image_id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=name,
        )
        i += 2
    return images


def load_images(model_dir: Path) -> dict[int, ImageRecord]:
    images_bin = model_dir / "images.bin"
    images_txt = model_dir / "images.txt"
    if images_bin.exists():
        return read_images_binary(images_bin)
    if images_txt.exists():
        return read_images_text(images_txt)
    raise FileNotFoundError(
        f"Could not find images.bin or images.txt in {model_dir}"
    )


def qvec_to_rotmat(qvec_wxyz: np.ndarray) -> np.ndarray:
    q = qvec_wxyz.astype(np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def image_to_pose(image: ImageRecord) -> Pose:
    rot_w2c = qvec_to_rotmat(image.qvec)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rot_w2c.T
    c2w[:3, 3] = -rot_w2c.T @ image.tvec
    return Pose(name=image.name, c2w=c2w)


def pose_key(name: str, mode: str) -> str:
    return Path(name).name if mode == "basename" else name


def match_poses(ref_images: dict[int, ImageRecord], query_images: dict[int, ImageRecord], mode: str) -> tuple[list[Pose], list[Pose]]:
    ref_map = {pose_key(image.name, mode): image_to_pose(image) for image in ref_images.values()}
    query_map = {pose_key(image.name, mode): image_to_pose(image) for image in query_images.values()}
    common_keys = sorted(set(ref_map).intersection(query_map))
    if not common_keys:
        raise ValueError("No common image names found between the two models.")
    return [ref_map[key] for key in common_keys], [query_map[key] for key in common_keys]


def umeyama_alignment(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    # Solve dst ~= scale * R @ src + t
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = (dst_centered.T @ src_centered) / src.shape[0]
    u, d, vt = np.linalg.svd(covariance)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1
    rot = u @ s @ vt
    src_var = np.mean(np.sum(src_centered ** 2, axis=1))
    scale = np.trace(np.diag(d) @ s) / src_var
    trans = dst_mean - scale * (rot @ src_mean)
    return float(scale), rot, trans


def rotation_error_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rel = rot_a @ rot_b.T
    trace = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def apply_sim3_to_pose(pose: Pose, scale: float, rot: np.ndarray, trans: np.ndarray) -> Pose:
    aligned = np.eye(4, dtype=np.float64)
    aligned[:3, :3] = rot @ pose.c2w[:3, :3]
    aligned[:3, 3] = scale * (rot @ pose.c2w[:3, 3]) + trans
    return Pose(name=pose.name, c2w=aligned)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "max": float(np.max(values)),
    }


def relative_pose_error(ref_poses: list[Pose], query_poses: list[Pose]) -> tuple[np.ndarray, np.ndarray]:
    trans_errors = []
    rot_errors = []
    for idx in range(len(ref_poses) - 1):
        ref_rel = np.linalg.inv(ref_poses[idx].c2w) @ ref_poses[idx + 1].c2w
        query_rel = np.linalg.inv(query_poses[idx].c2w) @ query_poses[idx + 1].c2w
        trans_errors.append(np.linalg.norm(ref_rel[:3, 3] - query_rel[:3, 3]))
        rot_errors.append(rotation_error_deg(ref_rel[:3, :3], query_rel[:3, :3]))
    return np.array(trans_errors, dtype=np.float64), np.array(rot_errors, dtype=np.float64)


def write_csv(path: Path, names: list[str], trans_errors: np.ndarray, rot_errors: np.ndarray) -> None:
    lines = ["image_name,translation_error,rotation_error_deg"]
    for name, terr, rerr in zip(names, trans_errors, rot_errors):
        lines.append(f"{name},{terr:.8f},{rerr:.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_trajectories(path: Path, ref_poses: list[Pose], aligned_query_poses: list[Pose], ref_label: str, query_label: str) -> None:
    ref_centers = np.stack([pose.c2w[:3, 3] for pose in ref_poses], axis=0)
    query_centers = np.stack([pose.c2w[:3, 3] for pose in aligned_query_poses], axis=0)

    fig = plt.figure(figsize=(8, 7))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(ref_centers[:, 0], ref_centers[:, 1], ref_centers[:, 2], label=ref_label, linewidth=2)
    axis.plot(query_centers[:, 0], query_centers[:, 1], query_centers[:, 2], label=query_label, linewidth=2)
    axis.scatter(ref_centers[0, 0], ref_centers[0, 1], ref_centers[0, 2], c="green", s=35)
    axis.scatter(query_centers[0, 0], query_centers[0, 1], query_centers[0, 2], c="red", s=35)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_title("Aligned Camera Trajectories")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ref_images = load_images(args.ref_model)
    query_images = load_images(args.query_model)
    ref_poses, query_poses = match_poses(ref_images, query_images, args.match_mode)

    ref_centers = np.stack([pose.c2w[:3, 3] for pose in ref_poses], axis=0)
    query_centers = np.stack([pose.c2w[:3, 3] for pose in query_poses], axis=0)
    scale, rot, trans = umeyama_alignment(query_centers, ref_centers)
    aligned_query_poses = [apply_sim3_to_pose(pose, scale, rot, trans) for pose in query_poses]
    aligned_query_centers = np.stack([pose.c2w[:3, 3] for pose in aligned_query_poses], axis=0)

    trans_errors = np.linalg.norm(ref_centers - aligned_query_centers, axis=1)
    rot_errors = np.array(
        [
            rotation_error_deg(ref_pose.c2w[:3, :3], query_pose.c2w[:3, :3])
            for ref_pose, query_pose in zip(ref_poses, aligned_query_poses)
        ],
        dtype=np.float64,
    )
    rpe_trans, rpe_rot = relative_pose_error(ref_poses, aligned_query_poses)

    summary = {
        "num_matched_images": len(ref_poses),
        "match_mode": args.match_mode,
        "sim3_alignment": {
            "scale": scale,
            "rotation_matrix": rot.tolist(),
            "translation": trans.tolist(),
        },
        "absolute_translation_error": summarize(trans_errors),
        "absolute_rotation_error_deg": summarize(rot_errors),
        "relative_translation_error": summarize(rpe_trans) if len(rpe_trans) else None,
        "relative_rotation_error_deg": summarize(rpe_rot) if len(rpe_rot) else None,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.json_out:
        args.json_out.expanduser().resolve().write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if args.csv_out:
        write_csv(
            args.csv_out.expanduser().resolve(),
            [pose.name for pose in ref_poses],
            trans_errors,
            rot_errors,
        )

    if args.plot_path:
        plot_trajectories(
            args.plot_path.expanduser().resolve(),
            ref_poses,
            aligned_query_poses,
            args.ref_label,
            args.query_label,
        )


if __name__ == "__main__":
    main()
