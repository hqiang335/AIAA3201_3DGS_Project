#!/usr/bin/env python3
"""Convert FRONT calib/pose files to RegGS-style intrinsics/cameras JSON.

Output format matches files like:
  - /root/autodl-fs/DL3DV-2/intrinsics.json
  - /root/autodl-fs/DL3DV-2/cameras.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FRONT calib + gt poses to RegGS JSON format."
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=Path("/root/autodl-fs/405841/FRONT/calib"),
        help="Directory containing FRONT per-frame calibration txt files.",
    )
    parser.add_argument(
        "--pose-dir",
        type=Path,
        default=Path("/root/autodl-fs/405841/FRONT/gt"),
        help="Directory containing per-frame 4x4 pose txt files (gt recommended).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/root/autodl-fs/405841/FRONT/images"),
        help="Directory containing frame images for resolution and names.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/autodl-fs/405841/FRONT"),
        help="Where to write intrinsics.json and cameras.json.",
    )
    parser.add_argument(
        "--pose-type",
        choices=["c2w", "w2c"],
        default="c2w",
        help="Interpretation of matrices in --pose-dir txt files.",
    )
    parser.add_argument(
        "--normalize-first",
        action="store_true",
        help="Normalize all c2w by inv(first) @ c2w, similar to RegGS dataset loader.",
    )
    return parser.parse_args()


def rotmat_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    trace = np.trace(rot)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    else:
        if rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif rot[1, 1] > rot[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat


def parse_calib_intrinsics(calib_file: Path) -> tuple[float, float, float, float]:
    text = calib_file.read_text(encoding="utf-8")
    # Example line:
    # fx: 2066.69 fy: 2066.69 cx: 950.55 cy: 641.18
    fx_match = re.search(r"fx:\s*([-\d.eE+]+)", text)
    fy_match = re.search(r"fy:\s*([-\d.eE+]+)", text)
    cx_match = re.search(r"cx:\s*([-\d.eE+]+)", text)
    cy_match = re.search(r"cy:\s*([-\d.eE+]+)", text)
    if not (fx_match and fy_match and cx_match and cy_match):
        raise ValueError(f"Failed to parse fx/fy/cx/cy from {calib_file}")
    return (
        float(fx_match.group(1)),
        float(fy_match.group(1)),
        float(cx_match.group(1)),
        float(cy_match.group(1)),
    )


def read_pose_matrix(pose_file: Path) -> np.ndarray:
    lines = [line.strip() for line in pose_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 4:
        raise ValueError(f"Pose file {pose_file} has fewer than 4 lines.")
    mat = []
    for i in range(4):
        row = [float(x) for x in lines[i].split()]
        if len(row) != 4:
            raise ValueError(f"Pose file {pose_file}, line {i + 1} is not 4 numbers.")
        mat.append(row)
    return np.array(mat, dtype=np.float64)


def main() -> None:
    args = parse_args()

    calib_files = sorted(args.calib_dir.glob("*.txt"), key=lambda p: p.stem)
    image_files = sorted(args.images_dir.glob("*.png"), key=lambda p: p.stem)
    if not image_files:
        image_files = sorted(args.images_dir.glob("*.jpg"), key=lambda p: p.stem)

    if not calib_files:
        raise FileNotFoundError(f"No calib txt files found in {args.calib_dir}")
    if not image_files:
        raise FileNotFoundError(f"No images found in {args.images_dir}")

    # Use the first calib file as the canonical shared intrinsics.
    fx, fy, cx, cy = parse_calib_intrinsics(calib_files[0])

    # Validate image resolution and build normalized intrinsics.
    with Image.open(image_files[0]) as im:
        width, height = im.size

    intrinsics = {
        "fx": fx / width,
        "fy": fy / height,
        "cx": cx / width,
        "cy": cy / height,
    }

    # Build cameras.json entries by image frame order.
    c2ws = []
    ordered_images = []
    for img in image_files:
        stem = img.stem
        pose_file = args.pose_dir / f"{stem}.txt"
        if not pose_file.exists():
            continue
        pose = read_pose_matrix(pose_file)
        if args.pose_type == "w2c":
            c2w = np.linalg.inv(pose)
        else:
            c2w = pose
        c2ws.append(c2w)
        ordered_images.append(img)

    if not c2ws:
        raise FileNotFoundError(
            f"No matching pose txt found in {args.pose_dir} for images in {args.images_dir}"
        )

    if args.normalize_first:
        first_inv = np.linalg.inv(c2ws[0])
        c2ws = [first_inv @ c2w for c2w in c2ws]

    cameras = []
    for idx, (img, c2w) in enumerate(zip(ordered_images, c2ws)):
        quat_xyzw = rotmat_to_quat_xyzw(c2w[:3, :3])
        trans = c2w[:3, 3]
        cameras.append(
            {
                "cam_id": idx,
                "cam_quat": quat_xyzw.tolist(),
                "cam_trans": trans.tolist(),
                "cx": intrinsics["cx"],
                "cy": intrinsics["cy"],
                "fx": intrinsics["fx"],
                "fy": intrinsics["fy"],
                "image_name": img.name,
                "timestamp": int(img.stem) if img.stem.isdigit() else idx,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    intrinsics_path = args.output_dir / "intrinsics.json"
    cameras_path = args.output_dir / "cameras.json"

    intrinsics_path.write_text(json.dumps(intrinsics, indent=4), encoding="utf-8")
    cameras_path.write_text(json.dumps(cameras, indent=4), encoding="utf-8")

    print(f"Wrote {intrinsics_path}")
    print(f"Wrote {cameras_path} ({len(cameras)} frames)")
    print(f"Image resolution used for normalization: {width}x{height}")


if __name__ == "__main__":
    main()

