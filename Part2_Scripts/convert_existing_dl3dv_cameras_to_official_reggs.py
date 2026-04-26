#!/usr/bin/env python3
"""Regenerate a RegGS cameras.json using the official DL3DV axis logic.

This is a fallback tool for cases where the original DL3DV `transforms.json`
is unavailable, but an existing `cameras.json` is already present.

Assumption:
    The existing pose in `cameras.json` should be interpreted as the original
    DL3DV/OpenGL-style camera-to-world matrix. We then apply the same
    conversion used by `src/noposplat/scripts/convert_dl3dv.py`:

        OpenGL c2w -> OpenCV w2c -> OpenCV c2w (for RegGS)

The output is written to a new file by default so the original file is kept.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an existing DL3DV cameras.json using official axis logic."
    )
    parser.add_argument(
        "--input-cameras",
        type=Path,
        default=Path("/root/autodl-fs/DL3DV-2/cameras.json"),
        help="Existing cameras.json file.",
    )
    parser.add_argument(
        "--output-cameras",
        type=Path,
        default=Path("/root/autodl-fs/DL3DV-2/cameras_official_reggs.json"),
        help="Output RegGS-format cameras.json file.",
    )
    return parser.parse_args()


def quat_xyzw_to_rotmat(quat_xyzw: list[float] | np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64)
    n = np.linalg.norm([x, y, z, w])
    if n == 0:
        raise ValueError("Zero quaternion is invalid.")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


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


def entry_to_c2w(entry: dict) -> np.ndarray:
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = quat_xyzw_to_rotmat(entry["cam_quat"])
    c2w[:3, 3] = np.asarray(entry["cam_trans"], dtype=np.float64)
    return c2w


def opengl_c2w_to_opencv_w2c(c2w: np.ndarray) -> np.ndarray:
    # Keep the exact axis conversion used by official convert_dl3dv.py.
    c2w = c2w.copy()
    c2w[2, :] *= -1
    c2w = c2w[np.array([1, 0, 2, 3]), :]
    c2w[0:3, 1:3] *= -1
    return np.linalg.inv(c2w)


def main() -> None:
    args = parse_args()

    cameras = json.loads(args.input_cameras.read_text(encoding="utf-8"))
    converted = []

    for entry in cameras:
        # Assumption: current entry encodes the same OpenGL c2w that the
        # official DL3DV conversion script would have read from transforms.json.
        source_opengl_c2w = entry_to_c2w(entry)
        opencv_w2c = opengl_c2w_to_opencv_w2c(source_opengl_c2w)
        opencv_c2w = np.linalg.inv(opencv_w2c)

        new_entry = dict(entry)
        new_entry["cam_quat"] = rotmat_to_quat_xyzw(opencv_c2w[:3, :3]).tolist()
        new_entry["cam_trans"] = opencv_c2w[:3, 3].tolist()
        converted.append(new_entry)

    args.output_cameras.write_text(
        json.dumps(converted, indent=4), encoding="utf-8"
    )
    print(f"Wrote {args.output_cameras}")
    print(
        "Assumption used: existing cameras.json represents OpenGL-style c2w; "
        "official DL3DV axis conversion was then applied."
    )


if __name__ == "__main__":
    main()
