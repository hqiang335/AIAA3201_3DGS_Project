#!/usr/bin/env python3
"""Convert a COLMAP sparse model into the data layout expected by RegGS.

Expected RegGS layout:
    <output_scene_dir>/
      images/
        00000.png
        00001.png
        ...
      intrinsics.json
      cameras.json

This script reads a COLMAP sparse reconstruction (binary or text), copies or
resizes the registered images, converts poses from COLMAP world-to-camera to
camera-to-world, and writes the JSON files used by RegGS.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


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
CAMERA_MODEL_NAME_TO_ID = {name: model_id for model_id, (name, _) in CAMERA_MODELS.items()}


@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class ImageRecord:
    image_id: int
    qvec: np.ndarray  # COLMAP qvec, wxyz, for world-to-camera rotation.
    tvec: np.ndarray  # COLMAP translation, world-to-camera.
    camera_id: int
    name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a COLMAP sparse model to RegGS dataset format."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing the original images referenced by COLMAP.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing COLMAP sparse model files such as cameras.bin/images.bin.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output scene directory to create for RegGS.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="Optional output image width. If omitted, keep COLMAP image width.",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=None,
        help="Optional output image height. If omitted, keep COLMAP image height.",
    )
    parser.add_argument(
        "--image-format",
        choices=["png", "jpg"],
        default="png",
        help="Output image format for RegGS images directory.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["name", "image_id"],
        default="name",
        help="How to order registered images before renaming/export.",
    )
    parser.add_argument(
        "--name-mode",
        choices=["sequential", "original"],
        default="sequential",
        help="Use sequential frame names or keep original image stems.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    parser.add_argument(
        "--allow-varying-intrinsics-use-first",
        action="store_true",
        help=(
            "RegGS expects one shared intrinsic matrix. By default the script "
            "fails if COLMAP cameras differ. With this flag it will continue "
            "and use the first camera intrinsics for intrinsics.json."
        ),
    )
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


def read_cameras_binary(path: Path) -> dict[int, Camera]:
    cameras = {}
    with path.open("rb") as fid:
        num_cameras = read_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_bytes(fid, 24, "iiQQ")
            model_name, num_params = CAMERA_MODELS[model_id]
            params = np.array(read_bytes(fid, 8 * num_params, "d" * num_params), dtype=np.float64)
            cameras[camera_id] = Camera(
                camera_id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=params,
            )
    return cameras


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


def read_cameras_text(path: Path) -> dict[int, Camera]:
    cameras = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        camera_id = int(tokens[0])
        model = tokens[1]
        width = int(tokens[2])
        height = int(tokens[3])
        params = np.array([float(x) for x in tokens[4:]], dtype=np.float64)
        cameras[camera_id] = Camera(
            camera_id=camera_id,
            model=model,
            width=width,
            height=height,
            params=params,
        )
    return cameras


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


def load_colmap_model(model_dir: Path) -> tuple[dict[int, Camera], dict[int, ImageRecord], str]:
    cameras_bin = model_dir / "cameras.bin"
    images_bin = model_dir / "images.bin"
    cameras_txt = model_dir / "cameras.txt"
    images_txt = model_dir / "images.txt"

    if cameras_bin.exists() and images_bin.exists():
        return read_cameras_binary(cameras_bin), read_images_binary(images_bin), "binary"
    if cameras_txt.exists() and images_txt.exists():
        return read_cameras_text(cameras_txt), read_images_text(images_txt), "text"
    raise FileNotFoundError(
        f"Could not find COLMAP cameras/images files in {model_dir}. "
        "Expected either cameras.bin/images.bin or cameras.txt/images.txt."
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


def rotmat_to_quat_xyzw(rotmat: np.ndarray) -> np.ndarray:
    m = rotmat
    trace = np.trace(m)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s

    quat_xyzw = np.array([x, y, z, w], dtype=np.float64)
    quat_xyzw /= np.linalg.norm(quat_xyzw)
    return quat_xyzw


def colmap_image_to_c2w(image: ImageRecord) -> np.ndarray:
    rot_w2c = qvec_to_rotmat(image.qvec)
    trans_w2c = image.tvec.reshape(3, 1)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = rot_w2c
    w2c[:3, 3:] = trans_w2c
    return np.linalg.inv(w2c)


def camera_to_pinhole(camera: Camera) -> tuple[float, float, float, float]:
    params = camera.params
    model = camera.model
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "FOV"}:
        f, cx, cy = params[:3]
        return float(f), float(f), float(cx), float(cy)
    if model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"}:
        fx, fy, cx, cy = params[:4]
        return float(fx), float(fy), float(cx), float(cy)
    raise NotImplementedError(
        f"Unsupported COLMAP camera model '{model}'. "
        "Please extend camera_to_pinhole() for this model."
    )


def ensure_shared_intrinsics(
    cameras: dict[int, Camera],
    images: list[ImageRecord],
    allow_varying: bool,
) -> tuple[Camera, list[str]]:
    warnings = []
    first_camera = cameras[images[0].camera_id]
    first_params = np.array(camera_to_pinhole(first_camera), dtype=np.float64)

    differing = []
    for image in images[1:]:
        current_camera = cameras[image.camera_id]
        current_params = np.array(camera_to_pinhole(current_camera), dtype=np.float64)
        same_size = (
            current_camera.width == first_camera.width
            and current_camera.height == first_camera.height
        )
        same_params = np.allclose(current_params, first_params, atol=1e-6, rtol=1e-6)
        if not (same_size and same_params):
            differing.append(
                {
                    "camera_id": current_camera.camera_id,
                    "model": current_camera.model,
                    "size": [current_camera.width, current_camera.height],
                    "params": current_params.tolist(),
                }
            )

    if differing and not allow_varying:
        raise ValueError(
            "Detected varying COLMAP intrinsics across registered images, but RegGS "
            "expects one shared intrinsic matrix. Re-run COLMAP with shared intrinsics, "
            "filter to one camera, or pass --allow-varying-intrinsics-use-first "
            "to force using the first camera."
        )

    if differing:
        warnings.append(
            "Detected varying intrinsics. The script is continuing with the first "
            "camera intrinsics because --allow-varying-intrinsics-use-first was set."
        )

    return first_camera, warnings


def resolve_output_size(camera: Camera, args: argparse.Namespace) -> tuple[int, int]:
    if (args.resize_width is None) != (args.resize_height is None):
        raise ValueError("Please set both --resize-width and --resize-height together.")
    if args.resize_width is None:
        return camera.width, camera.height
    if args.resize_width <= 0 or args.resize_height <= 0:
        raise ValueError("Resize dimensions must be positive integers.")
    return args.resize_width, args.resize_height


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    if output_dir.exists():
        existing = list(output_dir.iterdir())
        if existing and not overwrite:
            raise FileExistsError(
                f"Output directory {output_dir} is not empty. Use --overwrite to continue."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def export_image(src_path: Path, dst_path: Path, output_size: tuple[int, int], expected_size: tuple[int, int]) -> None:
    with Image.open(src_path) as image:
        image = image.convert("RGB")
        if image.size != expected_size:
            raise ValueError(
                f"Image size mismatch for {src_path}: actual {image.size}, "
                f"but COLMAP camera expects {expected_size}."
            )
        if image.size != output_size:
            image = image.resize(output_size, Image.Resampling.LANCZOS)
        image.save(dst_path)


def build_output_name(index: int, original_name: str, suffix: str, mode: str) -> str:
    if mode == "original":
        return f"{Path(original_name).stem}.{suffix}"
    return f"{index:05d}.{suffix}"


def main() -> int:
    args = parse_args()

    if not args.images_dir.is_dir():
        raise FileNotFoundError(f"Images directory does not exist: {args.images_dir}")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {args.model_dir}")

    cameras, images_map, model_format = load_colmap_model(args.model_dir)
    if not images_map:
        raise ValueError("No registered images were found in the COLMAP model.")

    images = list(images_map.values())
    if args.sort_by == "name":
        images.sort(key=lambda item: item.name)
    else:
        images.sort(key=lambda item: item.image_id)

    shared_camera, warnings = ensure_shared_intrinsics(
        cameras,
        images,
        allow_varying=args.allow_varying_intrinsics_use_first,
    )
    output_width, output_height = resolve_output_size(shared_camera, args)
    output_images_dir = prepare_output_dir(args.output_dir, args.overwrite)

    fx, fy, cx, cy = camera_to_pinhole(shared_camera)
    scale_x = output_width / shared_camera.width
    scale_y = output_height / shared_camera.height
    fx_out = fx * scale_x
    fy_out = fy * scale_y
    cx_out = cx * scale_x
    cy_out = cy * scale_y

    intrinsics_json = {
        "fx": fx_out / output_width,
        "fy": fy_out / output_height,
        "cx": cx_out / output_width,
        "cy": cy_out / output_height,
    }

    camera_entries = []
    frame_map = []
    expected_input_size = (shared_camera.width, shared_camera.height)

    for index, image in enumerate(images):
        src_path = args.images_dir / image.name
        if not src_path.is_file():
            raise FileNotFoundError(
                f"COLMAP references image '{image.name}', but it was not found under {args.images_dir}."
            )

        dst_name = build_output_name(index, image.name, args.image_format, args.name_mode)
        dst_path = output_images_dir / dst_name
        export_image(
            src_path=src_path,
            dst_path=dst_path,
            output_size=(output_width, output_height),
            expected_size=expected_input_size,
        )

        c2w = colmap_image_to_c2w(image)
        quat_xyzw = rotmat_to_quat_xyzw(c2w[:3, :3])
        trans = c2w[:3, 3]

        camera_entries.append(
            {
                "cam_id": index,
                "cam_quat": quat_xyzw.tolist(),
                "cam_trans": trans.tolist(),
                "cx": intrinsics_json["cx"],
                "cy": intrinsics_json["cy"],
                "fx": intrinsics_json["fx"],
                "fy": intrinsics_json["fy"],
                "image_name": dst_name,
                "timestamp": index,
            }
        )
        frame_map.append(
            {
                "index": index,
                "image_id": image.image_id,
                "camera_id": image.camera_id,
                "original_image_name": image.name,
                "reggs_image_name": dst_name,
            }
        )

    (args.output_dir / "intrinsics.json").write_text(
        json.dumps(intrinsics_json, indent=4),
        encoding="utf-8",
    )
    (args.output_dir / "cameras.json").write_text(
        json.dumps(camera_entries, indent=4),
        encoding="utf-8",
    )
    (args.output_dir / "frame_map.json").write_text(
        json.dumps(frame_map, indent=4),
        encoding="utf-8",
    )
    (args.output_dir / "conversion_summary.json").write_text(
        json.dumps(
            {
                "num_frames": len(images),
                "model_format": model_format,
                "source_images_dir": str(args.images_dir),
                "source_model_dir": str(args.model_dir),
                "output_dir": str(args.output_dir),
                "input_resolution": {
                    "width": shared_camera.width,
                    "height": shared_camera.height,
                },
                "output_resolution": {
                    "width": output_width,
                    "height": output_height,
                },
                "shared_camera_model": shared_camera.model,
                "warnings": warnings,
            },
            indent=4,
        ),
        encoding="utf-8",
    )

    print(f"Converted {len(images)} registered COLMAP images to RegGS format.")
    print(f"Output scene directory: {args.output_dir}")
    print(f"Output image size: {output_width}x{output_height}")
    if warnings:
        for warning in warnings:
            print(f"Warning: {warning}")
    print("Generated files: images/, intrinsics.json, cameras.json, frame_map.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
