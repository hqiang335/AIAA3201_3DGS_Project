#!/usr/bin/env python3
"""Fixed FRONT 20-view evaluation protocol for InstantSplat-style methods.

The baseline subcommand runs the native InstantSplat eval chain:
    init_geo -> train -> render.py --eval -> image metrics.

The existing subcommand adapts an already trained model/scene to the same
held-out test split, then runs the same render.py --eval and image metrics.
This is intended for later long-sequence Gaussian-submap methods.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from arguments import default_train_learning_rates

os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ.pop("MKL_SERVICE_FORCE_INTEL", None)

import numpy as np

from scene.colmap_loader import (
    Camera,
    Image,
    qvec2rotmat,
    read_extrinsics_text,
    read_intrinsics_text,
    rotmat2qvec,
    write_cameras_text,
    write_images_text,
)
from utils.camera_utils import generate_interpolated_path
from utils.sfm_utils import get_sorted_image_files, split_train_eval_views


DEFAULT_SOURCE_PATH = Path("/root/autodl-fs/405841/FRONT")
DEFAULT_MODEL_PATH = Path("/root/autodl-fs/405841/eval_protocol/front20_native_complete_1000")
DEFAULT_SCENE_GRAPH = "complete"
DEFAULT_N_TRAIN = 20
DEFAULT_N_TEST = 12
DEFAULT_ITERATIONS = 1000
DEFAULT_RESOLUTION = 1
DEFAULT_OPTIM_TEST_POSE_ITER = 500

_TRAIN_LR_DEFAULTS = default_train_learning_rates()


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    env.pop("MKL_SERVICE_FORCE_INTEL", None)
    return env


def _run(cmd: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=_subprocess_env())


def _source_image_dir(source_path: Path) -> Path:
    image_dir = source_path / "images"
    if image_dir.is_dir():
        return image_dir
    return source_path


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def _w2c_from_image_record(record: Image) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = qvec2rotmat(record.qvec)
    pose[:3, 3] = np.asarray(record.tvec, dtype=float)
    return pose


def _interpolate_test_w2c(train_w2c: np.ndarray, n_test: int) -> np.ndarray:
    n_train = train_w2c.shape[0]
    if n_train < n_test:
        n_interp = (n_test // (n_train - 1)) + 1
        chunks = []
        for i in range(n_train - 1):
            chunks.append(generate_interpolated_path(poses=train_w2c[i:i + 2], n_interp=n_interp))
        all_inter_pose = np.concatenate(chunks, axis=0)
        all_inter_pose = np.concatenate([all_inter_pose, train_w2c[-1][:3, :].reshape(1, 3, 4)], axis=0)
        indices = np.linspace(0, all_inter_pose.shape[0] - 1, n_test, dtype=int)
        sampled = all_inter_pose[indices]
        out = []
        for pose in sampled:
            tmp = np.eye(4, dtype=float)
            tmp[:3, :3] = pose[:3, :3]
            tmp[:3, 3] = pose[:3, 3]
            out.append(tmp)
        return np.stack(out, axis=0)

    indices = np.linspace(0, n_train - 1, n_test, dtype=int)
    return train_w2c[indices]


def prepare_eval_scene(
    full_source_path: Path,
    eval_scene_path: Path,
    n_train: int,
    n_test: int,
    image_mode: str,
    strict_train_split: bool,
) -> dict:
    image_files, _ = get_sorted_image_files(str(_source_image_dir(full_source_path)))
    train_files, test_files, train_indices, test_indices = split_train_eval_views(
        image_files,
        n_train=n_train,
        n_test=n_test,
        verbose=True,
    )
    train_basenames = [Path(path).name for path in train_files]
    test_basenames = [Path(path).name for path in test_files]

    scene_images = eval_scene_path / "images"
    for path in train_files + test_files:
        src = Path(path)
        _link_or_copy(src, scene_images / src.name, image_mode)

    sparse0 = eval_scene_path / f"sparse_{n_train}" / "0"
    sparse1 = eval_scene_path / f"sparse_{n_train}" / "1"
    if not (sparse0 / "images.txt").exists():
        raise FileNotFoundError(f"Missing train sparse images.txt: {sparse0 / 'images.txt'}")
    if not (sparse0 / "cameras.txt").exists():
        raise FileNotFoundError(f"Missing train sparse cameras.txt: {sparse0 / 'cameras.txt'}")

    train_records = read_extrinsics_text(sparse0 / "images.txt")
    cameras = read_intrinsics_text(sparse0 / "cameras.txt")
    records_by_name = {record.name: record for record in train_records.values()}
    sparse_train_names = sorted(records_by_name)
    expected = set(train_basenames)
    actual = set(sparse_train_names)
    mismatch = {
        "missing_expected_train": sorted(expected - actual),
        "unexpected_train": sorted(actual - expected),
    }
    if strict_train_split and (mismatch["missing_expected_train"] or mismatch["unexpected_train"]):
        raise RuntimeError(
            "Existing scene was not trained with the fixed eval split. "
            f"Mismatch: {json.dumps(mismatch, indent=2)}"
        )

    ordered_train_names = [name for name in train_basenames if name in records_by_name]
    if len(ordered_train_names) < 2:
        raise RuntimeError("Need at least two train poses to initialize test poses")
    train_w2c = np.stack([_w2c_from_image_record(records_by_name[name]) for name in ordered_train_names], axis=0)
    test_w2c = _interpolate_test_w2c(train_w2c, len(test_basenames))

    first_camera = cameras[sorted(cameras)[0]]
    test_cameras = {}
    test_images = {}
    for idx, (name, pose) in enumerate(zip(test_basenames, test_w2c), start=1):
        test_cameras[idx] = Camera(
            id=idx,
            model=first_camera.model,
            width=first_camera.width,
            height=first_camera.height,
            params=np.asarray(first_camera.params, dtype=float),
        )
        test_images[idx] = Image(
            id=idx,
            qvec=rotmat2qvec(pose[:3, :3]),
            tvec=pose[:3, 3],
            camera_id=idx,
            name=name,
            xys=np.empty((0, 2), dtype=float),
            point3D_ids=np.empty((0,), dtype=np.int64),
        )

    sparse1.mkdir(parents=True, exist_ok=True)
    write_cameras_text(test_cameras, sparse1 / "cameras.txt")
    write_images_text(test_images, sparse1 / "images.txt")

    manifest = {
        "schema": "front20_eval_protocol_v1",
        "full_source_path": str(full_source_path),
        "eval_scene_path": str(eval_scene_path),
        "n_train": n_train,
        "n_test": n_test,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_basenames": train_basenames,
        "test_basenames": test_basenames,
        "strict_train_split": strict_train_split,
        "train_mismatch": mismatch,
        "test_pose_initializer": "InstantSplat init_geo interpolation rule from method train sparse poses",
    }
    (eval_scene_path / "eval_protocol_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote eval sparse cameras to {sparse1}")
    return manifest


def run_baseline(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(root / "run_eval_pipeline.py"),
        "-s",
        str(args.source_path.resolve()),
        "-m",
        str(args.model_path.resolve()),
        "--n_train",
        str(args.n_train),
        "--n_test",
        str(args.n_test),
        "--iterations",
        str(args.iterations),
        "--resolution",
        str(args.resolution),
        "--optim_test_pose_iter",
        str(args.optim_test_pose_iter),
        "--skip_metrics",
        "--position_lr_init",
        str(args.position_lr_init),
        "--position_lr_final",
        str(args.position_lr_final),
        "--feature_lr",
        str(args.feature_lr),
        "--scaling_lr",
        str(args.scaling_lr),
        "--rotation_lr",
        str(args.rotation_lr),
    ]
    if args.scene_graph:
        cmd.extend(["--scene_graph", args.scene_graph])
    _run(cmd, root)
    if not args.skip_native_metrics:
        _run([
            sys.executable,
            str(root / "metrics.py"),
            "-s",
            str(args.source_path.resolve()),
            "-m",
            str(args.model_path.resolve()),
            "--n_views",
            str(args.n_train),
            "--n_test",
            str(args.n_test),
        ], root)
    if not args.skip_image_metrics:
        _run([
            sys.executable,
            str(root / "eval_image_metrics.py"),
            "-m",
            str(args.model_path.resolve()),
            "--split",
            "test",
            "--iteration",
            str(args.iterations),
        ], root)


def run_existing(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parent
    if not args.skip_prepare:
        prepare_eval_scene(
            args.source_path.resolve(),
            args.scene_path.resolve(),
            args.n_train,
            args.n_test,
            args.image_mode,
            args.strict_train_split,
        )
    _run([
        sys.executable,
        str(root / "render.py"),
        "-s",
        str(args.scene_path.resolve()),
        "-m",
        str(args.model_path.resolve()),
        "-r",
        str(args.resolution),
        "--n_views",
        str(args.n_train),
        "--iterations",
        str(args.iterations),
        "--eval",
        "--optim_test_pose_iter",
        str(args.optim_test_pose_iter),
    ], root)
    if not args.skip_native_metrics:
        _run([
            sys.executable,
            str(root / "metrics.py"),
            "-s",
            str(args.source_path.resolve()),
            "-m",
            str(args.model_path.resolve()),
            "--n_views",
            str(args.n_train),
            "--n_test",
            str(args.n_test),
        ], root)
    if not args.skip_image_metrics:
        _run([
            sys.executable,
            str(root / "eval_image_metrics.py"),
            "-m",
            str(args.model_path.resolve()),
            "--split",
            "test",
            "--iteration",
            str(args.iterations),
        ], root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    baseline = sub.add_parser("baseline", help="Run native InstantSplat baseline eval")
    baseline.add_argument("-s", "--source_path", type=Path, default=DEFAULT_SOURCE_PATH)
    baseline.add_argument("-m", "--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    baseline.add_argument("--n_train", type=int, default=DEFAULT_N_TRAIN)
    baseline.add_argument("--n_test", type=int, default=DEFAULT_N_TEST)
    baseline.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    baseline.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    baseline.add_argument("--optim_test_pose_iter", type=int, default=DEFAULT_OPTIM_TEST_POSE_ITER)
    baseline.add_argument("--scene_graph", type=str, default=DEFAULT_SCENE_GRAPH)
    baseline.add_argument(
        "--position_lr_init",
        type=float,
        default=_TRAIN_LR_DEFAULTS["position_lr_init"],
        help="train.py Gaussian position LR (start); default matches OptimizationParams.",
    )
    baseline.add_argument(
        "--position_lr_final",
        type=float,
        default=_TRAIN_LR_DEFAULTS["position_lr_final"],
        help="train.py Gaussian position LR (end); default matches OptimizationParams.",
    )
    baseline.add_argument(
        "--feature_lr",
        type=float,
        default=_TRAIN_LR_DEFAULTS["feature_lr"],
        help="train.py SH feature LR; default matches OptimizationParams.",
    )
    baseline.add_argument(
        "--scaling_lr",
        type=float,
        default=_TRAIN_LR_DEFAULTS["scaling_lr"],
        help="train.py scaling LR; default matches OptimizationParams.",
    )
    baseline.add_argument(
        "--rotation_lr",
        type=float,
        default=_TRAIN_LR_DEFAULTS["rotation_lr"],
        help="train.py rotation LR; default matches OptimizationParams.",
    )
    baseline.add_argument("--skip_native_metrics", action="store_true")
    baseline.add_argument("--skip_image_metrics", action="store_true")
    baseline.set_defaults(func=run_baseline)

    existing = sub.add_parser("existing", help="Evaluate an existing InstantSplat-style model on the fixed split")
    existing.add_argument("-s", "--source_path", type=Path, default=DEFAULT_SOURCE_PATH, help="Full original source with images/")
    existing.add_argument("--scene_path", type=Path, required=True, help="Scene used by the method, containing sparse_N/0")
    existing.add_argument("-m", "--model_path", type=Path, required=True, help="Trained model path to render")
    existing.add_argument("--n_train", type=int, default=DEFAULT_N_TRAIN)
    existing.add_argument("--n_test", type=int, default=DEFAULT_N_TEST)
    existing.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    existing.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    existing.add_argument("--optim_test_pose_iter", type=int, default=DEFAULT_OPTIM_TEST_POSE_ITER)
    existing.add_argument("--image_mode", choices=["symlink", "copy"], default="symlink")
    existing.add_argument("--skip_prepare", action="store_true")
    existing.add_argument("--skip_native_metrics", action="store_true")
    existing.add_argument("--skip_image_metrics", action="store_true")
    existing.add_argument("--strict_train_split", action="store_true")
    existing.set_defaults(func=run_existing)
    return parser


def main() -> None:
    if len(sys.argv) == 1:
        sys.argv.append("baseline")
    elif sys.argv[1].startswith("-"):
        sys.argv.insert(1, "baseline")
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        raise SystemExit(2)
    args.func(args)


if __name__ == "__main__":
    main()
