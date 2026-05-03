import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
STAGE_NAMES = ("init", "train", "render")

_COLMAP_LOADER_PATH = Path(__file__).resolve().parent / "scene" / "colmap_loader.py"
_COLMAP_SPEC = importlib.util.spec_from_file_location("instantsplat_colmap_loader", _COLMAP_LOADER_PATH)
if _COLMAP_SPEC is None or _COLMAP_SPEC.loader is None:
    raise ImportError(f"Cannot load COLMAP helper from {_COLMAP_LOADER_PATH}")
_COLMAP_LOADER = importlib.util.module_from_spec(_COLMAP_SPEC)
_COLMAP_SPEC.loader.exec_module(_COLMAP_LOADER)
qvec2rotmat = _COLMAP_LOADER.qvec2rotmat
read_extrinsics_text = _COLMAP_LOADER.read_extrinsics_text


@dataclass
class WindowJob:
    index: int
    name: str
    source_path: Path
    output_path: Path
    image_names: List[str]
    start_source_index: int
    end_source_index: int
    status: str = "pending"
    stages: Dict[str, int] = field(default_factory=dict)
    video_path: Optional[str] = None
    confidence_scores: List[float] = field(default_factory=list)
    train_l1: Optional[float] = None
    train_psnr: Optional[float] = None
    error: Optional[str] = None


def natural_key(path: Path) -> Tuple:
    parts = re.split(r"(\d+)", path.stem)
    return tuple(int(part) if part.isdigit() else part for part in parts) + (path.suffix.lower(),)


def resolve_image_dir(source: Path) -> Path:
    if source.name == "images" and source.is_dir():
        return source
    image_dir = source / "images"
    if image_dir.is_dir():
        return image_dir
    raise FileNotFoundError(f"Cannot find images directory under: {source}")


def list_images(image_dir: Path) -> List[Path]:
    images = [
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=natural_key)


def make_windows(
    images: Sequence[Path],
    window_size: int,
    stride: int,
    include_tail: bool,
) -> List[Tuple[int, List[Path]]]:
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if len(images) < window_size:
        raise ValueError(f"Need at least {window_size} images, found {len(images)}")

    windows: List[Tuple[int, List[Path]]] = []
    starts = list(range(0, len(images) - window_size + 1, stride))
    if include_tail and starts[-1] != len(images) - window_size:
        starts.append(len(images) - window_size)

    for start in starts:
        windows.append((start, list(images[start : start + window_size])))
    return windows


def safe_clean_dir(path: Path) -> None:
    if not path.exists():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def materialize_window_scene(
    window_root: Path,
    scene_stem: str,
    window_index: int,
    start_index: int,
    images: Sequence[Path],
    link_mode: str,
    overwrite: bool,
) -> WindowJob:
    scene_name = f"{scene_stem}_win{window_index:03d}"
    scene_path = window_root / scene_name
    image_path = scene_path / "images"
    if overwrite:
        safe_clean_dir(scene_path)
    image_path.mkdir(parents=True, exist_ok=True)

    for src in images:
        dst = image_path / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if link_mode == "copy":
            shutil.copy2(src, dst)
        else:
            os.symlink(src.resolve(), dst)

    source_info = {
        "window_index": window_index,
        "start_source_index": start_index,
        "end_source_index": start_index + len(images) - 1,
        "link_mode": link_mode,
        "images": [str(p) for p in images],
    }
    (scene_path / "source_info.json").write_text(
        json.dumps(source_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return WindowJob(
        index=window_index,
        name=scene_name,
        source_path=scene_path,
        output_path=Path(),
        image_names=[p.name for p in images],
        start_source_index=start_index,
        end_source_index=start_index + len(images) - 1,
    )


def build_commands(job: WindowJob, iterations: int, python_bin: str, scene_graph: str) -> List[Tuple[str, List[str]]]:
    n_views = len(job.image_names)
    return [
        (
            "init",
            [
                python_bin,
                "init_geo.py",
                "--source_path",
                str(job.source_path),
                "--model_path",
                str(job.output_path),
                "--n_views",
                str(n_views),
                "--focal_avg",
                "--co_vis_dsp",
                "--conf_aware_ranking",
                "--infer_video",
                "--scene_graph",
                scene_graph,
            ],
        ),
        (
            "train",
            [
                python_bin,
                "train.py",
                "-s",
                str(job.source_path),
                "-m",
                str(job.output_path),
                "--n_views",
                str(n_views),
                "--iterations",
                str(iterations),
                "--test_iterations",
                str(iterations),
                "--pp_optimizer",
                "--optim_pose",
            ],
        ),
        (
            "render",
            [
                python_bin,
                "render.py",
                "-s",
                str(job.source_path),
                "-m",
                str(job.output_path),
                "--n_views",
                str(n_views),
                "--iterations",
                str(iterations),
                "--infer_video",
            ],
        ),
    ]


def run_command(command: Sequence[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(command)}\n\n")
        log_file.flush()
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def parse_float_list(raw: str) -> List[float]:
    return [float(item) for item in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", raw)]


def parse_window_logs(job: WindowJob, log_dir: Path) -> None:
    init_log = log_dir / "init.log"
    if init_log.exists():
        text = init_log.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"Sorted average confidence scores:\s*\[([^\]]+)\]", text, re.S)
        if match:
            job.confidence_scores = parse_float_list(match.group(1))
        point_match = re.search(r"Number of points after downsampling:\s*(\d+)", text)
        if point_match:
            (job.output_path / "parsed_init_points.txt").write_text(point_match.group(1), encoding="utf-8")

    train_log = log_dir / "train.log"
    if train_log.exists():
        text = train_log.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"Evaluating train:\s*L1\s*([0-9.eE+-]+)\s*PSNR\s*([0-9.eE+-]+)", text)
        if matches:
            job.train_l1 = float(matches[-1][0])
            job.train_psnr = float(matches[-1][1])


def find_video(output_path: Path, iterations: int, n_views: int) -> Optional[str]:
    expected = output_path / f"interp/ours_{iterations}/interp_{n_views}_view.mp4"
    if expected.exists():
        return str(expected)
    videos = sorted(output_path.rglob("*.mp4"))
    return str(videos[0]) if videos else None


def read_camera_centers(sparse_dir: Path) -> Dict[str, np.ndarray]:
    images_txt = sparse_dir / "images.txt"
    if not images_txt.exists():
        return {}
    centers: Dict[str, np.ndarray] = {}
    for image in read_extrinsics_text(str(images_txt)).values():
        rotation = qvec2rotmat(image.qvec)
        center = -rotation.T @ image.tvec
        centers[image.name] = center.astype(float)
    return centers


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("Need at least 3 paired points with matching shape for Sim(3)")

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
    src_var = np.mean(np.sum(src_centered**2, axis=1))
    scale = float(np.sum(singular_values * np.diag(correction)) / src_var)
    translation = dst_mean - scale * rotation @ src_mean
    return scale, rotation, translation


def apply_similarity(point: np.ndarray, transform: Dict) -> np.ndarray:
    return transform["scale"] * np.asarray(transform["rotation"]) @ point + np.asarray(transform["translation"])


def parse_external_poses(path: Optional[Path]) -> Dict[str, np.ndarray]:
    if path is None:
        return {}
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "poses" in data:
            rows = data["poses"]
            return {row["image"]: np.asarray(row["position"], dtype=float) for row in rows}
        return {name: np.asarray(value, dtype=float) for name, value in data.items()}

    poses: Dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            image = row.get("image") or row.get("name") or row.get("filename")
            if image is None:
                raise ValueError("External pose CSV must include an image/name/filename column")
            poses[image] = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=float)
    return poses


def build_alignment(
    jobs: Sequence[WindowJob],
    report_root: Path,
    external_poses: Dict[str, np.ndarray],
) -> Dict:
    report_root.mkdir(parents=True, exist_ok=True)
    alignment = {
        "mode": "external_poses" if external_poses else "overlap_chain",
        "windows": [],
        "notes": [],
    }
    previous_centers: Optional[Dict[str, np.ndarray]] = None
    previous_transform: Optional[Dict] = None

    for job in jobs:
        sparse_dir = job.source_path / f"sparse_{len(job.image_names)}" / "0"
        centers = read_camera_centers(sparse_dir)
        record = {
            "name": job.name,
            "status": "missing_sparse" if not centers else "aligned",
            "shared_images": [],
            "scale": 1.0,
            "rotation": np.eye(3).tolist(),
            "translation": [0.0, 0.0, 0.0],
        }

        if not centers:
            alignment["windows"].append(record)
            continue

        if external_poses:
            shared = sorted(set(centers) & set(external_poses))
            record["shared_images"] = shared
            if len(shared) >= 3:
                src = np.stack([centers[name] for name in shared])
                dst = np.stack([external_poses[name] for name in shared])
                scale, rotation, translation = umeyama_similarity(src, dst)
                record.update(
                    {
                        "scale": scale,
                        "rotation": rotation.tolist(),
                        "translation": translation.tolist(),
                    }
                )
            else:
                record["status"] = "insufficient_external_pose_overlap"
        elif previous_centers is None or previous_transform is None:
            record["status"] = "reference"
        else:
            shared = sorted(set(centers) & set(previous_centers))
            record["shared_images"] = shared
            if len(shared) >= 3:
                src = np.stack([centers[name] for name in shared])
                dst = np.stack([apply_similarity(previous_centers[name], previous_transform) for name in shared])
                scale, rotation, translation = umeyama_similarity(src, dst)
                record.update(
                    {
                        "scale": scale,
                        "rotation": rotation.tolist(),
                        "translation": translation.tolist(),
                    }
                )
            else:
                record["status"] = "insufficient_overlap"

        previous_centers = centers
        previous_transform = {
            "scale": record["scale"],
            "rotation": record["rotation"],
            "translation": record["translation"],
        }
        alignment["windows"].append(record)

    camera_rows = []
    for job, window_alignment in zip(jobs, alignment["windows"]):
        centers = read_camera_centers(job.source_path / f"sparse_{len(job.image_names)}" / "0")
        if not centers or window_alignment["status"] in {"missing_sparse", "insufficient_overlap", "insufficient_external_pose_overlap"}:
            continue
        transform = {
            "scale": window_alignment["scale"],
            "rotation": window_alignment["rotation"],
            "translation": window_alignment["translation"],
        }
        for image_name, center in centers.items():
            global_center = apply_similarity(center, transform)
            camera_rows.append([job.name, image_name, *global_center.tolist()])

    (report_root / "global_alignment.json").write_text(
        json.dumps(alignment, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (report_root / "global_camera_path.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["window", "image", "x", "y", "z"])
        writer.writerows(camera_rows)

    return alignment


def write_reports(jobs: Sequence[WindowJob], report_root: Path, alignment: Dict) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "windows": [asdict(job) for job in jobs],
        "alignment": alignment,
    }
    (report_root / "global_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    with (report_root / "quality_report.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "window",
                "status",
                "frames",
                "frame_start",
                "frame_end",
                "mean_confidence",
                "train_l1",
                "train_psnr",
                "video_path",
                "error",
            ]
        )
        for job in jobs:
            mean_conf = np.mean(job.confidence_scores) if job.confidence_scores else ""
            writer.writerow(
                [
                    job.name,
                    job.status,
                    len(job.image_names),
                    job.image_names[0],
                    job.image_names[-1],
                    mean_conf,
                    job.train_l1 or "",
                    job.train_psnr or "",
                    job.video_path or "",
                    job.error or "",
                ]
            )

    lines = [
        "# Long Sequence InstantSplat Report",
        "",
        "| Window | Status | Frames | Mean Confidence | Train L1 | Train PSNR | Video |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for job in jobs:
        mean_conf = f"{np.mean(job.confidence_scores):.4f}" if job.confidence_scores else ""
        video = job.video_path or ""
        lines.append(
            f"| {job.name} | {job.status} | {job.image_names[0]}..{job.image_names[-1]} | "
            f"{mean_conf} | {job.train_l1 or ''} | {job.train_psnr or ''} | {video} |"
        )
    lines.extend(
        [
            "",
            "## Alignment",
            "",
            f"Mode: `{alignment.get('mode')}`",
            "",
            "See `global_alignment.json` and `global_camera_path.csv` for window transforms and camera centers.",
        ]
    )
    (report_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_jobs(
    jobs: Sequence[WindowJob],
    iterations: int,
    instant_splat_root: Path,
    python_bin: str,
    scene_graph: str,
    skip_existing: bool,
) -> None:
    for job in jobs:
        print(f"\n=== Window {job.index}: {job.name} ({len(job.image_names)} views) ===")
        job.output_path.mkdir(parents=True, exist_ok=True)
        commands = build_commands(job, iterations, python_bin, scene_graph)
        job.status = "running"
        for stage, command in commands:
            if skip_existing and stage == "render":
                existing_video = find_video(job.output_path, iterations, len(job.image_names))
                if existing_video:
                    job.stages[stage] = 0
                    job.video_path = existing_video
                    continue
            log_path = job.output_path / "logs" / f"{stage}.log"
            return_code = run_command(command, instant_splat_root, log_path)
            job.stages[stage] = return_code
            if return_code != 0:
                job.status = "failed"
                job.error = f"{stage} failed with exit code {return_code}; see {log_path}"
                break
        parse_window_logs(job, job.output_path / "logs")
        job.video_path = find_video(job.output_path, iterations, len(job.image_names))
        if job.status != "failed":
            job.status = "complete" if job.video_path else "rendered_without_video"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run InstantSplat on overlapping windows from a long driving sequence."
    )
    parser.add_argument("--source_path", required=True, type=Path, help="Scene directory or images directory.")
    parser.add_argument("--workspace_root", type=Path, default=Path("assets/long_sequence_windows"))
    parser.add_argument("--output_root", type=Path, default=Path("output_infer/long_sequence"))
    parser.add_argument("--report_root", type=Path, default=Path("output_infer/long_sequence_report"))
    parser.add_argument("--window_size", type=int, default=6)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--python", default="python")
    parser.add_argument(
        "--scene_graph",
        default="complete",
        help="MASt3R/DUSt3R image graph used inside each window.",
    )
    parser.add_argument("--link_mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--external_poses", type=Path, default=None)
    parser.add_argument("--no_include_tail", action="store_true")
    parser.add_argument("--overwrite_windows", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--prepare_only", action="store_true", help="Only create window scenes and reports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instant_splat_root = Path(__file__).resolve().parent
    image_dir = resolve_image_dir(args.source_path.resolve())
    images = list_images(image_dir)
    if not images:
        raise SystemExit(f"No images found in {image_dir}")

    resolved_source = args.source_path.resolve()
    scene_stem = resolved_source.parent.name if resolved_source.name == "images" else resolved_source.name
    workspace_root = (instant_splat_root / args.workspace_root).resolve()
    output_root = (instant_splat_root / args.output_root).resolve()
    report_root = (instant_splat_root / args.report_root).resolve()

    windows = make_windows(
        images,
        args.window_size,
        args.stride,
        include_tail=not args.no_include_tail,
    )
    jobs: List[WindowJob] = []
    for window_index, (start_index, window_images) in enumerate(windows):
        job = materialize_window_scene(
            workspace_root,
            scene_stem,
            window_index,
            start_index,
            window_images,
            args.link_mode,
            args.overwrite_windows,
        )
        job.output_path = output_root / job.name
        jobs.append(job)

    if not args.prepare_only:
        run_jobs(
            jobs,
            args.iterations,
            instant_splat_root,
            args.python,
            args.scene_graph,
            args.skip_existing,
        )

    external_poses = parse_external_poses(args.external_poses)
    alignment = build_alignment(jobs, report_root, external_poses)
    write_reports(jobs, report_root, alignment)

    print(f"\nPrepared {len(jobs)} windows from {len(images)} images.")
    print(f"Window scenes: {workspace_root}")
    print(f"Outputs: {output_root}")
    print(f"Report: {report_root / 'summary.md'}")


if __name__ == "__main__":
    main()
