#!/usr/bin/env python3
"""Run short-window pseudo-view validation experiments for InstantSplat.

The script creates contiguous windows from an existing sparse training split,
runs a short InstantSplat reconstruction for each window, renders pseudo views,
and optionally enhances/fuses them with Difix3D.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _run(cmd: list[str], cwd: Path, log_file: Path, dry_run: bool = False) -> None:
    text = shlex.join(cmd)
    header = f"\n[{dt.datetime.now().isoformat(timespec='seconds')}] $ {text}\n"
    print(header, flush=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log:
        log.write(header)
        log.flush()
        if dry_run:
            return
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=_subprocess_env(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
        if code != 0:
            raise subprocess.CalledProcessError(code, cmd)


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.pop("MKL_SERVICE_FORCE_INTEL", None)
    itt_stub = Path(sys.prefix) / "lib" / "libittnotify_stub.so"
    if itt_stub.exists():
        current_preload = env.get("LD_PRELOAD", "")
        preload_parts = [p for p in current_preload.split(":") if p]
        if str(itt_stub) not in preload_parts:
            preload_parts.insert(0, str(itt_stub))
            env["LD_PRELOAD"] = ":".join(preload_parts)
    return env


def _prepare_window_source(
    source_images_dir: Path,
    window_source: Path,
    basenames: list[str],
    mode: str,
) -> None:
    images_dir = window_source / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for name in basenames:
        src = source_images_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        dst = images_dir / name
        if dst.exists() or dst.is_symlink():
            continue
        if mode == "copy":
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src)


def _window_specs(train_basenames: list[str], train_indices: list[int], window_size: int, stride: int, max_windows: int):
    specs = []
    for start in range(0, len(train_basenames) - window_size + 1, stride):
        stop = start + window_size
        specs.append(
            {
                "window_id": len(specs),
                "start_train_position": start,
                "end_train_position": stop - 1,
                "train_positions": list(range(start, stop)),
                "train_basenames": train_basenames[start:stop],
                "train_indices": train_indices[start:stop],
            }
        )
        if max_windows > 0 and len(specs) >= max_windows:
            break
    return specs


def _skip(path: Path, skip_existing: bool, label: str) -> bool:
    if skip_existing and path.exists():
        print(f"[skip] {label}: {path}")
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--source_path", type=Path, required=True)
    parser.add_argument("--base_split_manifest", type=Path, required=True)
    parser.add_argument("-o", "--output_root", type=Path, required=True)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_windows", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pseudo_per_interval", type=int, default=3)
    parser.add_argument("-r", "--resolution", type=int, default=1)
    parser.add_argument("--scene_graph", type=str, default="complete")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_init_points", type=int, default=0)
    parser.add_argument("--point_sampling", type=str, default="grid_uniform_confidence")
    parser.add_argument("--sampling_grid_size", type=int, default=24)
    parser.add_argument("--min_point_distance_px", type=float, default=12.0)
    parser.add_argument("--point_conf_threshold", type=float, default=0.0)
    parser.add_argument("--disable_co_vis_dsp", action="store_true")
    parser.add_argument("--run_difix", action="store_true")
    parser.add_argument("--run_confidence_fusion", action="store_true")
    parser.add_argument("--side_conf_threshold", type=float, default=0.15)
    parser.add_argument("--agreement_threshold", type=float, default=0.25)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--link_mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--python_bin", type=Path, default=Path(sys.executable))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source_path = args.source_path.resolve()
    source_images_dir = source_path / "images"
    output_root = args.output_root.resolve()
    models_root = output_root / "models"
    sources_root = output_root / "sources"
    output_root.mkdir(parents=True, exist_ok=True)

    base_split = _load_json(args.base_split_manifest.resolve())
    train_basenames = list(base_split["train_basenames"])
    train_indices = list(base_split.get("train_indices", range(len(train_basenames))))
    specs = _window_specs(train_basenames, train_indices, args.window_size, args.stride, args.max_windows)
    if not specs:
        raise ValueError("No windows generated. Check --window_size and --stride.")

    master_manifest = {
        "schema": "instantsplat_window_pseudo_validation_v1",
        "source_path": str(source_path),
        "base_split_manifest": str(args.base_split_manifest.resolve()),
        "window_size": args.window_size,
        "stride": args.stride,
        "iterations": args.iterations,
        "pseudo_per_interval": args.pseudo_per_interval,
        "scene_graph": args.scene_graph,
        "max_init_points": args.max_init_points,
        "co_vis_dsp": not args.disable_co_vis_dsp,
        "windows": [],
    }

    for spec in specs:
        wid = int(spec["window_id"])
        start = int(spec["start_train_position"])
        end = int(spec["end_train_position"])
        tag = f"window_{wid:03d}_trainpos_{start:02d}_{end:02d}"
        window_source = sources_root / tag
        window_model = models_root / tag
        log_file = window_model / "window_pipeline.log"
        pseudo_dir = window_model / "pseudo_views"

        _prepare_window_source(source_images_dir, window_source, spec["train_basenames"], args.link_mode)
        window_manifest = {
            "schema": "instantsplat_window_source_v1",
            "window_tag": tag,
            **spec,
            "source_path": str(source_path),
            "window_source": str(window_source),
            "window_model": str(window_model),
        }
        _write_json(window_source / "window_manifest.json", window_manifest)
        _write_json(
            window_model / "split_manifest.json",
            {
                "schema": "instantsplat_train_test_split_v1",
                "n_train": args.window_size,
                "n_test": 0,
                "train_indices": spec["train_indices"],
                "test_indices": [],
                "train_basenames": spec["train_basenames"],
                "test_basenames": [],
                "window_train_positions": spec["train_positions"],
            },
        )

        init_done = window_source / f"sparse_{args.window_size}" / "0" / "points3D.ply"
        if not _skip(init_done, args.skip_existing, f"{tag} init_geo"):
            init_cmd = [
                str(args.python_bin),
                str(repo_root / "init_geo.py"),
                "-s",
                str(window_source),
                "-m",
                str(window_model),
                "--n_views",
                str(args.window_size),
                "--infer_video",
                "--focal_avg",
                "--conf_aware_ranking",
                "--scene_graph",
                args.scene_graph,
                "--image_size",
                str(args.image_size),
                "--max_init_points",
                str(args.max_init_points),
                "--point_sampling",
                args.point_sampling,
                "--sampling_grid_size",
                str(args.sampling_grid_size),
                "--min_point_distance_px",
                str(args.min_point_distance_px),
                "--point_conf_threshold",
                str(args.point_conf_threshold),
            ]
            if not args.disable_co_vis_dsp:
                init_cmd.append("--co_vis_dsp")
            _run(init_cmd, repo_root, log_file, args.dry_run)

        point_cloud_done = window_model / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"
        if not _skip(point_cloud_done, args.skip_existing, f"{tag} train"):
            train_cmd = [
                str(args.python_bin),
                str(repo_root / "train.py"),
                "-s",
                str(window_source),
                "-m",
                str(window_model),
                "-r",
                str(args.resolution),
                "--n_views",
                str(args.window_size),
                "--iterations",
                str(args.iterations),
                "--test_iterations",
                str(args.iterations),
                "--save_iterations",
                str(args.iterations),
            ]
            _run(train_cmd, repo_root, log_file, args.dry_run)

        pose_done = window_model / "pose" / f"ours_{args.iterations}" / "pose_optimized.npy"
        if not _skip(pose_done, args.skip_existing, f"{tag} render-train"):
            render_cmd = [
                str(args.python_bin),
                str(repo_root / "render.py"),
                "-s",
                str(window_source),
                "-m",
                str(window_model),
                "-r",
                str(args.resolution),
                "--n_views",
                str(args.window_size),
                "--iterations",
                str(args.iterations),
            ]
            _run(render_cmd, repo_root, log_file, args.dry_run)

        pseudo_manifest = pseudo_dir / "pseudo_manifest.json"
        if not _skip(pseudo_manifest, args.skip_existing, f"{tag} make-pseudo-poses"):
            make_pose_cmd = [
                str(args.python_bin),
                str(repo_root / "tools" / "make_pseudo_poses.py"),
                "-m",
                str(window_model),
                "--iteration",
                str(args.iterations),
                "--pseudo_per_interval",
                str(args.pseudo_per_interval),
                "--output_dir",
                str(pseudo_dir),
                "--visualize",
            ]
            _run(make_pose_cmd, repo_root, log_file, args.dry_run)

        rendered_manifest = pseudo_dir / "pseudo_manifest_rendered.json"
        if not _skip(rendered_manifest, args.skip_existing, f"{tag} render-pseudo"):
            render_pseudo_cmd = [
                str(args.python_bin),
                str(repo_root / "tools" / "render_pseudo_views.py"),
                "-s",
                str(window_source),
                "-m",
                str(window_model),
                "-r",
                str(args.resolution),
                "--n_views",
                str(args.window_size),
                "--iteration",
                str(args.iterations),
                "--pseudo_manifest",
                str(pseudo_manifest),
                "--output_dir",
                str(pseudo_dir),
            ]
            _run(render_pseudo_cmd, repo_root, log_file, args.dry_run)

        fused_dir = None
        if args.run_difix:
            difix_dir = pseudo_dir / "difix_enhanced"
            difix_cmd = [
                str(args.python_bin),
                str(repo_root / "tools" / "enhance_pseudo_with_difix.py"),
                "--pseudo_manifest",
                str(rendered_manifest),
                "--source_images_dir",
                str(window_source / "images"),
                "--output_dir",
                str(difix_dir),
                "--skip_existing",
            ]
            _run(difix_cmd, repo_root, log_file, args.dry_run)

            if args.run_confidence_fusion:
                fused_dir = pseudo_dir / f"difix_trinary_s{args.side_conf_threshold:.2f}_a{args.agreement_threshold:.2f}".replace(".", "")
                fusion_cmd = [
                    str(args.python_bin),
                    str(repo_root / "tools" / "build_pseudo_confidence.py"),
                    "--pseudo_manifest",
                    str(rendered_manifest),
                    "--source_images_dir",
                    str(window_source),
                    "--enhanced_left_dir",
                    str(difix_dir / "enhanced_left"),
                    "--enhanced_right_dir",
                    str(difix_dir / "enhanced_right"),
                    "--output_dir",
                    str(fused_dir),
                    "--mask_mode",
                    "trinary",
                    "--side_conf_threshold",
                    str(args.side_conf_threshold),
                    "--agreement_threshold",
                    str(args.agreement_threshold),
                    "--set_loss_weight_from_mask",
                ]
                _run(fusion_cmd, repo_root, log_file, args.dry_run)

        master_manifest["windows"].append(
            {
                **window_manifest,
                "pseudo_manifest_rendered": str(rendered_manifest),
                "raw_dir": str(pseudo_dir / "raw"),
                "difix_fused_dir": str(fused_dir) if fused_dir is not None else "",
                "log_file": str(log_file),
            }
        )
        _write_json(output_root / "window_validation_manifest.json", master_manifest)

    print(f"[window] windows: {len(specs)}")
    print(f"[window] wrote: {output_root / 'window_validation_manifest.json'}")


if __name__ == "__main__":
    main()
