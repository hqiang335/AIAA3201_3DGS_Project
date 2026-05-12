#!/usr/bin/env python3
# Copyright / usage: convenience driver for InstantSplat eval-style runs with explicit train/test counts.
"""
Run init_geo → train → [init_test_pose] → render (train) → render (--eval) → metrics.

Train / test framing matches ``split_train_eval_views`` (``init_geo.py --n_views / --n_test``):
  - Test: ``n_test`` indices via linspace on sorted images between frames 1 .. N-2 (InstantSplat default).
  - Train: ``n_train`` uniformly sampled indices among all frames *not* in the test set.

Writes ``<model_path>/split_manifest.json`` during ``init_geo`` (training indices, basenames, etc.).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys
from pathlib import Path

from arguments import default_train_learning_rates

_TRAIN_LR_DEFAULTS = default_train_learning_rates()


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


def _run(py: Path, args: list[str], cwd: Path, log_file: Path | None = None) -> None:
    cmd = [sys.executable, str(py)] + args
    cmd_text = shlex.join(cmd)
    header = f"\n[{dt.datetime.now().isoformat(timespec='seconds')}] $ {cmd_text}\n"
    print(header, flush=True)

    log_handle = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_file, "a", encoding="utf-8")
        log_handle.write(header)
        log_handle.flush()

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=_subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            if log_handle is not None:
                log_handle.write(line)
                log_handle.flush()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
    finally:
        if log_handle is not None:
            log_handle.close()


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--source_path", type=Path, required=True)
    parser.add_argument("-m", "--model_path", type=Path, required=True)
    parser.add_argument(
        "--n_train",
        type=int,
        required=True,
        help="Number of training views (passed to all stages as --n_views).",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=12,
        help="Number of held-out test views (linspace between sorted frames 1..N-2). Default: 12.",
    )
    parser.add_argument("-i", "--iterations", type=int, default=1000)
    parser.add_argument("-r", "--resolution", type=int, default=1, help="Resolution scale for train/render (e.g. 1, 2, 4).")
    parser.add_argument(
        "--optim_test_pose_iter",
        type=int,
        default=500,
        help="Per-view test pose refinement steps in render.py --eval.",
    )
    parser.add_argument("--run_init_test_pose", action="store_true", help="Run init_test_pose.py after training.")
    parser.add_argument("--skip_init", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_render_train", action="store_true")
    parser.add_argument("--skip_render_eval", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--no_pp_optimizer", action="store_true")
    parser.add_argument("--no_optim_pose", action="store_true")
    parser.add_argument(
        "--use_densification",
        action="store_true",
        help="Enable 3DGS clone/split/prune/opacity reset during train.py.",
    )
    parser.add_argument(
        "--scene_graph",
        type=str,
        default="complete",
        help="MASt3R image graph passed to init_geo.py",
    )
    parser.add_argument(
        "--max_init_points",
        type=int,
        default=0,
        help="Maximum points written by init_geo.py after co-visible filtering. 0 disables this cap.",
    )
    parser.add_argument(
        "--point_sampling",
        type=str,
        default="grid_uniform_confidence",
        choices=["grid_uniform_confidence", "confidence_random"],
        help="Point sampling strategy used by init_geo.py when --max_init_points is positive.",
    )
    parser.add_argument(
        "--sampling_grid_size",
        type=int,
        default=24,
        help="Grid size per image side for grid_uniform_confidence initialization sampling.",
    )
    parser.add_argument(
        "--min_point_distance_px",
        type=float,
        default=12.0,
        help="Minimum pixel distance between selected initialization points from the same frame.",
    )
    parser.add_argument(
        "--point_conf_threshold",
        type=float,
        default=0.0,
        help="Optional confidence threshold before max-point sampling. 0 disables thresholding.",
    )
    parser.add_argument(
        "--sampling_seed",
        type=int,
        default=42,
        help="Random seed for confidence_random initialization sampling.",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Pipeline log file. Default: <model_path>/pipeline.log",
    )
    parser.add_argument(
        "--position_lr_init",
        type=float,
        default=_TRAIN_LR_DEFAULTS["position_lr_init"],
        help="Forwarded to train.py (OptimizationParams default unless overridden).",
    )
    parser.add_argument(
        "--position_lr_final",
        type=float,
        default=_TRAIN_LR_DEFAULTS["position_lr_final"],
        help="Forwarded to train.py (OptimizationParams default unless overridden).",
    )
    parser.add_argument(
        "--feature_lr",
        type=float,
        default=_TRAIN_LR_DEFAULTS["feature_lr"],
        help="Forwarded to train.py (OptimizationParams default unless overridden).",
    )
    parser.add_argument(
        "--opacity_lr",
        type=float,
        default=_TRAIN_LR_DEFAULTS["opacity_lr"],
        help="Forwarded to train.py (OptimizationParams default unless overridden).",
    )
    parser.add_argument(
        "--scaling_lr",
        type=float,
        default=_TRAIN_LR_DEFAULTS["scaling_lr"],
        help="Forwarded to train.py (OptimizationParams default unless overridden).",
    )
    parser.add_argument(
        "--rotation_lr",
        type=float,
        default=_TRAIN_LR_DEFAULTS["rotation_lr"],
        help="Forwarded to train.py (OptimizationParams default unless overridden).",
    )
    parser.add_argument(
        "--percent_dense",
        type=float,
        default=_TRAIN_LR_DEFAULTS["percent_dense"],
        help="Forwarded to train.py for densification split/clone decisions.",
    )
    parser.add_argument(
        "--densification_interval",
        type=int,
        default=_TRAIN_LR_DEFAULTS["densification_interval"],
        help="Forwarded to train.py; densify every N iterations while enabled.",
    )
    parser.add_argument(
        "--opacity_reset_interval",
        type=int,
        default=_TRAIN_LR_DEFAULTS["opacity_reset_interval"],
        help="Forwarded to train.py; opacity reset interval during densification.",
    )
    parser.add_argument(
        "--densify_from_iter",
        type=int,
        default=_TRAIN_LR_DEFAULTS["densify_from_iter"],
        help="Forwarded to train.py; first iteration after which densification can run.",
    )
    parser.add_argument(
        "--densify_until_iter",
        type=int,
        default=_TRAIN_LR_DEFAULTS["densify_until_iter"],
        help="Forwarded to train.py; densification stops before this iteration.",
    )
    parser.add_argument(
        "--densify_grad_threshold",
        type=float,
        default=_TRAIN_LR_DEFAULTS["densify_grad_threshold"],
        help="Forwarded to train.py; gradient threshold for clone/split.",
    )
    parser.add_argument("--use_pseudo_views", action="store_true", help="Forward pseudo-view supervision options to train.py.")
    parser.add_argument("--pseudo_manifest", type=str, default="", help="Pseudo-view train manifest path.")
    parser.add_argument("--pseudo_start_iter", type=int, default=1000, help="First iteration that can sample pseudo views.")
    parser.add_argument("--pseudo_loss_weight", type=float, default=0.1, help="Global pseudo-view loss multiplier.")
    parser.add_argument("--pseudo_sample_ratio", type=float, default=0.25, help="Probability of sampling a pseudo view per iteration.")
    parser.add_argument("--pseudo_use_densification", action="store_true", help="Allow pseudo views to update densification stats.")

    args = parser.parse_args()
    source = args.source_path.resolve()
    model = args.model_path.resolve()
    log_file = args.log_file.resolve() if args.log_file is not None else model / "pipeline.log"

    nv_train_only = ["--n_views", str(args.n_train)]
    nv_init = nv_train_only + ["--n_test", str(args.n_test)]

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"InstantSplat pipeline started at {dt.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"source_path: {source}\n")
        f.write(f"model_path: {model}\n")
        f.write(f"args: {vars(args)}\n")

    if not args.skip_init:
        init_args = [
            "init_geo.py",
            "-s",
            str(source),
            "-m",
            str(model),
            *nv_init,
            "--focal_avg",
            "--co_vis_dsp",
            "--conf_aware_ranking",
            "--scene_graph",
            args.scene_graph,
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
            "--sampling_seed",
            str(args.sampling_seed),
        ]
        _run(root / "init_geo.py", init_args[1:], root, log_file)

    if not args.skip_train:
        train_cmd = [
            "train.py",
            "-s",
            str(source),
            "-m",
            str(model),
            "-r",
            str(args.resolution),
            *nv_train_only,
            "--iterations",
            str(args.iterations),
            "--test_iterations",
            str(args.iterations),
        ]
        if not args.no_pp_optimizer:
            train_cmd.append("--pp_optimizer")
        if not args.no_optim_pose:
            train_cmd.append("--optim_pose")
        if args.use_densification:
            train_cmd.append("--use_densification")
        if args.use_pseudo_views:
            train_cmd.extend(
                [
                    "--use_pseudo_views",
                    "--pseudo_manifest",
                    args.pseudo_manifest,
                    "--pseudo_start_iter",
                    str(args.pseudo_start_iter),
                    "--pseudo_loss_weight",
                    str(args.pseudo_loss_weight),
                    "--pseudo_sample_ratio",
                    str(args.pseudo_sample_ratio),
                ]
            )
            if args.pseudo_use_densification:
                train_cmd.append("--pseudo_use_densification")
        train_cmd.extend(
            [
                "--position_lr_init",
                str(args.position_lr_init),
                "--position_lr_final",
                str(args.position_lr_final),
                "--feature_lr",
                str(args.feature_lr),
                "--opacity_lr",
                str(args.opacity_lr),
                "--scaling_lr",
                str(args.scaling_lr),
                "--rotation_lr",
                str(args.rotation_lr),
                "--percent_dense",
                str(args.percent_dense),
                "--densification_interval",
                str(args.densification_interval),
                "--opacity_reset_interval",
                str(args.opacity_reset_interval),
                "--densify_from_iter",
                str(args.densify_from_iter),
                "--densify_until_iter",
                str(args.densify_until_iter),
                "--densify_grad_threshold",
                str(args.densify_grad_threshold),
            ]
        )
        _run(root / "train.py", train_cmd[1:], root, log_file)

    if args.run_init_test_pose:
        itp = [
            "init_test_pose.py",
            "-s",
            str(source),
            "-m",
            str(model),
            *nv_init,
            "--focal_avg",
        ]
        _run(root / "init_test_pose.py", itp[1:], root, log_file)

    base_render = [
        "-s",
        str(source),
        "-m",
        str(model),
        "-r",
        str(args.resolution),
        *nv_train_only,
        "--iterations",
        str(args.iterations),
    ]

    if not args.skip_render_train:
        _run(root / "render.py", base_render, root, log_file)

    if not args.skip_render_eval:
        eval_render = [
            *base_render,
            "--eval",
            "--optim_test_pose_iter",
            str(args.optim_test_pose_iter),
        ]
        _run(root / "render.py", eval_render, root, log_file)

    if not args.skip_metrics:
        metrics_cmd = [
            "metrics.py",
            "-s",
            str(source),
            "-m",
            str(model),
            "--n_views",
            str(args.n_train),
            "--n_test",
            str(args.n_test),
        ]
        _run(root / "metrics.py", metrics_cmd[1:], root, log_file)

    print("\nPipeline finished.", flush=True)
    print(f"  model_path: {model}", flush=True)
    print(f"  split_manifest: {model / 'split_manifest.json'}", flush=True)
    print(f"  log_file: {log_file}", flush=True)


if __name__ == "__main__":
    main()
