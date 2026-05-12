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
import os
import subprocess
import sys
from pathlib import Path

from arguments import default_train_learning_rates

_TRAIN_LR_DEFAULTS = default_train_learning_rates()


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    env.pop("MKL_SERVICE_FORCE_INTEL", None)
    return env


def _run(py: Path, args: list[str], cwd: Path) -> None:
    cmd = [sys.executable, str(py)] + args
    print("\n$ " + " ".join(cmd) + "\n", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=_subprocess_env())


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
        "--scene_graph",
        type=str,
        default="complete",
        help="MASt3R image graph passed to init_geo.py",
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

    args = parser.parse_args()
    source = args.source_path.resolve()
    model = args.model_path.resolve()

    nv_train_only = ["--n_views", str(args.n_train)]
    nv_init = nv_train_only + ["--n_test", str(args.n_test)]

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
        ]
        _run(root / "init_geo.py", init_args[1:], root)

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
        train_cmd.extend(
            [
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
        )
        _run(root / "train.py", train_cmd[1:], root)

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
        _run(root / "init_test_pose.py", itp[1:], root)

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
        _run(root / "render.py", base_render, root)

    if not args.skip_render_eval:
        eval_render = [
            *base_render,
            "--eval",
            "--optim_test_pose_iter",
            str(args.optim_test_pose_iter),
        ]
        _run(root / "render.py", eval_render, root)

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
        _run(root / "metrics.py", metrics_cmd[1:], root)

    print("\nPipeline finished.", flush=True)
    print(f"  model_path: {model}", flush=True)
    print(f"  split_manifest: {model / 'split_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
