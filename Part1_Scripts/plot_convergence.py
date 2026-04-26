#!/usr/bin/env python3
"""Parse 3DGS training logs and plot convergence curves.

Example:
    python plot_convergence.py \
        training_log_DL3DV-2_200images.txt \
        training_log_VGGT_DL3DV-2_200images-300000points-conf1.txt \
        --labels COLMAP VGGT \
        --output convergence_compare.png

Notes:
    - Existing logs usually contain only the evaluation iterations listed in
      `--test_iterations`, so PSNR/L1 eval curves may be sparse.
    - This script also parses tqdm-style `Training progress ... Loss=...`
      updates from the raw log, which gives a denser optimization curve.
    - For denser eval curves in future runs, pass denser test iterations to
      `train.py`, e.g.:
      --test_iterations 1000 3000 7000 15000 30000
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


EVAL_PATTERN = re.compile(
    r"\[ITER\s+(?P<iter>\d+)\]\s+Evaluating\s+"
    r"(?P<split>test|train):\s+L1\s+(?P<l1>[0-9.eE+-]+)\s+PSNR\s+(?P<psnr>[0-9.eE+-]+)"
)
PROGRESS_PATTERN = re.compile(
    r"Training progress:.*?\|\s*(?P<iter>\d+)/(?P<total>\d+)\s+\[[^\]]*?"
    r"Loss=(?P<loss>[0-9.eE+-]+),\s*Depth Loss=(?P<depth_loss>[0-9.eE+-]+)\]"
)
FINAL_PATTERN = re.compile(
    r"SSIM\s*:\s*(?P<ssim>[0-9.eE+-]+)\s*"
    r"PSNR\s*:\s*(?P<psnr>[0-9.eE+-]+)\s*"
    r"LPIPS:\s*(?P<lpips>[0-9.eE+-]+)",
    re.DOTALL,
)
INIT_POINTS_PATTERN = re.compile(r"Number of points at initialisation\s*:\s*(?P<points>\d+)")
DATASET_PATTERN = re.compile(r"正在处理\s+\[(?P<name>.+?)\]")


@dataclass
class EvalPoint:
    iteration: int
    split: str
    l1: float
    psnr: float


@dataclass
class ProgressPoint:
    iteration: int
    total_iters: int
    loss: float
    depth_loss: float


@dataclass
class LogSummary:
    label: str
    path: Path
    dataset_name: str | None
    init_points: int | None
    progress_points: list[ProgressPoint]
    eval_points: list[EvalPoint]
    final_ssim: float | None
    final_psnr: float | None
    final_lpips: float | None


def parse_log(path: Path, label: str) -> LogSummary:
    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    text = raw_text.replace("\r", "\n")

    progress_points = [
        ProgressPoint(
            iteration=int(match.group("iter")),
            total_iters=int(match.group("total")),
            loss=float(match.group("loss")),
            depth_loss=float(match.group("depth_loss")),
        )
        for match in PROGRESS_PATTERN.finditer(raw_text)
    ]
    progress_points.sort(key=lambda item: item.iteration)

    eval_points = [
        EvalPoint(
            iteration=int(match.group("iter")),
            split=match.group("split"),
            l1=float(match.group("l1")),
            psnr=float(match.group("psnr")),
        )
        for match in EVAL_PATTERN.finditer(text)
    ]
    eval_points.sort(key=lambda item: (item.split, item.iteration))

    final_match = FINAL_PATTERN.search(text)
    init_match = INIT_POINTS_PATTERN.search(text)
    dataset_match = DATASET_PATTERN.search(text)

    return LogSummary(
        label=label,
        path=path,
        dataset_name=dataset_match.group("name") if dataset_match else None,
        init_points=int(init_match.group("points")) if init_match else None,
        progress_points=progress_points,
        eval_points=eval_points,
        final_ssim=float(final_match.group("ssim")) if final_match else None,
        final_psnr=float(final_match.group("psnr")) if final_match else None,
        final_lpips=float(final_match.group("lpips")) if final_match else None,
    )


def group_eval_points(summary: LogSummary, split: str, metric: str) -> tuple[list[int], list[float]]:
    filtered = [point for point in summary.eval_points if point.split == split]
    xs = [point.iteration for point in filtered]
    ys = [getattr(point, metric.lower()) for point in filtered]
    return xs, ys


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values
    out: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        out.append(running_sum / min(index + 1, window))
    return out


def decimate(xs: list[int], ys: list[float], target_points: int = 1500) -> tuple[list[int], list[float]]:
    if len(xs) <= target_points:
        return xs, ys
    stride = math.ceil(len(xs) / target_points)
    return xs[::stride], ys[::stride]


def write_csv(summaries: list[LogSummary], output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "label",
                "dataset_name",
                "log_path",
                "init_points",
                "row_type",
                "split",
                "iteration",
                "total_iters",
                "loss",
                "depth_loss",
                "l1",
                "psnr",
                "final_ssim",
                "final_psnr",
                "final_lpips",
            ]
        )
        for summary in summaries:
            for point in summary.progress_points:
                writer.writerow(
                    [
                        summary.label,
                        summary.dataset_name or "",
                        str(summary.path),
                        summary.init_points if summary.init_points is not None else "",
                        "progress",
                        "",
                        point.iteration,
                        point.total_iters,
                        point.loss,
                        point.depth_loss,
                        "",
                        "",
                        summary.final_ssim if summary.final_ssim is not None else "",
                        summary.final_psnr if summary.final_psnr is not None else "",
                        summary.final_lpips if summary.final_lpips is not None else "",
                    ]
                )
            for point in summary.eval_points:
                writer.writerow(
                    [
                        summary.label,
                        summary.dataset_name or "",
                        str(summary.path),
                        summary.init_points if summary.init_points is not None else "",
                        "eval",
                        point.split,
                        point.iteration,
                        "",
                        "",
                        "",
                        point.l1,
                        point.psnr,
                        summary.final_ssim if summary.final_ssim is not None else "",
                        summary.final_psnr if summary.final_psnr is not None else "",
                        summary.final_lpips if summary.final_lpips is not None else "",
                    ]
                )
            if not summary.progress_points and not summary.eval_points:
                writer.writerow(
                    [
                        summary.label,
                        summary.dataset_name or "",
                        str(summary.path),
                        summary.init_points if summary.init_points is not None else "",
                        "final_only",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        summary.final_ssim if summary.final_ssim is not None else "",
                        summary.final_psnr if summary.final_psnr is not None else "",
                        summary.final_lpips if summary.final_lpips is not None else "",
                    ]
                )


def plot_curves(summaries: list[LogSummary], output_path: Path, title: str | None, smooth_window: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    specs = [
        ("progress", "loss", "Training Loss (dense from tqdm)"),
        ("test", "psnr", "Test PSNR"),
        ("test", "l1", "Test L1"),
        ("train", "psnr", "Train PSNR"),
    ]

    for axis, (split, metric, axis_title) in zip(axes.ravel(), specs):
        for summary in summaries:
            if split == "progress":
                xs = [point.iteration for point in summary.progress_points]
                ys = [point.loss for point in summary.progress_points]
                if not xs:
                    continue
                ys = moving_average(ys, smooth_window)
                xs, ys = decimate(xs, ys)
                axis.plot(xs, ys, linewidth=1.8, label=summary.label)
            else:
                xs, ys = group_eval_points(summary, split, metric)
                if not xs:
                    continue
                axis.plot(xs, ys, marker="o", linewidth=2, label=summary.label)
        axis.set_title(axis_title)
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Loss" if metric == "loss" else metric.upper())
        axis.grid(True, linestyle="--", alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))

    sparse_eval_labels = [
        summary.label
        for summary in summaries
        if len([point for point in summary.eval_points if point.split == "test"]) < 3
    ]
    subtitle = ""
    if sparse_eval_labels:
        subtitle = " | Sparse eval PSNR points: " + ", ".join(sparse_eval_labels)

    fig.suptitle((title or "3DGS Convergence Curves") + subtitle, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loss_only(
    summaries: list[LogSummary],
    output_path: Path,
    title: str | None,
    smooth_window: int,
    use_log_y: bool,
) -> None:
    fig, axis = plt.subplots(1, 1, figsize=(10, 5.5))

    for summary in summaries:
        xs = [point.iteration for point in summary.progress_points]
        ys = [point.loss for point in summary.progress_points]
        if not xs:
            continue
        ys = moving_average(ys, smooth_window)
        xs, ys = decimate(xs, ys)
        axis.plot(xs, ys, linewidth=2.0, label=summary.label)

    axis.set_title(title or "3DGS Training Loss")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Training Loss")
    axis.grid(True, linestyle="--", alpha=0.3)
    if use_log_y:
        axis.set_yscale("log")
    axis.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_summary(summaries: list[LogSummary]) -> None:
    print()
    print("Parsed log summary")
    print("-" * 108)
    print(f"{'Label':<18} {'InitPts':>10} {'LossPts':>10} {'EvalPts':>10} {'Final PSNR':>12} {'Final SSIM':>12} {'Final LPIPS':>13}")
    print("-" * 108)
    for summary in summaries:
        init_points = str(summary.init_points) if summary.init_points is not None else "-"
        loss_points = len(summary.progress_points)
        eval_points = len([point for point in summary.eval_points if point.split == "test"])
        final_psnr = f"{summary.final_psnr:.4f}" if summary.final_psnr is not None else "-"
        final_ssim = f"{summary.final_ssim:.4f}" if summary.final_ssim is not None else "-"
        final_lpips = f"{summary.final_lpips:.4f}" if summary.final_lpips is not None else "-"
        print(f"{summary.label:<18} {init_points:>10} {loss_points:>10} {eval_points:>10} {final_psnr:>12} {final_ssim:>12} {final_lpips:>13}")
    print("-" * 108)
    print()

    for summary in summaries:
        test_points = [point for point in summary.eval_points if point.split == "test"]
        if len(test_points) < 3:
            print(
                f"[提示] {summary.label} 只有 {len(test_points)} 个 test 评测点，"
                "所以 PSNR/L1 评测曲线较稀疏；但 training loss 曲线已经从 tqdm 里解析出来了。"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse 3DGS logs and plot convergence curves.")
    parser.add_argument("logs", nargs="+", help="One or more training log files.")
    parser.add_argument("--labels", nargs="*", help="Optional labels matching the log order.")
    parser.add_argument("--output", default="convergence_plot.png", help="Output image path.")
    parser.add_argument("--csv", help="Optional CSV export path for parsed metrics.")
    parser.add_argument("--title", help="Optional plot title.")
    parser.add_argument("--smooth-window", type=int, default=50, help="Moving average window for dense training loss.")
    parser.add_argument("--loss-only", action="store_true", help="Only plot the dense training loss curve.")
    parser.add_argument("--log-y", action="store_true", help="Use log scale on the y axis for loss plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_paths = [Path(log).expanduser().resolve() for log in args.logs]

    if args.labels and len(args.labels) != len(log_paths):
        raise SystemExit("--labels 的数量必须和日志文件数量一致。")

    labels = args.labels or [path.stem for path in log_paths]
    summaries = [parse_log(path, label) for path, label in zip(log_paths, labels)]

    output_path = Path(args.output).expanduser().resolve()
    if args.loss_only:
        plot_loss_only(summaries, output_path, args.title, args.smooth_window, args.log_y)
    else:
        plot_curves(summaries, output_path, args.title, args.smooth_window)

    if args.csv:
        write_csv(summaries, Path(args.csv).expanduser().resolve())

    print_summary(summaries)
    print(f"曲线图已保存到: {output_path}")
    if args.csv:
        print(f"解析结果 CSV 已保存到: {Path(args.csv).expanduser().resolve()}")


if __name__ == "__main__":
    main()
