#!/usr/bin/env python3
"""Summarize train/test PSNR across saved InstantSplat checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EVAL_RE = re.compile(r"\[ITER\s+(\d+)\]\s+Evaluating\s+(\w+):.*?PSNR\s+([0-9.]+)")
DENSIFY_RE = re.compile(r"\[ITER\s+(\d+)\]\s+Densification points:\s+(\d+)\s+->\s+(\d+)")


def _load_train_log(log_path: Path) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, int]]]:
    evals: dict[int, dict[str, float]] = {}
    densify: dict[int, dict[str, int]] = {}
    if not log_path.exists():
        return evals, densify

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = EVAL_RE.search(line)
        if match:
            iteration = int(match.group(1))
            split = match.group(2)
            psnr = float(match.group(3))
            evals.setdefault(iteration, {})[f"{split}_psnr"] = psnr
            continue

        match = DENSIFY_RE.search(line)
        if match:
            iteration = int(match.group(1))
            densify[iteration] = {
                "points_before": int(match.group(2)),
                "points_after": int(match.group(3)),
            }
    return evals, densify


def _load_render_metrics(results_path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    if not results_path.exists():
        return out

    with results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    for method, metrics in results.items():
        match = re.search(r"ours_(\d+)", method)
        if not match:
            continue
        iteration = int(match.group(1))
        out[iteration] = {
            "heldout_psnr": float(metrics.get("PSNR", 0.0)),
            "heldout_ssim": float(metrics.get("SSIM", 0.0)),
            "heldout_lpips": float(metrics.get("LPIPS", 0.0)),
            "ate": float(metrics.get("ATE", 0.0)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model_path", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=35.0)
    args = parser.parse_args()

    model_path = args.model_path.resolve()
    train_evals, densify = _load_train_log(model_path / "pipeline.log")
    render_metrics = _load_render_metrics(model_path / "results.json")

    iterations = sorted(set(train_evals) | set(render_metrics) | set(densify))
    rows = []
    for iteration in iterations:
        row = {"iteration": iteration}
        row.update(train_evals.get(iteration, {}))
        row.update(render_metrics.get(iteration, {}))
        row.update(densify.get(iteration, {}))
        rows.append(row)

    first_over = next(
        (row for row in rows if row.get("train_psnr", 0.0) >= args.threshold),
        None,
    )
    summary = {
        "model_path": str(model_path),
        "threshold": args.threshold,
        "first_train_psnr_over_threshold": first_over,
        "rows": rows,
    }
    out_path = model_path / "checkpoint_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("iteration train_psnr heldout_psnr heldout_ssim heldout_lpips points_after")
    for row in rows:
        print(
            row["iteration"],
            f"{row.get('train_psnr', 0.0):.4f}",
            f"{row.get('heldout_psnr', 0.0):.4f}",
            f"{row.get('heldout_ssim', 0.0):.4f}",
            f"{row.get('heldout_lpips', 0.0):.4f}",
            row.get("points_after", ""),
        )
    if first_over is None:
        print(f"No train PSNR >= {args.threshold:.2f} found.")
    else:
        print(
            f"First train PSNR >= {args.threshold:.2f}: "
            f"iter {first_over['iteration']} "
            f"(train={first_over.get('train_psnr', 0.0):.4f}, "
            f"heldout={first_over.get('heldout_psnr', 0.0):.4f})"
        )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
