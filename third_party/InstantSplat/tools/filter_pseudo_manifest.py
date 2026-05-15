#!/usr/bin/env python3
"""Filter a pseudo-view manifest by score while preserving temporal metadata.

This is intentionally a small dataset-building helper: it rewrites only the
``views`` list and records the ranking that produced the subset. The downstream
``build_pseudo_as_real_split.py`` script can then build a time-ordered source
directory from the filtered manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def view_key(view: dict[str, Any]) -> str:
    image_name = view.get("image_name")
    if image_name:
        return str(image_name)
    pseudo_id = view.get("pseudo_id")
    if pseudo_id is not None:
        return f"pseudo_{int(pseudo_id):05d}"
    raise ValueError(f"Cannot build view key for view: {view}")


def build_proxy_score_map(path: Path, metric: str) -> dict[str, float]:
    data = load_json(path)
    rows = data.get("rows", data if isinstance(data, list) else [])
    scores: dict[str, float] = {}
    for row in rows:
        key = str(row.get("image_name", ""))
        if not key:
            continue
        if metric == "proxy_full_psnr":
            scores[key] = float(row["full"]["psnr"])
        elif metric == "proxy_full_ssim":
            scores[key] = float(row["full"]["ssim"])
        elif metric == "proxy_masked_psnr":
            scores[key] = float(row["masked"]["psnr"])
        elif metric == "proxy_masked_ssim":
            scores[key] = float(row["masked"].get("ssim", row["full"]["ssim"]))
        else:
            raise ValueError(f"Unsupported proxy metric: {metric}")
    return scores


def builtin_score(view: dict[str, Any], score_name: str) -> float:
    if score_name == "manifest_loss_weight":
        return float(view.get("loss_weight", 1.0) or 0.0)
    if score_name == "manifest_mask_mean":
        return float(view.get("mask_mean", 0.0) or 0.0)
    if score_name == "manifest_feature_matches_min":
        left = float(view.get("feature_left_matches", 0.0) or 0.0)
        right = float(view.get("feature_right_matches", 0.0) or 0.0)
        return min(left, right)
    if score_name == "temporal_midpoint":
        t = float(view.get("interval_t", 0.5))
        return 1.0 - abs(t - 0.5) * 2.0
    if score_name == "temporal_endpoint":
        t = float(view.get("interval_t", 0.5))
        return abs(t - 0.5) * 2.0
    raise ValueError(f"Unsupported score: {score_name}")


def absolutize_view_paths(view: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    """Make file paths robust after writing the filtered manifest elsewhere."""
    result = dict(view)
    for key in (
        "image_path",
        "wide_image_path",
        "aspect_image_path",
        "mask_path",
        "feature_mask_path",
        "feature_left_path",
        "feature_right_path",
        "confidence_vis_path",
    ):
        value = result.get(key)
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = (manifest_dir / path).resolve()
        result[key] = str(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_manifest", type=Path, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--keep_count", type=int, default=None)
    parser.add_argument("--keep_fraction", type=float, default=None)
    parser.add_argument(
        "--score",
        default="manifest_loss_weight",
        choices=[
            "manifest_loss_weight",
            "manifest_mask_mean",
            "manifest_feature_matches_min",
            "temporal_midpoint",
            "temporal_endpoint",
            "proxy_full_psnr",
            "proxy_full_ssim",
            "proxy_masked_psnr",
            "proxy_masked_ssim",
        ],
    )
    parser.add_argument(
        "--score_json",
        type=Path,
        default=None,
        help="Required for proxy_* scores. Usually quality/pseudo_vs_gt_metrics.json.",
    )
    parser.add_argument("--reverse", action="store_true", help="Keep lowest scores instead of highest.")
    parser.add_argument("--min_score", type=float, default=None)
    parser.add_argument("--max_score", type=float, default=None)
    parser.add_argument("--keep_ids", default="", help="Comma-separated pseudo ids to force keep.")
    parser.add_argument("--drop_ids", default="", help="Comma-separated pseudo ids to force drop.")
    args = parser.parse_args()

    input_manifest = args.input_manifest.resolve()
    input_manifest_dir = input_manifest.parent
    manifest = load_json(input_manifest)
    views = list(manifest.get("views", []))
    if not views:
        raise ValueError(f"No views found in {args.input_manifest}")

    proxy_scores: dict[str, float] = {}
    if args.score.startswith("proxy_"):
        if args.score_json is None:
            raise ValueError("--score_json is required for proxy_* scores")
        proxy_scores = build_proxy_score_map(args.score_json, args.score)

    force_keep = {int(x) for x in args.keep_ids.split(",") if x.strip()}
    force_drop = {int(x) for x in args.drop_ids.split(",") if x.strip()}

    ranked = []
    for view in views:
        key = view_key(view)
        pseudo_id = int(view.get("pseudo_id", key.split("_")[-1]))
        if args.score.startswith("proxy_"):
            if key not in proxy_scores:
                raise KeyError(f"Missing score for {key} in {args.score_json}")
            score = proxy_scores[key]
        else:
            score = builtin_score(view, args.score)
        ranked.append((score, pseudo_id, view))

    candidates = []
    for score, pseudo_id, view in ranked:
        if pseudo_id in force_drop:
            continue
        if args.min_score is not None and score < args.min_score:
            continue
        if args.max_score is not None and score > args.max_score:
            continue
        candidates.append((score, pseudo_id, view))

    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=not args.reverse)

    if args.keep_fraction is not None:
        keep_count = max(1, int(round(len(candidates) * float(args.keep_fraction))))
    elif args.keep_count is not None:
        keep_count = int(args.keep_count)
    else:
        keep_count = len(candidates)
    keep_count = min(keep_count, len(candidates))

    selected = candidates[:keep_count]
    selected_ids = {pseudo_id for _, pseudo_id, _ in selected} | force_keep
    selected_views = []
    ranking = []
    score_by_id = {pseudo_id: score for score, pseudo_id, _ in ranked}
    for score, pseudo_id, view in sorted(ranked, key=lambda item: item[1]):
        keep = pseudo_id in selected_ids and pseudo_id not in force_drop
        ranking.append(
            {
                "pseudo_id": pseudo_id,
                "image_name": view.get("image_name"),
                "score": score,
                "kept": keep,
                "left_image": view.get("left_image"),
                "right_image": view.get("right_image"),
                "interval_t": view.get("interval_t"),
            }
        )
        if keep:
            kept_view = absolutize_view_paths(view, input_manifest_dir)
            kept_view["filter_score"] = score_by_id[pseudo_id]
            kept_view["filter_score_name"] = args.score
            selected_views.append(kept_view)

    selected_views.sort(key=lambda view: int(view.get("pseudo_id", 0)))
    output = dict(manifest)
    output["views"] = selected_views
    output["num_pseudo_views"] = len(selected_views)
    output["filtering"] = {
        "input_manifest": str(input_manifest),
        "score": args.score,
        "score_json": str(args.score_json.resolve()) if args.score_json else None,
        "keep_count_requested": args.keep_count,
        "keep_fraction_requested": args.keep_fraction,
        "num_input_views": len(views),
        "num_candidate_views": len(candidates),
        "num_kept_views": len(selected_views),
        "reverse": bool(args.reverse),
        "min_score": args.min_score,
        "max_score": args.max_score,
        "force_keep_ids": sorted(force_keep),
        "force_drop_ids": sorted(force_drop),
        "ranking": ranking,
    }

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote filtered manifest: {args.output_manifest}")
    print(f"Input pseudo views: {len(views)}")
    print(f"Kept pseudo views: {len(selected_views)}")
    print("Kept ids:", " ".join(str(v.get("pseudo_id")) for v in selected_views))


if __name__ == "__main__":
    main()
