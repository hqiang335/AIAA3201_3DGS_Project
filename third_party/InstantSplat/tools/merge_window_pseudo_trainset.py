#!/usr/bin/env python3
"""Merge short-window pseudo views into one global pseudo training manifest.

Window reconstructions have their own local coordinate systems.  This script
aligns each window back to a base/global InstantSplat reconstruction using the
real views shared by that window, filters low-confidence pseudo candidates, and
keeps the best duplicate when overlapping windows generate the same interval.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _as_4x4(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = pose
        return out
    raise ValueError(f"Unsupported pose shape: {pose.shape}")


def _camera_center(w2c: np.ndarray) -> np.ndarray:
    pose = _as_4x4(w2c)
    r = pose[:3, :3]
    t = pose[:3, 3]
    return -r.T @ t


def _umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, rotation, translation such that dst ~= scale * R @ src + t."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected Nx3 point sets, got {src.shape} and {dst.shape}")
    if len(src) < 3:
        raise ValueError("Need at least three shared real cameras to align a window robustly.")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / len(src)
    u, singular, vt = np.linalg.svd(cov)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rot = u @ correction @ vt
    var_src = np.sum(src_c**2) / len(src)
    if var_src <= 1e-12:
        raise ValueError("Cannot align window with near-degenerate camera centers.")
    scale = float(np.sum(singular * np.diag(correction)) / var_src)
    trans = mu_dst - scale * (rot @ mu_src)
    return scale, rot.astype(np.float32), trans.astype(np.float32)


def _align_w2c(local_w2c: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Convert local w2c pose to global w2c pose under X_g = s R X_l + t."""
    local = _as_4x4(local_w2c)
    r_local = local[:3, :3]
    c_local = _camera_center(local)
    c_global = scale * (rot @ c_local) + trans
    r_global = r_local @ rot.T
    t_global = -r_global @ c_global
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = r_global.astype(np.float32)
    out[:3, 3] = t_global.astype(np.float32)
    return out


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_or_scale_depth(src: Path, dst: Path, scale: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".npy":
        depth = np.load(src).astype(np.float32) * float(scale)
        np.save(dst, depth)
    else:
        shutil.copy2(src, dst)


def _candidate_score(
    view: dict,
    score_mask_weight: float,
    score_feature_weight: float,
    score_match_weight: float,
) -> float:
    mask_mean = float(view.get("mask_mean", view.get("combined_mask_mean", 0.0)))
    feature_mean = float(view.get("feature_mask_mean", 0.0))
    left_matches = float(view.get("feature_left_matches", 0.0))
    right_matches = float(view.get("feature_right_matches", 0.0))
    match_score = math.log1p(min(left_matches, right_matches)) / math.log1p(2048.0)
    return score_mask_weight * mask_mean + score_feature_weight * feature_mean + score_match_weight * match_score


def _candidate_manifest_path(window: dict, relative: str, fallback_relative: str) -> Path | None:
    if window.get("candidate_manifest"):
        path = Path(window["candidate_manifest"])
        if path.exists():
            return path
    model = Path(window["window_model"])
    for rel in (relative, fallback_relative):
        if not rel:
            continue
        path = model / rel
        if path.exists():
            return path
    difix_fused = window.get("difix_fused_dir") if fallback_relative == "__ALLOW_DIFIX_FALLBACK__" else ""
    if difix_fused:
        path = Path(difix_fused) / "pseudo_manifest_train.json"
        if path.exists():
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window_validation_manifest", type=Path, required=True)
    parser.add_argument("--base_split_manifest", type=Path, required=True)
    parser.add_argument("--base_pose_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--candidate_manifest_relative",
        type=str,
        default="pseudo_views/feature_masked/pseudo_manifest_train.json",
        help="Candidate manifest path relative to each window_model.",
    )
    parser.add_argument(
        "--fallback_candidate_manifest_relative",
        type=str,
        default="",
    )
    parser.add_argument(
        "--allow_difix_fallback",
        action="store_true",
        help="Allow fallback to each window's difix_fused_dir when the requested feature manifest is missing.",
    )
    parser.add_argument("--min_mask_mean", type=float, default=0.15)
    parser.add_argument("--min_feature_mask_mean", type=float, default=0.03)
    parser.add_argument("--min_feature_matches", type=int, default=32)
    parser.add_argument("--score_mask_weight", type=float, default=1.0)
    parser.add_argument("--score_feature_weight", type=float, default=1.0)
    parser.add_argument("--score_match_weight", type=float, default=0.25)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--set_loss_weight_from_mask", action="store_true")
    parser.add_argument("--loss_weight_scale", type=float, default=1.5)
    parser.add_argument("--min_loss_weight", type=float, default=0.05)
    parser.add_argument("--max_loss_weight", type=float, default=1.0)
    args = parser.parse_args()

    window_master = _load_json(args.window_validation_manifest.resolve())
    base_split = _load_json(args.base_split_manifest.resolve())
    base_pose_path = args.base_pose_path.resolve()
    base_poses = np.stack([_as_4x4(p) for p in np.load(base_pose_path)], axis=0)

    base_basenames = list(base_split["train_basenames"])
    base_name_to_index = {name: idx for idx, name in enumerate(base_basenames)}
    output_dir = args.output_dir.resolve()
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    depth_dir = output_dir / "depth"

    candidates_by_key: dict[tuple[int, int, int], dict] = {}
    filtered = []

    for window in window_master.get("windows", []):
        candidate_manifest = _candidate_manifest_path(
            window,
            args.candidate_manifest_relative,
            "__ALLOW_DIFIX_FALLBACK__" if args.allow_difix_fallback else args.fallback_candidate_manifest_relative,
        )
        if candidate_manifest is None:
            filtered.append({**window, "filtered_reason": "missing_candidate_manifest"})
            continue
        manifest = _load_json(candidate_manifest)
        manifest_dir = candidate_manifest.parent
        source_pose_value = manifest.get("source_pose_path")
        if not source_pose_value:
            filtered.append({**window, "filtered_reason": "missing_source_pose_path"})
            continue
        local_source_poses = np.stack(
            [_as_4x4(p) for p in np.load(_resolve(manifest_dir, source_pose_value))],
            axis=0,
        )
        local_pseudo_poses = np.stack(
            [_as_4x4(p) for p in np.load(_resolve(manifest_dir, manifest["pose_path"]))],
            axis=0,
        )

        local_centers = []
        global_centers = []
        local_to_global = {}
        for local_idx, basename in enumerate(window["train_basenames"]):
            if basename not in base_name_to_index:
                raise KeyError(f"Window real frame {basename} is not in base split.")
            global_idx = int(base_name_to_index[basename])
            local_to_global[int(local_idx)] = global_idx
            local_centers.append(_camera_center(local_source_poses[local_idx]))
            global_centers.append(_camera_center(base_poses[global_idx]))
        scale, rot, trans = _umeyama_similarity(np.stack(local_centers), np.stack(global_centers))

        for view in manifest.get("views", []):
            mask_mean = float(view.get("mask_mean", view.get("combined_mask_mean", 0.0)))
            feature_mean = float(view.get("feature_mask_mean", 0.0))
            left_matches = int(view.get("feature_left_matches", 0))
            right_matches = int(view.get("feature_right_matches", 0))
            if mask_mean < args.min_mask_mean:
                filtered.append({**view, "filtered_reason": "low_mask_mean", "source_window": window["window_tag"]})
                continue
            if feature_mean < args.min_feature_mask_mean:
                filtered.append({**view, "filtered_reason": "low_feature_mask_mean", "source_window": window["window_tag"]})
                continue
            if min(left_matches, right_matches) < args.min_feature_matches:
                filtered.append({**view, "filtered_reason": "low_feature_matches", "source_window": window["window_tag"]})
                continue

            left_global = local_to_global[int(view["left_train_index"])]
            right_global = local_to_global[int(view["right_train_index"])]
            interval_t = float(view.get("interval_t", 0.5))
            dedup_key = (left_global, right_global, int(round(interval_t * 10000.0)))
            score = _candidate_score(
                view,
                args.score_mask_weight,
                args.score_feature_weight,
                args.score_match_weight,
            )

            record = {
                "view": view,
                "manifest_dir": manifest_dir,
                "source_window": window["window_tag"],
                "score": score,
                "scale": scale,
                "rot": rot,
                "trans": trans,
                "pose": _align_w2c(local_pseudo_poses[int(view["pose_index"])], scale, rot, trans),
                "left_global_train_index": left_global,
                "right_global_train_index": right_global,
            }
            existing = candidates_by_key.get(dedup_key)
            if existing is None or score > existing["score"]:
                if existing is not None:
                    filtered.append({**existing["view"], "filtered_reason": "duplicate_lower_score", "source_window": existing["source_window"]})
                candidates_by_key[dedup_key] = record
            else:
                filtered.append({**view, "filtered_reason": "duplicate_lower_score", "source_window": window["window_tag"]})

    records = sorted(candidates_by_key.values(), key=lambda r: (r["left_global_train_index"], r["right_global_train_index"], r["view"].get("interval_t", 0.5)))
    if args.top_k > 0:
        keep = sorted(records, key=lambda r: r["score"], reverse=True)[: args.top_k]
        keep_ids = {id(r) for r in keep}
        filtered.extend(
            {**r["view"], "filtered_reason": "outside_top_k", "source_window": r["source_window"]}
            for r in records
            if id(r) not in keep_ids
        )
        records = sorted(keep, key=lambda r: (r["left_global_train_index"], r["right_global_train_index"], r["view"].get("interval_t", 0.5)))

    output_poses = []
    output_views = []
    for idx, record in enumerate(records):
        view = record["view"]
        manifest_dir = record["manifest_dir"]
        image_src = _resolve(manifest_dir, view["image_path"])
        mask_src = _resolve(manifest_dir, view["mask_path"])
        left_name = Path(view["left_image"]).stem
        right_name = Path(view["right_image"]).stem
        t_tag = int(round(float(view.get("interval_t", 0.5)) * 1000.0))
        image_name = f"pseudo_{record['source_window']}_{left_name}_{right_name}_t{t_tag:03d}"
        image_rel = Path("images") / f"{image_name}{image_src.suffix.lower()}"
        mask_rel = Path("masks") / f"{image_name}.png"
        _copy_image(image_src, output_dir / image_rel)
        _copy_image(mask_src, output_dir / mask_rel)

        new_view = {
            **view,
            "pose_index": idx,
            "pseudo_id": idx,
            "image_name": image_name,
            "image_path": str(image_rel),
            "mask_path": str(mask_rel),
            "left_train_index": record["left_global_train_index"],
            "right_train_index": record["right_global_train_index"],
            "left_image": base_basenames[record["left_global_train_index"]],
            "right_image": base_basenames[record["right_global_train_index"]],
            "source_window": record["source_window"],
            "window_to_global_scale": float(record["scale"]),
            "quality_score": float(record["score"]),
        }
        if args.set_loss_weight_from_mask:
            mask_mean = float(new_view.get("mask_mean", 0.0))
            new_view["loss_weight"] = float(np.clip(mask_mean * args.loss_weight_scale, args.min_loss_weight, args.max_loss_weight))

        depth_value = view.get("depth_path")
        if depth_value:
            depth_src = _resolve(manifest_dir, depth_value)
            if depth_src.exists():
                depth_rel = Path("depth") / f"{image_name}{depth_src.suffix.lower()}"
                _copy_or_scale_depth(depth_src, output_dir / depth_rel, record["scale"])
                new_view["depth_path"] = str(depth_rel)
                new_view["depth_scale_applied"] = float(record["scale"])

        output_poses.append(record["pose"])
        output_views.append(new_view)

    if not output_views:
        raise ValueError("No pseudo views survived window merge filtering.")

    pose_path = output_dir / "pseudo_poses_global.npy"
    np.save(pose_path, np.stack(output_poses, axis=0).astype(np.float32))

    base_pose_copy = output_dir / base_pose_path.name
    if base_pose_copy.resolve() != base_pose_path.resolve():
        shutil.copy2(base_pose_path, base_pose_copy)

    manifest = {
        "schema": "instantsplat_window_feature_pseudo_trainset_v1",
        "window_validation_manifest": str(args.window_validation_manifest.resolve()),
        "base_split_manifest": str(args.base_split_manifest.resolve()),
        "source_pose_path": base_pose_copy.name,
        "pose_path": pose_path.name,
        "views": output_views,
        "merge_filter": {
            "min_mask_mean": args.min_mask_mean,
            "min_feature_mask_mean": args.min_feature_mask_mean,
            "min_feature_matches": args.min_feature_matches,
            "top_k": args.top_k,
            "score_mask_weight": args.score_mask_weight,
            "score_feature_weight": args.score_feature_weight,
            "score_match_weight": args.score_match_weight,
        },
    }
    train_manifest = output_dir / "pseudo_manifest_train.json"
    _write_json(train_manifest, manifest)

    summary = {
        "num_kept_views": len(output_views),
        "num_filtered": len(filtered),
        "mask_mean_mean": float(np.mean([float(v.get("mask_mean", 0.0)) for v in output_views])),
        "feature_mask_mean_mean": float(np.mean([float(v.get("feature_mask_mean", 0.0)) for v in output_views])),
        "quality_score_mean": float(np.mean([float(v.get("quality_score", 0.0)) for v in output_views])),
        "filtered": filtered,
    }
    _write_json(output_dir / "merge_stats.json", summary)
    print(f"[merge-window] kept {len(output_views)}, filtered {len(filtered)}")
    print(f"[merge-window] wrote: {train_manifest}")


if __name__ == "__main__":
    main()
