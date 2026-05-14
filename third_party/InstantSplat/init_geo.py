import os
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from time import time

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from icecream import ic
ic(torch.cuda.is_available())  # Check if CUDA is available
ic(torch.cuda.device_count())

from mast3r.model import AsymmetricMASt3R
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.utils.device import to_numpy
from dust3r.utils.geometry import inv
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils.sfm_utils import (save_intrinsics, save_extrinsic, save_points3D, save_time, save_images_and_masks,
                             init_filestructure, get_sorted_image_files, split_train_eval_views, load_images, compute_co_vis_masks)
from utils.camera_utils import viewmatrix


def _entry_image_name(entry):
    if isinstance(entry, str):
        return entry
    for key in ("image", "image_path", "name", "basename", "file"):
        value = entry.get(key)
        if value:
            return value
    raise ValueError(f"Manifest entry is missing an image field: {entry}")


def _entry_time(entry, fallback):
    if isinstance(entry, str):
        return float(fallback)
    for key in ("time", "time_index", "frame_index", "frame_id", "index"):
        value = entry.get(key)
        if value is not None:
            return float(value)
    return float(fallback)


def _resolve_manifest_image(image_ref, image_dir):
    image_path = Path(image_ref)
    if image_path.is_absolute():
        return str(image_path)
    return str(image_dir / image_path)


def load_explicit_split_manifest(split_manifest, image_dir, n_views, n_test):
    """Load explicit train/test image order for pseudo-as-real experiments."""
    manifest_path = Path(split_manifest)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_entries = manifest.get("train") or manifest.get("train_views")
    test_entries = manifest.get("test") or manifest.get("test_views") or []
    if not train_entries:
        raise ValueError(f"{manifest_path} does not contain a non-empty train list")

    def normalize_entries(entries, split_name):
        normalized = []
        for fallback, entry in enumerate(entries):
            image_ref = _entry_image_name(entry)
            image_path = _resolve_manifest_image(image_ref, image_dir)
            if not Path(image_path).exists():
                raise FileNotFoundError(f"{split_name} image from manifest does not exist: {image_path}")
            normalized.append(
                {
                    "path": image_path,
                    "basename": Path(image_path).name,
                    "time": _entry_time(entry, fallback),
                    "entry": entry,
                }
            )
        normalized.sort(key=lambda item: item["time"])
        return normalized

    train_items = normalize_entries(train_entries, "train")
    test_items = normalize_entries(test_entries, "test") if test_entries else []

    if len(train_items) != n_views:
        raise ValueError(
            f"Explicit split has {len(train_items)} train images, but --n_views={n_views}. "
            "Pass --n_train/--n_views equal to the manifest train count."
        )
    if test_items and len(test_items) != n_test:
        raise ValueError(
            f"Explicit split has {len(test_items)} test images, but --n_test={n_test}. "
            "Pass --n_test equal to the manifest test count."
        )

    print(f">> Using explicit split manifest: {manifest_path}")
    print(" - train_set_times:  ", [round(item["time"], 6) for item in train_items])
    print(" - train_set_names:  ", [item["basename"] for item in train_items])
    if test_items:
        print(" - test_set_times:   ", [round(item["time"], 6) for item in test_items])
        print(" - test_set_names:   ", [item["basename"] for item in test_items])

    train_img_files = [item["path"] for item in train_items]
    test_img_files = [item["path"] for item in test_items]
    train_inds = [float(item["time"]) for item in train_items]
    test_inds = [float(item["time"]) for item in test_items]
    return manifest, train_items, test_items, train_img_files, test_img_files, train_inds, test_inds


def interpolate_pose_pair_by_alpha(left_pose, right_pose, alpha, rot_weight=0.1):
    """Interpolate one pose at a specific temporal alpha between two MASt3R poses."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    poses = np.stack([left_pose[:3, :4], right_pose[:3, :4]], axis=0)
    pos = poses[:, :3, 3]
    lookat = pos - rot_weight * poses[:, :3, 2]
    up = pos + rot_weight * poses[:, :3, 1]

    p = (1.0 - alpha) * pos[0] + alpha * pos[1]
    l = (1.0 - alpha) * lookat[0] + alpha * lookat[1]
    u = (1.0 - alpha) * up[0] + alpha * up[1]
    return viewmatrix(p - l, u - p, p)


def interpolate_test_poses_by_frame_indices(train_poses, train_indices, test_indices):
    """Generate test poses using each held-out frame's position between train frames."""
    train_indices = np.asarray(train_indices, dtype=np.float64)
    test_indices = np.asarray(test_indices, dtype=np.float64)
    if len(train_indices) != len(train_poses):
        raise ValueError(
            f"train_indices/pose count mismatch: {len(train_indices)} vs {len(train_poses)}"
        )

    test_poses = []
    for test_idx in test_indices:
        right = int(np.searchsorted(train_indices, test_idx, side="left"))
        if right <= 0:
            test_poses.append(train_poses[0][:3, :4])
            continue
        if right >= len(train_indices):
            test_poses.append(train_poses[-1][:3, :4])
            continue

        left = right - 1
        denom = max(train_indices[right] - train_indices[left], 1.0)
        alpha = (test_idx - train_indices[left]) / denom
        test_poses.append(
            interpolate_pose_pair_by_alpha(train_poses[left], train_poses[right], alpha)
        )

    return np.asarray(test_poses, dtype=np.float32).reshape(-1, 3, 4)


def main(source_path, model_path, ckpt_path, device, batch_size, image_size, schedule, lr, niter,
         min_conf_thr, llffhold, n_views, n_test, co_vis_dsp, depth_thre, conf_aware_ranking=False,
         focal_avg=False, infer_video=False, scene_graph="complete", max_init_points=0,
         point_sampling="grid_uniform_confidence", sampling_grid_size=24, min_point_distance_px=12.0,
         point_conf_threshold=0.0, sampling_seed=42, split_manifest=None):

    # ---------------- (1) Load model and images ----------------  
    save_path, sparse_0_path, sparse_1_path = init_filestructure(Path(source_path), n_views)
    model = AsymmetricMASt3R.from_pretrained(ckpt_path).to(device)
    image_dir = Path(source_path) / 'images'
    image_files, image_suffix = get_sorted_image_files(image_dir)
    explicit_split = None
    train_items = []
    test_items = []
    if split_manifest:
        if infer_video:
            raise ValueError("--split_manifest is for train/test eval runs and cannot be combined with --infer_video")
        explicit_split, train_items, test_items, train_img_files, test_img_files, train_inds, test_inds = (
            load_explicit_split_manifest(split_manifest, image_dir, n_views, n_test)
        )
        image_suffix = Path(train_img_files[0]).suffix
    elif infer_video:
        train_img_files = image_files
        test_img_files = []
    else:
        train_img_files, test_img_files, train_inds, test_inds = split_train_eval_views(
            image_files, n_train=n_views, n_test=n_test, verbose=True
        )
        mp = Path(model_path)
        mp.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": "instantsplat_train_test_split_v1",
            "n_train": len(train_img_files),
            "n_test": len(test_img_files),
            "train_indices": train_inds,
            "test_indices": test_inds,
            "train_basenames": [Path(p).name for p in train_img_files],
            "test_basenames": [Path(p).name for p in test_img_files],
            "split_source_manifest": None,
            "init_point_sampling": {
                "max_init_points": max_init_points,
                "point_sampling": point_sampling,
                "sampling_grid_size": sampling_grid_size,
                "min_point_distance_px": min_point_distance_px,
                "point_conf_threshold": point_conf_threshold,
                "sampling_seed": sampling_seed,
            },
        }
        with open(mp / "split_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    if explicit_split is not None:
        mp = Path(model_path)
        mp.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": "instantsplat_train_test_split_v1",
            "split_mode": "explicit_manifest",
            "split_source_manifest": str(Path(split_manifest).resolve()),
            "n_train": len(train_img_files),
            "n_test": len(test_img_files),
            "train_indices": train_inds,
            "test_indices": test_inds,
            "train_basenames": [Path(p).name for p in train_img_files],
            "test_basenames": [Path(p).name for p in test_img_files],
            "train_items": [
                {
                    "image": item["basename"],
                    "time": item["time"],
                    "source_entry": item["entry"],
                }
                for item in train_items
            ],
            "test_items": [
                {
                    "image": item["basename"],
                    "time": item["time"],
                    "source_entry": item["entry"],
                }
                for item in test_items
            ],
            "init_point_sampling": {
                "max_init_points": max_init_points,
                "point_sampling": point_sampling,
                "sampling_grid_size": sampling_grid_size,
                "min_point_distance_px": min_point_distance_px,
                "point_conf_threshold": point_conf_threshold,
                "sampling_seed": sampling_seed,
            },
        }
        with open(mp / "split_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    
    # when geometry init, only use train images
    image_files = train_img_files
    images, org_imgs_shape = load_images(image_files, size=image_size)

    start_time = time()
    print(f'>> Making pairs...')
    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    print(f">> Scene graph: {scene_graph} ({len(pairs)} directed pairs)")
    print(f'>> Inference...')
    output = inference(pairs, model, device, batch_size=1, verbose=True)
    print(f'>> Global alignment...')
    scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=300, schedule=schedule, lr=lr, focal_avg=args.focal_avg)

    # Extract scene information
    extrinsics_w2c = inv(to_numpy(scene.get_im_poses()))
    intrinsics = to_numpy(scene.get_intrinsics())
    focals = to_numpy(scene.get_focals())
    imgs = np.array(scene.imgs)
    pts3d = to_numpy(scene.get_pts3d())
    pts3d = np.array(pts3d)
    depthmaps = to_numpy(scene.im_depthmaps.detach().cpu().numpy())
    values = [param.detach().cpu().numpy() for param in scene.im_conf]
    confs = np.array(values)
    
    if conf_aware_ranking:
        print(f'>> Confiden-aware Ranking...')
        avg_conf_scores = confs.mean(axis=(1, 2))
        sorted_conf_indices = np.argsort(avg_conf_scores)[::-1]
        sorted_conf_avg_conf_scores = avg_conf_scores[sorted_conf_indices]
        print("Sorted indices:", sorted_conf_indices)
        print("Sorted average confidence scores:", sorted_conf_avg_conf_scores)
    else:
        sorted_conf_indices = np.arange(n_views)
        print("Sorted indices:", sorted_conf_indices)

    # Calculate the co-visibility mask
    print(f'>> Calculate the co-visibility mask...')
    if depth_thre > 0:
        overlapping_masks = compute_co_vis_masks(sorted_conf_indices, depthmaps, pts3d, intrinsics, extrinsics_w2c, imgs.shape, depth_threshold=depth_thre)
        overlapping_masks = ~overlapping_masks
    else:
        co_vis_dsp = False
        overlapping_masks = None
    end_time = time()
    Train_Time = end_time - start_time
    print(f"Time taken for {n_views} views: {Train_Time} seconds")
    save_time(model_path, '[1] coarse_init_TrainTime', Train_Time)

    # ---------------- (2) Interpolate training pose to get initial testing pose ----------------
    if not infer_video:
        n_train = len(train_img_files)
        n_test = len(test_img_files)

        pose_test_init = interpolate_test_poses_by_frame_indices(extrinsics_w2c, train_inds, test_inds)

        save_extrinsic(sparse_1_path, pose_test_init, test_img_files, image_suffix)
        test_focals = np.repeat(focals[0], n_test)
        save_intrinsics(sparse_1_path, test_focals, org_imgs_shape, imgs.shape, save_focals=False)
    # -----------------------------------------------------------------------------------------

    # Save results
    focals = np.repeat(focals[0], n_views)
    print(f'>> Saving results...')
    end_time = time()
    save_time(model_path, '[1] init_geo', end_time - start_time)
    save_extrinsic(sparse_0_path, extrinsics_w2c, image_files, image_suffix)
    save_intrinsics(sparse_0_path, focals, org_imgs_shape, imgs.shape, save_focals=True)
    pts_num = save_points3D(
        sparse_0_path,
        imgs,
        pts3d,
        confs.reshape(pts3d.shape[0], -1),
        overlapping_masks,
        use_masks=co_vis_dsp,
        save_all_pts=True,
        save_txt_path=model_path,
        depth_threshold=depth_thre,
        max_pts_num=max_init_points,
        sampling_strategy=point_sampling,
        sampling_grid_size=sampling_grid_size,
        min_point_distance_px=min_point_distance_px,
        conf_threshold=point_conf_threshold,
        sampling_seed=sampling_seed,
    )
    save_images_and_masks(sparse_0_path, n_views, imgs, overlapping_masks, image_files, image_suffix)
    print(f'[INFO] MASt3R Reconstruction is successfully converted to COLMAP files in: {str(sparse_0_path)}')
    print(f'[INFO] Number of points: {pts3d.reshape(-1, 3).shape[0]}')    
    print(f'[INFO] Number of points after downsampling: {pts_num}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and save results.')
    parser.add_argument('--source_path', '-s', type=str, required=True, help='Directory containing images')
    parser.add_argument('--model_path', '-m', type=str, required=True, help='Directory to save the results')
    parser.add_argument('--ckpt_path', type=str,
        default='./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth', help='Path to the model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing images')
    parser.add_argument('--image_size', type=int, default=512, help='Size to resize images')
    parser.add_argument('--schedule', type=str, default='cosine', help='Learning rate schedule')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--niter', type=int, default=300, help='Number of iterations')
    parser.add_argument('--min_conf_thr', type=float, default=5, help='Minimum confidence threshold')
    parser.add_argument('--llffhold', type=int, default=8, help='')
    parser.add_argument('--n_views', type=int, default=3, help='Number of training views (uniform subsample from non-test frames)')
    parser.add_argument('--n_test', type=int, default=12, help='Number of test views (default 12, linspace between frames 1..N-2)')
    # parser.add_argument('--focal_avg', type=bool, default=False, help='')
    parser.add_argument('--focal_avg', action="store_true")
    parser.add_argument('--conf_aware_ranking', action="store_true")
    parser.add_argument('--co_vis_dsp', action="store_true")
    parser.add_argument('--depth_thre', type=float, default=0.01, help='Depth threshold')
    parser.add_argument('--infer_video', action="store_true")
    parser.add_argument(
        '--scene_graph',
        type=str,
        default='complete',
        help='MASt3R/DUSt3R image graph, e.g. complete, swin-3-noncyclic, logwin-3-noncyclic',
    )
    parser.add_argument(
        '--max_init_points',
        type=int,
        default=0,
        help='Maximum points written to sparse_*/0/points3D.ply after co-visible filtering. 0 disables this cap.',
    )
    parser.add_argument(
        '--point_sampling',
        type=str,
        default='grid_uniform_confidence',
        choices=['grid_uniform_confidence', 'confidence_random'],
        help='Point sampling strategy used when --max_init_points is positive.',
    )
    parser.add_argument(
        '--sampling_grid_size',
        type=int,
        default=24,
        help='Grid size per image side for grid_uniform_confidence sampling. Set 0 to sample per-frame only.',
    )
    parser.add_argument(
        '--min_point_distance_px',
        type=float,
        default=12.0,
        help='Minimum pixel distance between selected points within the same frame.',
    )
    parser.add_argument(
        '--point_conf_threshold',
        type=float,
        default=0.0,
        help='Optional confidence threshold before max-point sampling. 0 disables thresholding.',
    )
    parser.add_argument(
        '--sampling_seed',
        type=int,
        default=42,
        help='Random seed for confidence_random point sampling.',
    )
    parser.add_argument(
        '--split_manifest',
        type=str,
        default=None,
        help='Explicit train/test split manifest. Images are read in ascending manifest time order.',
    )

    args = parser.parse_args()
    main(args.source_path, args.model_path, args.ckpt_path, args.device, args.batch_size, args.image_size, args.schedule, args.lr, args.niter,         
          args.min_conf_thr, args.llffhold, args.n_views, args.n_test, args.co_vis_dsp, args.depth_thre,
          args.conf_aware_ranking, args.focal_avg, args.infer_video, args.scene_graph,
          args.max_init_points, args.point_sampling, args.sampling_grid_size,
          args.min_point_distance_px, args.point_conf_threshold, args.sampling_seed,
          args.split_manifest)
