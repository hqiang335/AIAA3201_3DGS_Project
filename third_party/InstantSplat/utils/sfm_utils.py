import math
import os
import time
from pathlib import Path
from typing import List, NamedTuple, Tuple
import collections
import re

import numpy as np
import torch
import cv2
import PIL.Image
from PIL.ImageOps import exif_transpose
from plyfile import PlyData, PlyElement
import torchvision.transforms as tvf
import torchvision.transforms.functional as tf
import roma
import scipy
import open3d as o3d
from tqdm import tqdm

from dust3r.utils.image import _resize_pil_image
from dust3r.utils.device import to_numpy
from scene.colmap_loader import (
    qvec2rotmat, read_extrinsics_binary, rotmat2qvec,
    write_cameras_binary, write_cameras_text,
    write_images_text, write_images_binary
)

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

ImgNorm = tvf.Compose([
    tvf.ToTensor(), 
    tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def save_time(time_dir, process_name, sub_time):
    if isinstance(time_dir, str):
        time_dir = Path(time_dir)
    time_dir.mkdir(parents=True, exist_ok=True)
    minutes, seconds = divmod(sub_time, 60)
    formatted_time = f"{int(minutes)} min {int(seconds)} sec"  
    with open(time_dir / f'train_time.txt', 'a') as f:
        f.write(f'{process_name}: {formatted_time}\n')


def split_train_eval_views(
    image_files,
    n_train,
    n_test=12,
    verbose=True,
):
    """
    Hold out a fixed test set, then uniformly subsample training views from the remainder.

    Test indices match the legacy InstantSplat rule: ``linspace(1, N-2, n_test)`` on the
    sorted image list (excludes first and last frame indices 0 and N-1 from the test grid ends).

    Training indices: `n_train` uniformly spaced picks from all indices **not** in the test set,
    in ascending global index order along that candidate pool (same as legacy ``split_train_test``).

    ``image_files`` may be a list of paths or any sequence supporting ``len`` and integer indexing
    (e.g. `metrics.py` passes a pose array).
    """
    n = len(image_files)
    if n_train is None or n_train < 1:
        raise ValueError("n_train must be a positive integer")
    if n_test < 1:
        raise ValueError("n_test must be a positive integer")
    if n < 3:
        raise ValueError(f"Need at least 3 frames for the default test index rule, got {n}")

    test_idx = np.linspace(1, n - 2, num=n_test, dtype=int)
    test_set = {int(x) for x in test_idx.tolist()}
    train_pool = [i for i in range(n) if i not in test_set]

    if len(train_pool) < n_train:
        raise ValueError(
            f"Not enough training candidates after holding out {n_test} test slots: "
            f"pool={len(train_pool)}, n_train={n_train}, images={n}"
        )

    sparse_idx = np.linspace(0, len(train_pool) - 1, num=n_train, dtype=int)
    train_indices = [train_pool[int(i)] for i in sparse_idx]
    test_indices = [int(x) for x in test_idx.tolist()]

    if verbose:
        print(">> Spliting Train-Test Set: ")
        print(" - train_set_indices:  ", train_indices)
        print(" - test_set_indices:   ", test_indices)

    train_img_files = [image_files[i] for i in train_indices]
    test_img_files = [image_files[i] for i in test_indices]
    return train_img_files, test_img_files, train_indices, test_indices


def split_train_test(image_files, llffhold=8, n_views=None, n_test=12, verbose=True):
    """
    Backward-compatible wrapper. ``llffhold`` is unused (kept for call-site compatibility).

    ``n_views`` is the number of **training** views (what `init_geo` calls ``--n_views``).
    """
    if n_views is None:
        raise ValueError("n_views (number of training views) must be set")
    train_img_files, test_img_files, _, _ = split_train_eval_views(
        image_files, n_train=n_views, n_test=n_test, verbose=verbose
    )
    return train_img_files, test_img_files


def get_sorted_image_files(image_dir: str) -> Tuple[List[str], List[str]]:
    """
    Get sorted image files from the given directory.

    Args:
        image_dir (str): Path to the directory containing images.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
            - List of sorted image file paths
            - List of corresponding file suffixes
    """
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.JPG', '.PNG'}
    image_path = Path(image_dir)
    
    def extract_number(filename):
        match = re.search(r'\d+', filename.stem)
        return int(match.group()) if match else float('inf')
    
    image_files = [
        str(f) for f in image_path.iterdir()
        if f.is_file() and f.suffix.lower() in allowed_extensions
    ]
    
    sorted_files = sorted(image_files, key=lambda x: extract_number(Path(x)))
    suffixes = [Path(file).suffix for file in sorted_files]
    
    return sorted_files, suffixes[0]    


def rigid_points_registration(pts1, pts2, conf=None):
    R, T, s = roma.rigid_points_registration(
        pts1.reshape(-1, 3), pts2.reshape(-1, 3), weights=conf, compute_scaling=True)
    return s, R, T  # return un-scaled (R, T)


def init_filestructure(save_path, n_views=None):
    if n_views is not None and n_views != 0:        
        sparse_0_path = save_path / f'sparse_{n_views}/0'    
        sparse_1_path = save_path / f'sparse_{n_views}/1'       
        print(f'>> Doing {n_views} views reconstrution!')
    elif n_views is None or n_views == 0:
        sparse_0_path = save_path / 'sparse_0/0'    
        sparse_1_path = save_path / 'sparse_0/1'
        print(f'>> Doing full views reconstrution!')

    save_path.mkdir(exist_ok=True, parents=True)
    sparse_0_path.mkdir(exist_ok=True, parents=True)    
    sparse_1_path.mkdir(exist_ok=True, parents=True)
    return save_path, sparse_0_path, sparse_1_path


def load_images(folder_or_list, size, square_ok=False, verbose=True):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png', '.JPG', 'PNG']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs, (W1,H1)


import collections
CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])
      
def save_extrinsic(sparse_path, extrinsics_w2c, img_files, image_suffix):
    images_bin_file = sparse_path / 'images.bin'
    images_txt_file = sparse_path / 'images.txt'
    images = {}
    
    for i, (w2c, img_file) in enumerate(zip(extrinsics_w2c, img_files), start=1):  # Start enumeration from 1
        name = Path(img_file).stem + image_suffix
        rotation_matrix = w2c[:3, :3]
        qvec = rotmat2qvec(rotation_matrix)
        tvec = w2c[:3, 3]
        
        images[i] = BaseImage(
            id=i,
            qvec=qvec,
            tvec=tvec,
            camera_id=i,
            name=name,
            xys=[],  # Empty list as we don't have 2D point information
            point3D_ids=[]  # Empty list as we don't have 3D point IDs
        )
    
    write_images_binary(images, images_bin_file)
    write_images_text(images, images_txt_file)


def save_intrinsics(sparse_path, focals, org_imgs_shape, imgs_shape, save_focals=False):
    org_width, org_height = org_imgs_shape
    scale_factor_x = org_width / imgs_shape[2]
    scale_factor_y = org_height / imgs_shape[1]
    cameras_bin_file = sparse_path / 'cameras.bin'
    cameras_txt_file = sparse_path / 'cameras.txt'

    cameras = {}
    for i, focal in enumerate(focals, start=1):  # Start enumeration from 1
        cameras[i] = Camera(
            id=i,
            model="PINHOLE",
            width=org_width,
            height=org_height,
            params=[focal*scale_factor_x, focal*scale_factor_y, org_width/2, org_height/2]
        )    
    print(f' - scaling focal: ({focal}, {focal}) --> ({focal*scale_factor_x}, {focal*scale_factor_y})' )
    write_cameras_binary(cameras, cameras_bin_file)
    write_cameras_text(cameras, cameras_txt_file)
    if save_focals:
        np.save(sparse_path / 'non_scaled_focals.npy', focals)


def _can_select_point(occupied_buckets, bucket_size, frame_idx, y, x, min_distance_sq):
    bucket_y = int(y // bucket_size)
    bucket_x = int(x // bucket_size)
    frame_buckets = occupied_buckets[frame_idx]

    for ny in range(bucket_y - 1, bucket_y + 2):
        for nx in range(bucket_x - 1, bucket_x + 2):
            for py, px in frame_buckets.get((ny, nx), []):
                if (py - y) ** 2 + (px - x) ** 2 < min_distance_sq:
                    return False
    return True


def _register_point(occupied_buckets, bucket_size, frame_idx, y, x):
    bucket_y = int(y // bucket_size)
    bucket_x = int(x // bucket_size)
    occupied_buckets[frame_idx].setdefault((bucket_y, bucket_x), []).append((y, x))


def select_grid_uniform_confident_points(conf_maps, valid_mask, max_points, grid_size, min_point_distance_px):
    """Select high-confidence points while spreading samples across frames and image cells."""
    valid_count = int(valid_mask.sum())
    if max_points <= 0 or valid_count <= max_points:
        return valid_mask

    num_frames, height, width = conf_maps.shape
    cell_candidates = []

    if grid_size <= 0:
        for frame_idx in range(num_frames):
            frame_conf = conf_maps[frame_idx]
            frame_valid = valid_mask[frame_idx]
            if not frame_valid.any():
                continue
            y_idx, x_idx = np.nonzero(frame_valid)
            conf_vals = frame_conf[y_idx, x_idx]
            order = np.argsort(conf_vals)[::-1]
            frame_idx_arr = np.full(order.size, frame_idx, dtype=np.int32)
            cell_candidates.append(
                np.stack([frame_idx_arr, y_idx[order].astype(np.int32), x_idx[order].astype(np.int32)], axis=1)
            )
    else:
        grid_size = max(1, min(int(grid_size), height, width))
        h_edges = np.linspace(0, height, grid_size + 1, dtype=int)
        w_edges = np.linspace(0, width, grid_size + 1, dtype=int)

        for frame_idx in range(num_frames):
            frame_conf = conf_maps[frame_idx]
            frame_valid = valid_mask[frame_idx]
            for gh in range(grid_size):
                h_start, h_end = h_edges[gh], h_edges[gh + 1]
                for gw in range(grid_size):
                    w_start, w_end = w_edges[gw], w_edges[gw + 1]
                    cell_valid = frame_valid[h_start:h_end, w_start:w_end]
                    if not cell_valid.any():
                        continue

                    local_y, local_x = np.nonzero(cell_valid)
                    local_conf = frame_conf[h_start:h_end, w_start:w_end][local_y, local_x]
                    order = np.argsort(local_conf)[::-1]
                    y_idx = h_start + local_y[order]
                    x_idx = w_start + local_x[order]
                    frame_idx_arr = np.full(order.size, frame_idx, dtype=np.int32)
                    cell_candidates.append(
                        np.stack([frame_idx_arr, y_idx.astype(np.int32), x_idx.astype(np.int32)], axis=1)
                    )

    selected_flat = []
    candidate_offsets = np.zeros(len(cell_candidates), dtype=np.int32)
    occupied_buckets = [dict() for _ in range(num_frames)]
    bucket_size = max(1.0, float(min_point_distance_px))
    min_distance_sq = float(min_point_distance_px) ** 2

    while len(selected_flat) < max_points:
        made_progress = False
        for cell_idx, candidates in enumerate(cell_candidates):
            offset = candidate_offsets[cell_idx]
            while offset < len(candidates):
                frame_idx, y, x = candidates[offset]
                offset += 1
                if not _can_select_point(
                    occupied_buckets,
                    bucket_size,
                    int(frame_idx),
                    int(y),
                    int(x),
                    min_distance_sq,
                ):
                    continue

                _register_point(occupied_buckets, bucket_size, int(frame_idx), int(y), int(x))
                flat_idx = np.ravel_multi_index((int(frame_idx), int(y), int(x)), dims=conf_maps.shape)
                selected_flat.append(flat_idx)
                made_progress = True
                break
            candidate_offsets[cell_idx] = offset
            if len(selected_flat) >= max_points:
                break
        if not made_progress:
            break

    selected_mask = np.zeros(conf_maps.size, dtype=bool)
    selected_mask[np.asarray(selected_flat, dtype=np.int64)] = True
    return selected_mask.reshape(conf_maps.shape)


def select_confidence_random_points(conf_maps, valid_mask, max_points, sampling_seed=42):
    valid_count = int(valid_mask.sum())
    if max_points <= 0 or valid_count <= max_points:
        return valid_mask

    valid_indices = np.flatnonzero(valid_mask.reshape(-1))
    valid_conf = conf_maps.reshape(-1)[valid_indices].astype(np.float64)
    conf_min = np.min(valid_conf)
    conf_max = np.max(valid_conf)
    if conf_max > conf_min:
        weights = (valid_conf - conf_min) / (conf_max - conf_min)
    else:
        weights = np.ones_like(valid_conf)
    weights = weights + 1.0
    weights = weights / np.sum(weights)
    rng = np.random.default_rng(sampling_seed)
    selected_indices = rng.choice(valid_indices, size=max_points, replace=False, p=weights)

    selected_mask = np.zeros(conf_maps.size, dtype=bool)
    selected_mask[selected_indices] = True
    return selected_mask.reshape(valid_mask.shape)


def save_points3D(sparse_path, imgs, pts3d, confs, masks=None, use_masks=True, save_all_pts=False,
                  save_txt_path=None, depth_threshold=0.1, max_pts_num=150 * 10**10,
                  sampling_strategy="grid_uniform_confidence", sampling_grid_size=24,
                  min_point_distance_px=12.0, conf_threshold=0.0, sampling_seed=42):
    
    points3D_bin_file = sparse_path / 'points3D.bin'
    points3D_txt_file = sparse_path / 'points3D.txt'
    points3D_ply_file = sparse_path / 'points3D.ply'

    # Convert inputs to numpy arrays
    imgs = to_numpy(imgs)
    pts3d = to_numpy(pts3d)
    confs = to_numpy(confs)
    if confs is not None:
        np.save(sparse_path / 'confidence.npy', confs)

    num_images = pts3d.shape[0]
    pts_flat = pts3d.reshape(num_images, -1, 3)
    col_flat = imgs.reshape(num_images, -1, 3)
    conf_flat = confs.reshape(num_images, -1)

    image_hw = None
    if pts3d.ndim == 4 and pts3d.shape[-1] == 3:
        image_hw = pts3d.shape[1:3]

    # Process points and colors
    if use_masks:
        masks = to_numpy(masks)
        valid_mask = masks.reshape(num_images, -1).astype(bool)
    else:
        valid_mask = np.ones(conf_flat.shape, dtype=bool)

    co_mask_dsp_pts_num = int(valid_mask.sum())
    conf_threshold_pts_num = co_mask_dsp_pts_num
    if conf_threshold is not None and conf_threshold > 0:
        threshold_mask = conf_flat >= conf_threshold
        thresholded_valid_mask = valid_mask & threshold_mask
        conf_threshold_pts_num = int(thresholded_valid_mask.sum())
        if conf_threshold_pts_num > 0:
            valid_mask = thresholded_valid_mask
        else:
            print(
                f"Confidence threshold {conf_threshold} removed all points; "
                "falling back to the co-visible mask only."
            )
            conf_threshold_pts_num = co_mask_dsp_pts_num

    try:
        max_pts_num = int(max_pts_num)
    except (TypeError, ValueError):
        max_pts_num = 0

    selected_mask = valid_mask
    if max_pts_num > 0 and int(valid_mask.sum()) > max_pts_num:
        print(f"Downsampling points from {int(valid_mask.sum())} to {max_pts_num} using {sampling_strategy}")
        if sampling_strategy == "grid_uniform_confidence" and image_hw is not None:
            h, w = image_hw
            selected_mask = select_grid_uniform_confident_points(
                conf_flat.reshape(num_images, h, w),
                valid_mask.reshape(num_images, h, w),
                max_pts_num,
                sampling_grid_size,
                min_point_distance_px,
            ).reshape(num_images, -1)
        elif sampling_strategy == "confidence_random":
            selected_mask = select_confidence_random_points(conf_flat, valid_mask, max_pts_num, sampling_seed)
        else:
            raise ValueError(
                "Unknown or unsupported point sampling strategy "
                f"'{sampling_strategy}'. Use 'grid_uniform_confidence' or 'confidence_random'."
            )

    pts = pts_flat[selected_mask].reshape(-1, 3)
    col = col_flat[selected_mask].reshape(-1, 3) * 255.
    confs = conf_flat[selected_mask].reshape(-1, 1)
    conf_dsp_pts_num = pts.shape[0]

    if confs is not None:
        np.save(sparse_path / 'confidence_dsp.npy', confs)

    storePly(points3D_ply_file, pts, col)
    if save_all_pts:
        np.save(sparse_path / 'points3D_all.npy', pts3d)
        np.save(sparse_path / 'pointsColor_all.npy', imgs)
    
    # Write pts_num.txt
    if isinstance(save_txt_path, str):
        save_txt_path = Path(save_txt_path)
    pts_num_file = save_txt_path / f'pts_num.txt'  # New file for pts_num
    with open(pts_num_file, 'a') as f:
        f.write(f"Depth threshold: {depth_threshold}\n")
        f.write(f"Vanilla points num: {pts3d.reshape(-1, 3).shape[0]}\n")
        f.write(f"Co_Mask DSP points num: {co_mask_dsp_pts_num}\n")
        f.write(f"Co_Mask DSP ratio: {co_mask_dsp_pts_num / pts3d.reshape(-1, 3).shape[0]}\n")
        f.write(f"Confidence threshold: {conf_threshold}\n")
        f.write(f"Confidence-threshold points num: {conf_threshold_pts_num}\n")
        f.write(f"Point sampling strategy: {sampling_strategy}\n")
        f.write(f"Max init points: {max_pts_num}\n")
        f.write(f"Sampling grid size: {sampling_grid_size}\n")
        f.write(f"Min point distance px: {min_point_distance_px}\n")
        f.write(f"Sampling seed: {sampling_seed}\n")
        f.write(f"Final DSP points num: {conf_dsp_pts_num}\n")
        f.write(f"Final DSP ratio: {conf_dsp_pts_num / pts3d.reshape(-1, 3).shape[0]}\n")
        f.write("\n")
    
    return pts.shape[0]


# Save images and masks
def save_images_and_masks(sparse_0_path, n_views, imgs, overlapping_masks, image_files, image_suffix):

    images_path = sparse_0_path / f'imgs_{n_views}'
    overlapping_masks_path = sparse_0_path / f'overlapping_masks_{n_views}'

    images_path.mkdir(exist_ok=True, parents=True)
    overlapping_masks_path.mkdir(exist_ok=True, parents=True)

    for i, (image, name, overlapping_mask) in enumerate(zip(imgs, image_files, overlapping_masks)):
        imgname = Path(name).stem
        image_save_path = images_path / f"{imgname}{image_suffix}"
        overlapping_mask_save_path = overlapping_masks_path / f"{imgname}{image_suffix}"
        overlapping_mask_save_path = overlapping_masks_path / f"{imgname}{image_suffix}"

        # Save overlapping masks
        overlapping_mask = np.repeat(np.expand_dims(overlapping_mask, -1), 3, axis=2) * 255
        PIL.Image.fromarray(overlapping_mask.astype(np.uint8)).save(overlapping_mask_save_path)

        # Save images   
        rgb_image = cv2.cvtColor(image * 255, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(image_save_path), rgb_image)


def cal_co_vis_mask(points, depths, curr_depth_map, depth_threshold, camera_intrinsics, extrinsics_w2c):

    h, w = curr_depth_map.shape
    overlapping_mask = np.zeros((h, w), dtype=bool)
    # Project 3D points to image j
    points_2d, _ = project_points(points, camera_intrinsics, extrinsics_w2c)
    
    # Check if points are within image bounds
    valid_points = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
                   (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
        
    # Check depth consistency using vectorized operations
    valid_points_2d = points_2d[valid_points].astype(int)
    valid_depths = depths[valid_points]

    # Extract x and y coordinates
    x_coords, y_coords = valid_points_2d[:, 0], valid_points_2d[:, 1]

    # Compute depth differences
    depth_differences = np.abs(valid_depths - curr_depth_map[y_coords, x_coords])

    # Create a mask for points where the depth difference is below the threshold
    consistent_depth_mask = depth_differences < depth_threshold

    # Update the overlapping masks using the consistent depth mask
    overlapping_mask[y_coords[consistent_depth_mask], x_coords[consistent_depth_mask]] = True

    return overlapping_mask

def normalize_depth(depth_map):
    """Normalize the depth map to a range between 0 and 1."""
    return (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

def compute_co_vis_masks(sorted_conf_indices, depthmaps, pointmaps, camera_intrinsics, extrinsics_w2c, image_sizes, depth_threshold=0.1):

    num_images, h, w, _ = image_sizes
    pointmaps = pointmaps.reshape(num_images, h, w, 3)
    overlapping_masks = np.zeros((num_images, h, w), dtype=bool)
    
    for i, curr_map_idx in tqdm(enumerate(sorted_conf_indices), total=len(sorted_conf_indices)):

        # if frame_idx is 0, set its occ_mask to be all False
        if i == 0:
            continue

        # get before and after curr_frame's indices
        idx_before = sorted_conf_indices[:i]
        
        # idx_after = sorted_conf_indices[i+1:]

        # get partial pointmaps and depthmaps
        points_before = pointmaps[idx_before].reshape(-1, 3)
        depths_before = depthmaps[idx_before].reshape(-1)    
        # points_after = pointmaps[idx_after].reshape(-1, 3)        
        # depths_after = depthmaps[idx_after].reshape(-1)
        # get current frame's depth map
        curr_depth_map = depthmaps[curr_map_idx].reshape(h, w)

        # normalize depth for comparison
        depths_before = normalize_depth(depths_before)
        # depths_after = normalize_depth(depths_after)
        curr_depth_map = normalize_depth(curr_depth_map)

        # before_mask = overlapping_masks[idx_before]
        # after_mask = overlapping_masks[idx_after]
        # curr_mask = before_mask & after_mask

        before_mask = cal_co_vis_mask(points_before, depths_before, curr_depth_map, depth_threshold, camera_intrinsics[curr_map_idx], extrinsics_w2c[curr_map_idx])
        # after_mask = cal_co_vis_mask(points_after, depths_after, camera_intrinsics[i], extrinsics_w2c[i], curr_depth_map, depth_threshold)
        
        # white/True means co-visible redundant area: we need to remove
        overlapping_masks[curr_map_idx] = before_mask# & after_mask
        
    return overlapping_masks


def project_points(points_3d, intrinsics, extrinsics):
    # Convert to homogeneous coordinates
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    
    # Apply extrinsic matrix
    points_camera = np.dot(extrinsics, points_3d_homogeneous.T).T
    
    # Apply intrinsic matrix
    points_2d_homogeneous = np.dot(intrinsics, points_camera[:, :3].T).T
    
    # Convert to 2D coordinates
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]
    depths = points_camera[:, 2]
    
    return points_2d, depths

def read_colmap_gt_pose(gt_pose_path, llffhold=8):
    colmap_cam_extrinsics = read_extrinsics_binary(gt_pose_path + '/sparse/0/images.bin')
    colmap_cam_extrinsics = {k: v for k, v in sorted(colmap_cam_extrinsics.items(), key=lambda item: item[1].name)}
    all_pose=[]
    for idx, key in enumerate(colmap_cam_extrinsics):
        extr = colmap_cam_extrinsics[key]
        # print(idx, extr.name)
        R = np.transpose(qvec2rotmat(extr.qvec))
        # R = np.array(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        pose = np.eye(4,4)
        pose[:3, :3] = R
        pose[:3, 3] = T
        all_pose.append(pose)
    colmap_pose = np.array(all_pose)
    return colmap_pose


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = PIL.Image.open(renders_dir / fname)
        gt = PIL.Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def align_pose(pose1, pose2):
    mtx1 = np.array(pose1, dtype=np.double, copy=True)
    mtx2 = np.array(pose2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = scipy.linalg.orthogonal_procrustes(mtx1, mtx2)
    mtx2 = mtx2 * s

    return mtx1, mtx2, R

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
