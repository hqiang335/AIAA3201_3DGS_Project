#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import json
import random
from pathlib import Path
from random import randint
from time import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.camera_utils import generate_interpolated_path
from utils.general_utils import safe_state
from utils.general_utils import PILtoTorch
from utils.graphics_utils import getWorld2View2_torch
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.pose_utils import get_camera_from_tensor, get_tensor_from_camera
from utils.sfm_utils import save_time
from lpipsPyTorch.modules.lpips import LPIPS
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

_PSEUDO_LPIPS_MODELS = {}


def get_pseudo_lpips_model(device, net_type="vgg"):
    key = (str(device), net_type)
    model = _PSEUDO_LPIPS_MODELS.get(key)
    if model is None:
        model = LPIPS(net_type=net_type).to(device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        _PSEUDO_LPIPS_MODELS[key] = model
    return model


def save_pose(path, quat_pose, train_cams, llffhold=2):
    # Get camera IDs and convert quaternion poses to camera matrices
    camera_ids = [cam.colmap_id for cam in train_cams]
    world_to_camera = [get_camera_from_tensor(quat) for quat in quat_pose]
    
    # Reorder poses according to colmap IDs
    colmap_poses = []
    for i in range(len(camera_ids)):
        idx = camera_ids.index(i + 1)  # Find position of camera i+1
        pose = world_to_camera[idx]
        colmap_poses.append(pose)
    
    # Convert to numpy array and save
    colmap_poses = torch.stack(colmap_poses).detach().cpu().numpy()
    np.save(path, colmap_poses)


def load_and_prepare_confidence(confidence_path, device='cuda', scale=(0.1, 1.0)):
    """
    Loads, normalizes, inverts, and scales confidence values to obtain learning rate modifiers.
    
    Args:
        confidence_path (str): Path to the .npy confidence file.
        device (str): Device to load the tensor onto.
        scale (tuple): Desired range for the learning rate modifiers.
    
    Returns:
        torch.Tensor: Learning rate modifiers.
    """
    # Load and normalize
    confidence_np = np.load(confidence_path)
    confidence_tensor = torch.from_numpy(confidence_np).float().to(device)
    normalized_confidence = torch.sigmoid(confidence_tensor)

    # Invert confidence and scale to desired range
    inverted_confidence = 1.0 - normalized_confidence
    min_scale, max_scale = scale
    lr_modifiers = inverted_confidence * (max_scale - min_scale) + min_scale
    
    return lr_modifiers


def _resolve_manifest_path(manifest_path: str, model_path: str) -> Path:
    path = Path(manifest_path)
    if path.is_absolute():
        return path
    candidate = Path(model_path) / path
    if candidate.exists():
        return candidate
    return path.resolve()


def _resolve_relative_path(base_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    parent_candidate = base_dir.parent / path
    if parent_candidate.exists():
        return parent_candidate
    return candidate


def _load_depth_tensor(depth_path: Path, resolution: tuple[int, int]) -> torch.Tensor:
    depth = np.load(depth_path).astype(np.float32)
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D pseudo depth map at {depth_path}, got shape {depth.shape}")

    width, height = resolution
    tensor = torch.from_numpy(depth)[None, None, ...]
    if tensor.shape[-2:] != (height, width):
        tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return tensor.squeeze(0).cuda()


def load_pseudo_cameras(opt, scene, train_cams):
    if not getattr(opt, "use_pseudo_views", False):
        return []
    manifest_value = getattr(opt, "pseudo_manifest", "")
    if not manifest_value:
        raise ValueError("--use_pseudo_views requires --pseudo_manifest")

    manifest_path = _resolve_manifest_path(manifest_value, scene.model_path)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest_dir = manifest_path.parent
    pose_path = _resolve_relative_path(manifest_dir, manifest["pose_path"])
    poses = np.load(pose_path)

    if not train_cams:
        raise ValueError("Cannot build pseudo cameras without real training cameras.")
    template = train_cams[0]
    resolution = (template.image_width, template.image_height)

    pseudo_cameras = []
    for idx, view in enumerate(manifest.get("views", [])):
        if "image_path" not in view:
            raise ValueError(f"Pseudo view {idx} is missing image_path.")
        pose = poses[int(view["pose_index"])]
        if pose.shape == (3, 4):
            pose_4x4 = np.eye(4, dtype=np.float32)
            pose_4x4[:3, :4] = pose
            pose = pose_4x4

        image_path = _resolve_relative_path(manifest_dir, view["image_path"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = PILtoTorch(image, resolution)[:3, ...]
        mask_value = view.get("mask_path") or view.get("feature_mask_path")
        if mask_value:
            mask_path = _resolve_relative_path(manifest_dir, mask_value)
            mask_image = Image.open(mask_path).convert("L")
            mask_tensor = PILtoTorch(mask_image, resolution)[:1, ...].clamp(0.0, 1.0).cuda()
        else:
            mask_tensor = torch.ones((1, resolution[1], resolution[0]), device="cuda")
        depth_tensor = None
        if getattr(opt, "pseudo_depth_weight", 0.0) > 0.0:
            depth_value = view.get("depth_path")
            if not depth_value:
                raise ValueError(
                    f"Pseudo view {idx} is missing depth_path, but --pseudo_depth_weight is enabled."
                )
            depth_tensor = _load_depth_tensor(_resolve_relative_path(manifest_dir, depth_value), resolution)

        cam = Camera(
            colmap_id=100000 + idx,
            R=pose[:3, :3].transpose(),
            T=pose[:3, 3],
            FoVx=template.FoVx,
            FoVy=template.FoVy,
            image=image_tensor,
            gt_alpha_mask=None,
            image_name=view.get("image_name", f"pseudo_{idx:05d}"),
            uid=idx,
            data_device=template.data_device,
        )
        cam.is_pseudo = True
        cam.confidence_mask = mask_tensor
        cam.loss_weight = float(view.get("loss_weight", 1.0))
        cam.pseudo_depth = depth_tensor
        cam.fixed_camera_pose = get_tensor_from_camera(cam.world_view_transform.transpose(0, 1)).detach()
        pseudo_cameras.append(cam)

    if not pseudo_cameras:
        raise ValueError(f"No pseudo views found in {manifest_path}")
    mask_mean = torch.stack([cam.confidence_mask.mean().detach().cpu() for cam in pseudo_cameras]).mean().item()
    print(f"Loaded pseudo cameras: {len(pseudo_cameras)} from {manifest_path}")
    print(f"Pseudo confidence mask mean: {mask_mean:.4f}")
    if getattr(opt, "pseudo_depth_weight", 0.0) > 0.0:
        depth_count = sum(1 for cam in pseudo_cameras if cam.pseudo_depth is not None)
        print(f"Loaded pseudo depth maps: {depth_count}/{len(pseudo_cameras)}")
    return pseudo_cameras


def pop_random_pseudo_camera(pseudo_cameras, pseudo_stack):
    if not pseudo_stack:
        pseudo_stack = pseudo_cameras.copy()
    rand_idx = randint(0, len(pseudo_stack) - 1)
    return pseudo_stack.pop(rand_idx), pseudo_stack


def pseudo_loss_ramp(opt, iteration):
    if iteration < opt.pseudo_start_iter:
        return 0.0
    ramp_until = max(int(getattr(opt, "pseudo_ramp_until", opt.pseudo_start_iter)), opt.pseudo_start_iter)
    if ramp_until <= opt.pseudo_start_iter:
        return 1.0
    return min(1.0, max(0.0, (iteration - opt.pseudo_start_iter) / float(ramp_until - opt.pseudo_start_iter)))


def pseudo_supervision_loss(viewpoint_cam, gaussians, pipe, bg, opt, render_pkg=None, iteration=None):
    need_pseudo_depth = (
        getattr(opt, "pseudo_depth_weight", 0.0) > 0.0
        and getattr(viewpoint_cam, "pseudo_depth", None) is not None
    )
    if render_pkg is None:
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            camera_pose=viewpoint_cam.fixed_camera_pose,
            return_depth=need_pseudo_depth,
        )
    image = render_pkg["render"]
    gt_image = viewpoint_cam.original_image.cuda()
    mask_base = viewpoint_cam.confidence_mask.cuda().clamp(0.0, 1.0)
    confidence_floor = float(getattr(opt, "pseudo_confidence_floor", 0.0))
    if confidence_floor > 0.0:
        confidence_floor = min(max(confidence_floor, 0.0), 1.0)
        mask_base = confidence_floor + (1.0 - confidence_floor) * mask_base
    mask_gamma = max(float(getattr(opt, "pseudo_mask_gamma", 1.0)), 1e-6)
    mask = torch.pow(mask_base, mask_gamma).expand_as(image)
    rgb_l1 = (torch.abs(image - gt_image) * mask).sum() / (mask.sum() + 1e-6)
    pseudo_loss = image.sum() * 0.0
    if getattr(opt, "pseudo_rgb_weight", 1.0) > 0.0:
        pseudo_loss = pseudo_loss + opt.pseudo_rgb_weight * rgb_l1
    if getattr(opt, "pseudo_charbonnier_weight", 0.0) > 0.0:
        charbonnier = torch.sqrt((image - gt_image).pow(2) + 1e-6)
        charbonnier = (charbonnier * mask).sum() / (mask.sum() + 1e-6)
        pseudo_loss = pseudo_loss + opt.pseudo_charbonnier_weight * charbonnier
    if getattr(opt, "pseudo_ssim_weight", 0.0) > 0.0:
        pseudo_ssim = 1.0 - ssim(image * mask, gt_image * mask)
        pseudo_loss = pseudo_loss + opt.pseudo_ssim_weight * pseudo_ssim
    if getattr(opt, "pseudo_lpips_weight", 0.0) > 0.0:
        lpips_model = get_pseudo_lpips_model(image.device, getattr(opt, "pseudo_lpips_net", "vgg"))
        pseudo_lpips = lpips_model((image * mask).unsqueeze(0), (gt_image * mask).unsqueeze(0)).mean()
        pseudo_loss = pseudo_loss + opt.pseudo_lpips_weight * pseudo_lpips
    if need_pseudo_depth:
        pred_depth = render_pkg["depth"]
        target_depth = viewpoint_cam.pseudo_depth.cuda()
        depth_mask = mask_base
        valid_depth = depth_mask * (target_depth > 0).float() * (pred_depth.detach() > 0).float()
        if opt.pseudo_depth_loss == "relative_l1":
            denom = (0.5 * (pred_depth.detach().abs() + target_depth.abs())).clamp_min(1e-3)
            depth_residual = torch.abs(pred_depth - target_depth) / denom
        elif opt.pseudo_depth_loss == "l1":
            depth_residual = torch.abs(pred_depth - target_depth)
        else:
            raise ValueError(f"Unsupported pseudo_depth_loss: {opt.pseudo_depth_loss}")
        depth_loss = (depth_residual * valid_depth).sum() / (valid_depth.sum() + 1e-6)
        pseudo_loss = pseudo_loss + opt.pseudo_depth_weight * depth_loss

    ramp = pseudo_loss_ramp(opt, iteration) if iteration is not None else 1.0
    return opt.pseudo_loss_weight * ramp * viewpoint_cam.loss_weight * pseudo_loss, rgb_l1


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    # per-point-optimizer
    confidence_path = os.path.join(dataset.source_path, f"sparse_{dataset.n_views}/0", "confidence_dsp.npy")
    confidence_lr = load_and_prepare_confidence(confidence_path, device='cuda', scale=(1, 100))
    scene = Scene(dataset, gaussians)
    if confidence_lr.shape[0] != gaussians.get_xyz.shape[0]:
        print(
            "Per-point confidence count does not match initialized Gaussians: "
            f"{confidence_lr.shape[0]} vs {gaussians.get_xyz.shape[0]}. "
            "Using uniform per-Gaussian LR."
        )
        confidence_lr = torch.ones((gaussians.get_xyz.shape[0], 1), device='cuda')

    if opt.use_densification and opt.pp_optimizer:
        raise ValueError("Densification currently supports the standard Adam optimizer only. Use --no_pp_optimizer.")

    if opt.pp_optimizer:
        gaussians.training_setup_pp(opt, confidence_lr)                          
    else:
        gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    train_cams_init = scene.getTrainCameras().copy()
    inline_pseudo_cameras = [cam for cam in train_cams_init if getattr(cam, "is_pseudo", False)]
    real_train_cameras = [cam for cam in train_cams_init if not getattr(cam, "is_pseudo", False)]
    for cam in inline_pseudo_cameras:
        if not hasattr(cam, "fixed_camera_pose"):
            cam.fixed_camera_pose = get_tensor_from_camera(cam.world_view_transform.transpose(0, 1)).detach()
    if inline_pseudo_cameras:
        print(
            f"Using inline pseudo cameras from explicit split: "
            f"{len(inline_pseudo_cameras)} pseudo, {len(real_train_cameras)} real"
        )
    external_pseudo_cameras = load_pseudo_cameras(opt, scene, train_cams_init)
    pseudo_cameras = inline_pseudo_cameras + external_pseudo_cameras
    if not real_train_cameras:
        real_train_cameras = train_cams_init
    for save_iter in saving_iterations:
        os.makedirs(scene.model_path + f'/pose/ours_{save_iter}', exist_ok=True)
        save_pose(scene.model_path + f'/pose/ours_{save_iter}/pose_org.npy', gaussians.P, train_cams_init)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = real_train_cameras.copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    pseudo_stack = pseudo_cameras.copy()
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")    
    first_iter += 1
    start = time()
    for iteration in range(first_iter, opt.iterations + 1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        pose_lr_scale = getattr(opt, "pose_lr_scale", 1.0)
        if pose_lr_scale != 1.0:
            for param_group in gaussians.optimizer.param_groups:
                if param_group.get("name") == "pose":
                    param_group["lr"] *= pose_lr_scale

        if opt.optim_pose==False:
            gaussians.P.requires_grad_(False)
        elif iteration <= getattr(opt, "pose_freeze_iters", 0):
            gaussians.P.requires_grad_(False)
        else:
            gaussians.P.requires_grad_(True)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera.  By default pseudo views replace a real-view
        # sample probabilistically; with --pseudo_pair_with_real, every
        # iteration keeps a real-view loss and optionally adds a pseudo loss.
        pair_pseudo_with_real = getattr(opt, "pseudo_pair_with_real", False)
        use_pseudo = (
            not pair_pseudo_with_real
            and len(pseudo_cameras) > 0
            and iteration >= opt.pseudo_start_iter
            and random.random() < opt.pseudo_sample_ratio
        )
        if use_pseudo:
            viewpoint_cam, pseudo_stack = pop_random_pseudo_camera(pseudo_cameras, pseudo_stack)
            pose = getattr(viewpoint_cam, "fixed_camera_pose", None)
            if pose is None:
                pose = gaussians.get_RT(viewpoint_cam.uid)
        else:
            if not viewpoint_stack:
                viewpoint_stack = real_train_cameras.copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            viewpoint_indices.pop(rand_idx)
            pose = gaussians.get_RT(viewpoint_cam.uid)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        is_pseudo_view = getattr(viewpoint_cam, "is_pseudo", False)
        need_pseudo_depth = (
            is_pseudo_view
            and getattr(opt, "pseudo_depth_weight", 0.0) > 0.0
            and getattr(viewpoint_cam, "pseudo_depth", None) is not None
        )

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose, return_depth=need_pseudo_depth)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if is_pseudo_view:
            loss, Ll1 = pseudo_supervision_loss(viewpoint_cam, gaussians, pipe, bg, opt, render_pkg=render_pkg, iteration=iteration)
        else:
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
            add_pseudo = (
                pair_pseudo_with_real
                and len(pseudo_cameras) > 0
                and iteration >= opt.pseudo_start_iter
                and random.random() < opt.pseudo_sample_ratio
            )
            if add_pseudo:
                pseudo_cam, pseudo_stack = pop_random_pseudo_camera(pseudo_cameras, pseudo_stack)
                pseudo_loss, _ = pseudo_supervision_loss(pseudo_cam, gaussians, pipe, bg, opt, iteration=iteration)
                loss = loss + pseudo_loss
        loss.backward()
        iter_end.record()
        # for param_group in gaussians.optimizer.param_groups:
        #     for param in param_group['params']:
        #         if param is gaussians.P:
        #             print(viewpoint_cam.uid, param.grad)
        #             break
        # print("Gradient of self.P:", gaussians.P.grad)
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Densification
            allow_densification = not getattr(viewpoint_cam, "is_pseudo", False) or opt.pseudo_use_densification
            if opt.use_densification and allow_densification and iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning.
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    before_points = gaussians.get_xyz.shape[0]
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    after_points = gaussians.get_xyz.shape[0]
                    print(f"\n[ITER {iteration}] Densification points: {before_points} -> {after_points}")
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # Log train/test PSNR at requested checkpoints.  Previously this
            # only ran at the final iteration, which made --test_iterations
            # misleading for overfitting checks.
            if iteration in testing_iterations or iteration % 5000 == 0:
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))

            # Log and save
            if iteration == opt.iterations:
                end = time()
                train_time_wo_log = end - start
                save_time(scene.model_path, '[2] train_joint_TrainTime', train_time_wo_log)
                if iteration not in testing_iterations and iteration % 5000 != 0:
                    training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                save_pose(scene.model_path + f'/pose/ours_{iteration}/pose_optimized.npy', gaussians.P, train_cams_init)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                
    end = time()
    train_time = end - start
    save_time(scene.model_path, '[2] train_joint', train_time)


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations or iteration % 5000 == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(len(scene.getTrainCameras()))]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if config['name']=="train":
                        pose = scene.gaussians.get_RT(viewpoint.uid)
                    else:
                        pose = scene.gaussians.get_RT_test(viewpoint.uid)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, camera_pose=pose)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=True)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    os.makedirs(args.model_path, exist_ok=True)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
