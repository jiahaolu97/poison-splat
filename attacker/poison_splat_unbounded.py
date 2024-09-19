#
# Copyright (C) 2024, Jiahao Lu @ Skywork AI
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use.
#
# For inquiries contact jiahao.lu@u.nus.edu
import torch
import torchvision
import numpy as np
import os
import multiprocessing
import shutil
from random import sample
from argparse import ArgumentParser
from utils.loss_utils import l1_loss, ssim, image_total_variation
from utils.general_utils import safe_state, fix_all_random_seed
from utils.image_utils import psnr
from utils.log_utils import gpu_monitor_worker, plot_record, record_decoy_model_stats
from utils.attack_utils import (set_default_arguments, find_proxy_model, 
                            decoy_densify_and_prune, build_poisoned_data_folder)
from scene import Scene, GaussianModel
from scene.gaussian_renderer import render

def poison_splat_unbounded(args):
    fix_all_random_seed()
    decoy_gaussians = GaussianModel(args.sh_degree)
    proxy_model_path = find_proxy_model(args)
    scene = Scene(args, decoy_gaussians, load_proxy_path=proxy_model_path, shuffle=False)
    decoy_gaussians.training_setup(args)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    adv_viewpoint_stack = scene.getTrainCameras().copy()
    camera_num = len(adv_viewpoint_stack)

    gaussian_num_record = []
    gpu_monitor_stop_event = multiprocessing.Event()
    os.makedirs(f'{args.decoy_log_path}/decoy/', exist_ok=True)
    gpu_log_file_handle = open(f'{args.decoy_log_path}/decoy/gpu.log', 'w')
    gpu_monitor_process = multiprocessing.Process(target=gpu_monitor_worker, args=(gpu_monitor_stop_event, gpu_log_file_handle, args.gpu))
    psnr_record = []
    l1_record = []
    ssim_record = []
    decoy_render_tv_record = []
    poison_data_tv_record = []
    adv_alpha = args.adv_alpha
    adv_dsf_ratio = args.adv_dsf_ratio
    adv_dsf_interval = args.adv_dsf_interval
    adv_iters = args.adv_iters
    adv_decoy_update_interval = args.adv_decoy_update_interval
    adv_image_search_iters = args.adv_image_search_iters

    # Start Poisoning!
    gpu_monitor_process.start()
    viewpoint_seq = None
    seq_decoy_render_tv = []
    seq_poison_data_tv = []
    for attack_iter in range(1, adv_iters + 1):
        if not viewpoint_seq:
            viewpoint_seq = sample(range(camera_num), camera_num)
        viewpoint_cam_id = viewpoint_seq.pop(0)
        viewpoint_cam = adv_viewpoint_stack[viewpoint_cam_id]
        render_pkg = render(viewpoint_cam, decoy_gaussians, args, background)
        rendered_image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        seq_decoy_render_tv.append(image_total_variation(rendered_image).item())
        
        # Search the max Total Variation perturbation for adv_image_search_iters
        adv_gt_image = rendered_image.detach().clone().requires_grad_(True)
        for adv_image_search_iter in range(adv_image_search_iters):
            neg_tv_loss = image_total_variation(adv_gt_image) * -1
            neg_tv_loss.backward(inputs=[adv_gt_image])
            perturbation = adv_alpha * adv_gt_image.grad.sign()
            adv_image_unclipped = adv_gt_image.data - perturbation
            adv_gt_image = torch.clamp(adv_image_unclipped, 0, 1).requires_grad_(True)
        viewpoint_cam.set_adv_image(adv_gt_image)
        seq_poison_data_tv.append(image_total_variation(adv_gt_image).item())

        # Update the decoy gaussians by learning from max TV image
        Ll1 = l1_loss(rendered_image, adv_gt_image)
        Lssim = ssim(rendered_image, adv_gt_image)
        recon_loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - Lssim)
        recon_loss.backward()

        with torch.no_grad():
            if attack_iter % 10 == 0:
                print(f"[GPU-{args.gpu}] Attack iter {attack_iter}, recons loss {recon_loss.item():.3f}, {decoy_gaussians.print_Gaussian_num()}")
            # Densification
            decoy_gaussians.max_radii2D[visibility_filter] = torch.max(decoy_gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            decoy_gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            if attack_iter % adv_dsf_interval == 0 and attack_iter != adv_iters:
                decoy_densify_and_prune(
                    decoy_gaussians,
                    max_grad = args.densify_grad_threshold,
                    min_opacity = 0.005,
                    extent = scene.cameras_extent,
                    max_screen_size = 20
                )
            if attack_iter % adv_decoy_update_interval == 0:
                decoy_gaussians.optimizer.step()
                decoy_gaussians.optimizer.zero_grad(set_to_none=True)
        gaussian_num_record.append(decoy_gaussians.get_gaussian_num)
        psnr_record.append(psnr(rendered_image, adv_gt_image).mean().item())
        l1_record.append(Ll1.item())
        ssim_record.append(Lssim.item())
        if not viewpoint_seq:
            decoy_render_tv_record.append(sum(seq_decoy_render_tv) / len(seq_decoy_render_tv))
            seq_decoy_render_tv = []
            poison_data_tv_record.append(sum(seq_poison_data_tv) / len(seq_poison_data_tv))
            seq_poison_data_tv = []

    # Poisoning ends!
    gpu_monitor_stop_event.set()
    gpu_monitor_process.join()
    gpu_log_file_handle.flush()
    gpu_log_file_handle.close()
    gaussian_num_record_numpy = np.array(gaussian_num_record)
    np.save(f'{args.decoy_log_path}/decoy/gaussian_num_record.npy', gaussian_num_record_numpy)
    psnr_record_numpy = np.array(psnr_record)
    np.save(f'{args.decoy_log_path}/decoy/psnr_record.npy', psnr_record_numpy)
    l1_record_numpy = np.array(l1_record)
    np.save(f'{args.decoy_log_path}/decoy/l1_record.npy', l1_record_numpy)
    ssim_record_numpy = np.array(ssim_record)
    np.save(f'{args.decoy_log_path}/decoy/ssim_record.npy', ssim_record_numpy)
    decoy_render_tv_record_numpy = np.array(decoy_render_tv_record)
    np.save(f'{args.decoy_log_path}/decoy/decoy_render_tv_record.npy', decoy_render_tv_record_numpy)
    poison_data_tv_record_numpy = np.array(poison_data_tv_record)
    np.save(f'{args.decoy_log_path}/decoy/poison_data_tv_record.npy', poison_data_tv_record_numpy)
    record_decoy_model_stats(f'{args.decoy_log_path}/decoy/')

    # Save the poisoned images
    poisoned_data_folder = build_poisoned_data_folder(args)
    for viewpoint_cam_id, viewpoint_cam in enumerate(adv_viewpoint_stack):
        clean_image = viewpoint_cam.original_image
        render_pkg = render(viewpoint_cam, decoy_gaussians, args, background)
        rendered_image = render_pkg["render"].detach()
        poisoned_image = rendered_image
        image_name = viewpoint_cam.image_name
        torchvision.utils.save_image(poisoned_image.cpu(), f'{poisoned_data_folder}/{image_name}.{args.image_format}')

if __name__ == '__main__':
    parser = ArgumentParser(description='Poison-splat-unbounded-attack')
    parser.add_argument('--adv_iters', type=int, default=6000)
    parser.add_argument('--data_path', type=str, default='dataset/nerf_synthetic/chair')
    parser.add_argument('--decoy_log_path', type=str, default='log/decoy_nerf_synthetic_eps24/chair/')
    parser.add_argument('--data_output_path', type=str, default='dataset/nerf_synthetic_eps16/chair/')
    parser.add_argument('--adv_alpha', type=float, default=2/255.0)
    parser.add_argument('--adv_dsf_ratio', type=float, default=0.2)
    parser.add_argument('--adv_dsf_interval', type=int, default=100)
    parser.add_argument('--adv_image_search_iters', type=int, default=25) 
    parser.add_argument('--adv_decoy_update_interval', type=int, default=1)
    parser.add_argument('--densify_grad_threshold', type=float, default=0.0002)
    parser.add_argument('--adv_proxy_model_path', type=str, default=None)
    args = set_default_arguments(parser)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.makedirs(args.data_output_path, exist_ok = True)
    os.makedirs(args.decoy_log_path, exist_ok = True)
    safe_state(args, silent=False)
    
    # copy the camera config json files, and other necessary files
    args.output_path = args.decoy_log_path
    poison_splat_unbounded(args)

## go to root folder
# python attacker/poison_splat_unbounded.py --gpu 0  --adv_image_search_iters 10 --adv_iters 600\
#                     --data_path dataset/Nerf_Synthetic/chair/ \
#                     --data_output_path dataset/Nerf_Synthetic_unbounded/chair/ \
#                     --decoy_log_path log/test/attacker-unbounded/ns/chair/ \
#                     --adv_proxy_model_path /mnt/data/jiahaolu/dev-poison-splat/rebuttal_exp_output/nerf_synthetic_clean/chair/victim_model.ply