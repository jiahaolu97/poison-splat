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
import sys
import multiprocessing
import shutil
from random import sample
from argparse import ArgumentParser
attacker_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'attacker')
sys.path.append(attacker_dir)
from utils.loss_utils import l1_loss, ssim, image_total_variation
from utils.general_utils import safe_state, fix_all_random_seed
from utils.image_utils import psnr
from utils.log_utils import gpu_monitor_worker, plot_record, record_decoy_model_stats
from utils.attack_utils import (set_default_arguments, find_proxy_model, 
                            decoy_densify_and_prune, build_poisoned_data_folder)
from scene import Scene, GaussianModel
from scene.gaussian_renderer import render


def naive_tv_max_bounded(args):
    fix_all_random_seed()
    decoy_gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, decoy_gaussians, shuffle=False)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    adv_viewpoint_stack = scene.getTrainCameras().copy()
    camera_num = len(adv_viewpoint_stack)
    adv_iters = camera_num

    adv_alpha = args.adv_alpha
    adv_epsilon = args.adv_epsilon
    adv_image_search_iters = args.adv_image_search_iters

    # Start Poisoning!
    viewpoint_seq = None
    for attack_iter in range(1, adv_iters + 1):
        if not viewpoint_seq:
            viewpoint_seq = sample(range(camera_num), camera_num)                
        viewpoint_cam_id = viewpoint_seq.pop(0)
        viewpoint_cam = adv_viewpoint_stack[viewpoint_cam_id]
        
        # Search the max Total Variation perturbation inside the epsilon ball
        clean_gt_image = viewpoint_cam.original_image
        adv_gt_image = clean_gt_image.detach().clone().requires_grad_(True)
        for adv_image_search_iter in range(adv_image_search_iters):
            neg_tv_loss = image_total_variation(adv_gt_image) * -1
            neg_tv_loss.backward(inputs=[adv_gt_image])
            perturbation = adv_alpha * adv_gt_image.grad.sign()
            adv_image_unclipped = adv_gt_image.data - perturbation
            clipped_perturbation = torch.clamp(adv_image_unclipped - clean_gt_image, -adv_epsilon, adv_epsilon) # clip into epsilon ball
            adv_gt_image = torch.clamp(clean_gt_image + clipped_perturbation, 0, 1).requires_grad_(True)
        viewpoint_cam.set_adv_image(adv_gt_image)
        print(f"Naive TV-max attack view {attack_iter} done")


    # Poisoning ends!
    # Save the poisoned images
    poisoned_data_folder = build_poisoned_data_folder(args)
    for viewpoint_cam_id, viewpoint_cam in enumerate(adv_viewpoint_stack):
        poisoned_image = viewpoint_cam.adv_image
        image_name = viewpoint_cam.image_name
        torchvision.utils.save_image(poisoned_image.cpu(), f'{poisoned_data_folder}/{image_name}.{args.image_format}')

if __name__ == '__main__':
    parser = ArgumentParser(description='Poison-splat-bounded-attack')
    parser.add_argument('--adv_epsilon', type=int, default=16)
    parser.add_argument('--adv_iters', type=int, default=6000)
    parser.add_argument('--data_path', type=str, default='dataset/nerf_synthetic/chair')
    parser.add_argument('--decoy_log_path', type=str, default='log/decoy_nerf_synthetic_eps24/chair/')
    parser.add_argument('--data_output_path', type=str, default='dataset/nerf_synthetic_eps16/chair/')
    parser.add_argument('--adv_image_search_iters', type=int, default=100) 
    parser.add_argument('--adv_proxy_model_path', type=str, default=None)
    args = set_default_arguments(parser)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.makedirs(args.data_output_path, exist_ok = True)
    args.adv_epsilon = args.adv_epsilon / 255.0
    args.adv_alpha = 2 / 255
    safe_state(args, silent=False)
    
    # copy the camera config json files, and other necessary files
    args.output_path = args.decoy_log_path
    naive_tv_max_bounded(args)
