#
# Copyright (C) 2024, Jiahao Lu @ Skywork AI
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use.
#
# For inquiries contact jiahao.lu@u.nus.edu

import torch
import numpy as np
import os
import sys
import random
from random import randint
import uuid
import time
import re
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import multiprocessing
from gpuinfo import GPUInfo
from datetime import datetime
import matplotlib.pyplot as plt

def benchmark_recon_quality(args):
    gaussians = GaussianModel(args.sh_degree)
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.load_ply(args.model_path + '/victim_model.ply')
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()
    SSIM_views = []
    PSNR_views = []
    for camid, cam in enumerate(viewpoint_stack):
        gt_image = cam.original_image.cuda()
        render_image = render(cam, gaussians, pipe, background)["render"]
        SSIM_views.append(ssim(render_image, gt_image).item())
        PSNR_views.append(psnr(render_image, gt_image).mean().item())
    mean_SSIM = round(sum(SSIM_views)/len(SSIM_views), 4)
    mean_PSNR = round(sum(PSNR_views)/len(PSNR_views), 4)
    print(f"Mean SSIM: {mean_SSIM}")
    print(f"Mean PSNR: {mean_PSNR}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="3DGS Victim Benchmark")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--exp_runs", type=int, default=3)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    benchmark_recon_quality(args)
    

    ## usage:
    # python benchmark.py -s [data path] -m [output path] --gpu [x]