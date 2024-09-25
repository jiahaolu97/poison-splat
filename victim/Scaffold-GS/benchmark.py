#
# Copyright (C) 2024, Jiahao Lu @ Skywork AI
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use.
#
# For inquiries contact jiahao.lu@u.nus.edu

import os
import numpy as np

import subprocess
# cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
# os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

# os.system('echo $CUDA_VISIBLE_DEVICES')


import torch
import torchvision
import json
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import multiprocessing
from gpuinfo import GPUInfo
from datetime import datetime
import matplotlib.pyplot as plt
import re
import random

# torch.set_num_threads(32)
#lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

def gpu_monitor_worker(stop_event, log_file_handle, gpuid=0):
    while not stop_event.is_set():
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        dt_object = datetime.fromtimestamp(timestamp)
        formatted_date = dt_object.strftime('%Y-%m-%d %H:%M:%S')
        percent, memory = GPUInfo.gpu_usage()
        if isinstance(percent, list):
            percent = [percent[gpuid]]
            memory = [memory[gpuid]]
        log_file_handle.write(f'[{formatted_date}] GPU:{gpuid} uses {percent}% and {memory} MB\n')
        log_file_handle.flush()
        time.sleep(0.2)
    print(f'GPU {gpuid} monitor stops')

def plot_record(file_name, record_name, xlabel='Iteration'):
    if not os.path.exists(file_name):
        return
    record = np.load(file_name)
    plt.figure()
    plt.plot(record, label=record_name)
    plt.xlabel(xlabel)
    plt.ylabel(record_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name.replace('.npy', '.png'))
    plt.close()

def fix_all_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_gaussian_num(gaussians):
    return gaussians._offset.size(0) * gaussians._offset.size(1)

def victim_training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None, exp_run=1):
    os.makedirs(f'{args.model_path}/exp_run_{exp_run}/', exist_ok=True)
    # ==============Tools for monitoring victim status==================
    record_gaussian_num = []
    record_iter_elapse = []
    record_l1 = []
    record_ssim = []
    record_psnr = []
    gpu_monitor_stop_event = multiprocessing.Event()
    gpu_log_file_handle = open(f'{args.model_path}/exp_run_{exp_run}/gpu.log', 'w')
    gpu_monitor_process = multiprocessing.Process(target=gpu_monitor_worker, args=(gpu_monitor_stop_event, gpu_log_file_handle, args.gpu))
    gpu_monitor_process.start()
    fix_all_random_seed()
    # =================================================================

    first_iter = 0
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe,background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        Lssim = ssim(image, gt_image)
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - Lssim) + 0.01*scaling_reg

        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                print(f"[GPU {args.gpu}]: Run {exp_run} iteration {iteration} - Loss {ema_loss_for_log:.3f}")
            
            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                
                # densification
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # ==================== Record iteration victim status =======================
            try:
                iter_elapse = iter_start.elapsed_time(iter_end)
                record_iter_elapse.append(iter_elapse)
            except:
                pass
            record_gaussian_num.append(get_gaussian_num(gaussians))
            record_psnr.append(psnr(image, gt_image).mean().item())
            record_l1.append(Ll1.item())
            record_ssim.append(Lssim.item())
            # ==========================================================================

     # ==================== Write Victim Records ==============================
    gpu_monitor_stop_event.set()
    gpu_monitor_process.join()
    gpu_log_file_handle.flush()
    gpu_log_file_handle.close()

    gaussians.save_ply(f'{args.model_path}/exp_run_{exp_run}/victim_model.ply')

    gaussian_num_record_numpy = np.array(record_gaussian_num)
    np.save(f'{args.model_path}/exp_run_{exp_run}/gaussian_num_record.npy', gaussian_num_record_numpy)
    plot_record(f'{args.model_path}/exp_run_{exp_run}/gaussian_num_record.npy', 'Number of Gaussians')

    iter_elapse_record_numpy = np.array(record_iter_elapse)
    np.save(f'{args.model_path}/exp_run_{exp_run}/iter_elapse_record.npy', iter_elapse_record_numpy)
    plot_record(f'{args.model_path}/exp_run_{exp_run}/iter_elapse_record.npy', 'Iteration Elapse Time [ms]', 'Time')

    psnr_record_numpy = np.array(record_psnr)
    np.save(f'{args.model_path}/exp_run_{exp_run}/psnr_record.npy', psnr_record_numpy)
    plot_record(f'{args.model_path}/exp_run_{exp_run}/psnr_record.npy', 'PSNR')
    l1_record_numpy = np.array(record_l1)
    np.save(f'{args.model_path}/exp_run_{exp_run}/l1_record.npy', l1_record_numpy)
    plot_record(f'{args.model_path}/exp_run_{exp_run}/l1_record.npy', 'L1 Loss')
    ssim_record_numpy = np.array(record_ssim)
    np.save(f'{args.model_path}/exp_run_{exp_run}/ssim_record.npy', ssim_record_numpy)
    plot_record(f'{args.model_path}/exp_run_{exp_run}/ssim_record.npy', 'SSIM')

    
    gpu_log = open(f'{args.model_path}/exp_run_{exp_run}/gpu.log', 'r')
    timestamps = []
    gpu_usage_percentage = []
    gpu_mem_cost = []
    for line in gpu_log:
        pattern = r'\[(.*?)\]'
        matches = re.findall(pattern, line)
        timestamps.append(matches[0])
        gpu_usage_percentage.append(int(matches[1]))
        gpu_mem_cost.append(int(matches[2]))
    plt.figure()
    plt.plot(gpu_mem_cost, label='GPU memory cost [MB]')
    plt.xlabel('Training time')
    plt.ylabel('GPU memory cost [MB]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{args.model_path}/exp_run_{exp_run}/gpu_mem_cost.png')
    plt.close()
    training_start_timestamp = timestamps[0]
    training_end_timestamp = timestamps[-1]
    training_start_time = datetime.strptime(training_start_timestamp, "%Y-%m-%d %H:%M:%S")
    training_end_time = datetime.strptime(training_end_timestamp, "%Y-%m-%d %H:%M:%S")
    training_time_diff = training_end_time - training_start_time
    training_time = training_time_diff.seconds / 60
    max_gaussian_nums = max(record_gaussian_num) / 1000 / 1000
    max_GPU_mem = max(gpu_mem_cost)
    result_log = open(f'{args.model_path}/exp_run_{exp_run}/benchmark_result.log', 'w')
    result_str = ''
    result_str += f"Max Gaussian Number: {max_gaussian_nums:.3f} M\n"
    result_str += f"Max GPU mem: {int(max_GPU_mem)} MB\n"
    result_str += f"Training time: {training_time:.3f} min\n"
    print(result_str)
    result_log.write(result_str)
    result_log.flush()
    result_log.close()
    # ======================================================================


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)


        # gts
        gt = view.original_image[0:3, :, :]
        
        # error maps
        errormap = (rendering - gt).abs()


        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

def conclude_victim_multiple_runs(args):
    max_gaussian_nums_runs = []
    max_gpu_mem_runs = []
    training_time_runs = []
    for exp_run in range(1, args.exp_runs + 1):
        record_gaussian_num = np.load(f'{args.model_path}/exp_run_{exp_run}/gaussian_num_record.npy')
        gpu_log = open(f'{args.model_path}/exp_run_{exp_run}/gpu.log', 'r')
        timestamps = []
        gpu_usage_percentage = []
        gpu_mem_cost = []
        for line in gpu_log:
            pattern = r'\[(.*?)\]'
            matches = re.findall(pattern, line)
            timestamps.append(matches[0])
            gpu_usage_percentage.append(int(matches[1]))
            gpu_mem_cost.append(int(matches[2]))
        training_start_timestamp = timestamps[0]
        training_end_timestamp = timestamps[-1]
        training_start_time = datetime.strptime(training_start_timestamp, "%Y-%m-%d %H:%M:%S")
        training_end_time = datetime.strptime(training_end_timestamp, "%Y-%m-%d %H:%M:%S")
        training_time_diff = training_end_time - training_start_time
        training_time = training_time_diff.seconds / 60
        max_gaussian_nums = max(record_gaussian_num) / 1000 / 1000
        max_GPU_mem = max(gpu_mem_cost)

        max_gaussian_nums_runs.append(max_gaussian_nums)
        max_gpu_mem_runs.append(max_GPU_mem)
        training_time_runs.append(training_time)
    
    max_gaussian_nums_runs_mean = np.mean(np.array(max_gaussian_nums_runs))
    max_gaussian_nums_runs_std = np.std(np.array(max_gaussian_nums_runs))
    max_gpu_mem_runs_mean = np.mean(np.array(max_gpu_mem_runs))
    max_gpu_mem_runs_std = np.std(np.array(max_gpu_mem_runs))
    training_time_runs_mean = np.mean(np.array(training_time_runs))
    training_time_runs_std = np.std(np.array(training_time_runs))

    result_log = open(f'{args.model_path}/benchmark_result.log', 'w')
    result_str = ''
    result_str += f"Max Gaussian Number: {max_gaussian_nums_runs_mean:.3f} M +- {max_gaussian_nums_runs_std:.3f} M\n"
    result_str += f"Max GPU mem: {int(max_gpu_mem_runs_mean)} MB +- {int(max_gpu_mem_runs_std)} MB\n"
    result_str += f"Training time: {training_time_runs_mean:.2f} min +- {training_time_runs_std} min\n"
    print(result_str)
    result_log.write(result_str)
    result_log.flush()
    result_log.close()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Scaffold-GS Victim Benchmark")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=int, default = '-1')
    parser.add_argument("--exp_runs", type=int, default=3)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # enable logging
    
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')

    if args.gpu != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    # if args.use_wandb:
    #     wandb.login()
    #     run = wandb.init(
    #         # Set the project where this run will be logged
    #         project=f"Scaffold-GS-{dataset}",
    #         name=exp_name,
    #         # Track hyperparameters and run metadata
    #         settings=wandb.Settings(start_method="fork"),
    #         config=vars(args)
    #     )
    # else:
    wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training
    for exp_run in range(1, args.exp_runs + 1):
        victim_training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  
        args.test_iterations, args.save_iterations, args.checkpoint_iterations, 
        args.start_checkpoint, args.debug_from, wandb, logger, exp_run=exp_run)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        for exp_run in range(1, args.exp_runs + 1):
            victim_training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  
            args.test_iterations, args.save_iterations, args.checkpoint_iterations, 
            args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, 
            ply_path=new_ply_path, exp_run=exp_run)

    # All done
    logger.info("\nTraining complete.")

    conclude_victim_multiple_runs(args)

    # # rendering
    # logger.info(f'\nStarting Rendering~')
    # visible_count = render_sets(lp.extract(args), -1, pp.extract(args), wandb=wandb, logger=logger)
    # logger.info("\nRendering complete.")

    # # calc metrics
    # logger.info("\n Starting evaluation...")
    # evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
    # logger.info("\nEvaluating complete.")


    ## usage:
    # python benchmark.py -s [data path] -m [output path] --gpu [x]
