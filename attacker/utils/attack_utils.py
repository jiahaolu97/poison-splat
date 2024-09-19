#
# Copyright (C) 2024, Jiahao Lu @ Skywork AI
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use.
#
# For inquiries contact jiahao.lu@u.nus.edu
import torch
import os
import shutil
import sys

def set_default_arguments(parser):
    #======================
    # new params
    parser.add_argument('--quick', action='store_true', default=False,
                        help='if yes, run the optimization procedure in 3000 steps; otherwise stick to original hyper-parameter settings')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='if yes, visualize test set reconstruction')
    parser.add_argument('--init_gaussian_num', type=int, default=10_000,
                        help='for victim algorithm, number of initial gaussians')
    parser.add_argument('--sh_degree', type=int, default=3,
                        help='order of spherical harmonics to be used; if 0, use rgb. Default is 0')
    parser.add_argument('--input_resolution_downscale', type=int, default=-1,
                        help='if set, downscale input images by this factor. default is -1, downscale to 1.6k')
    parser.add_argument('--camera_shuffle', action='store_true', default=False,
                        help='if yes, the camera orders will shuffle')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which GPU to use')
    # original 3DGS params

    args = parser.parse_args(sys.argv[1:])
    args.quick = False
    
    #=====================
    # original 3DGS params (default hyperparameters)
    args.feature_lr = 0.0025
    args.opacity_lr = 0.05
    args.scaling_lr = 0.005 
    args.rotation_lr = 0.001
    args.position_lr_init = 0.00016
    args.position_lr_final = 0.0000016
    args.position_lr_delay_mult = 0.01
    if not hasattr(args, 'densify_grad_threshold'):
        args.densify_grad_threshold = 0.0002 # Limit that decides if points should be densified based on 2D position gradient, 0.0002 by default.
    args.lambda_dssim = 0.2 # Influence of SSIM on total loss from 0 to 1, 0.2 by default.
    args.percent_dense = 0.01 # Percentage of scene extent (0--1) a point must exceed to be forcibly densified, 0.01 by default.
    
    if args.quick:
        args.iterations = 3000
        args.test_iterations = [1500, 3000]
        args.save_iterations = []
        args.densify_from_iter = 50
        args.densify_until_iter = 3000
        args.densification_interval = 10
        args.opacity_reset_interval = 300
        args.upgrade_SH_degree_interval = 100
    else:
        args.iterations = 30_000
        args.test_iterations = [15_000, 30_000]
        args.save_iterations = [30_000]
        args.densify_from_iter = 500
        args.densify_until_iter = 15_000
        args.densification_interval = 100
        args.opacity_reset_interval = 3000
        args.upgrade_SH_degree_interval = 1000
        
    args.position_lr_max_steps = args.iterations # Number of steps (from 0) where position learning rate goes from initial to final. 30_000 by default.
    
    # deprecated 3DGS params (set to default values for code compatibility)
    args.images = 'images' # Altenative subdirectory for COLMAP images
    args.extend_train_set = False # if yes, extend test set to train set (100 -> 300 train cameras in NerfSynthetic)
    args.data_device = 'cuda' # where to put the source image data (can set to 'cpu' if training on high resolution dataset)
    args.white_background = False
    args.convert_SHs_python = False # if yes, make pipeline compute SHs with Pytorch instead of author's kernels
    args.compute_cov3D_python = False # if yes, make pipeline compute 3D covariances with Pytorch instead of author's kernels
    args.debug = False
    args.quiet = False # if yes, omit any text written to terminal
    
    return args

def find_proxy_model(args):
    if args.adv_proxy_model_path is not None:
        return args.adv_proxy_model_path
    if 'Nerf_Synthetic' in args.data_path:
        proxy_model_path = args.data_path.replace("dataset/Nerf_Synthetic/", "log/01_main_exp/victim_gs_nerf_synthetic_clean/")
    elif 'MIP_Nerf_360' in args.data_path:
        proxy_model_path = args.data_path.replace("dataset/MIP_Nerf_360/", "log/01_main_exp/victim_gs_mip_nerf_360_clean/")
    elif 'Tanks_and_Temples' in args.data_path:
        proxy_model_path = args.data_path.replace("dataset/Tanks_and_Temples", "log/01_main_exp/victim_gs_tanks_and_temples_clean/")
    else:
        assert False, f"Dataset not supported: {args.data_path}"
    proxy_model_path += "exp_run_1/victim_model.ply"
    assert os.path.exists(proxy_model_path), "Please provide proxy model path in [args.adv_proxy_model_path]"
    return proxy_model_path

def build_poisoned_data_folder(args):
    # build poisoned folder
    if 'Nerf_Synthetic' in args.data_path:
        for subset in ['train', 'test', 'val']:
            os.makedirs(f'{args.data_output_path}/{subset}', exist_ok = True)
            camera_config_json_file_src = f'{args.data_path}/transforms_{subset}.json'
            camera_config_json_file_dst = f'{args.data_output_path}/transforms_{subset}.json'
            shutil.copy2(camera_config_json_file_src, camera_config_json_file_dst)
        args.image_format = 'png'
        return f'{args.data_output_path}/train/'
    elif 'MIP_Nerf_360' in args.data_path:
        shutil.copy2(args.data_path + 'poses_bounds.npy', args.data_output_path + 'poses_bounds.npy')
        shutil.copytree(args.data_path + 'sparse', args.data_output_path + 'sparse', dirs_exist_ok=True)
        os.makedirs(f'{args.data_output_path}/{args.images}/', exist_ok = True)
        args.image_format = 'JPG'
        return f'{args.data_output_path}/{args.images}/'
    elif 'Tanks_and_Temples' in args.data_path:
        shutil.copytree(args.data_path + 'sparse', args.data_output_path + 'sparse', dirs_exist_ok=True)
        os.makedirs(f'{args.data_output_path}/{args.images}/', exist_ok = True)
        args.image_format = 'jpg'
        return f'{args.data_output_path}/{args.images}/'
    else:
        print(f"Dataset {args.data_path} not supported yet")
        assert False
    
def decoy_densify_and_prune(gaussians, max_grad, min_opacity, extent, max_screen_size):
    # Now is extractly same as victim densification
    grads = gaussians.xyz_gradient_accum / gaussians.denom
    grads[grads.isnan()] = 0.0

    # Use the same densification strategy as victim
    gaussians.densify_and_clone(grads, max_grad, extent)
    gaussians.densify_and_split(grads, max_grad, extent)

    # use the original pruning strategy?
    prune_mask = (gaussians.get_opacity < min_opacity).squeeze()
    if max_screen_size:
        big_points_vs = gaussians.max_radii2D > max_screen_size
        big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    gaussians.prune_points(prune_mask)

    torch.cuda.empty_cache()