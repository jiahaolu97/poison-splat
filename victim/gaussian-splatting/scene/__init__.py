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
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], exp_run=1):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            if args.poison_ratio > 0.0:
                poison_ratio_round = round(args.poison_ratio * 100)
                scene = os.path.basename(os.path.normpath(args.source_path))
                poison_dataset = os.path.dirname(os.path.normpath(args.source_path))
                ref_folder = f"dataset/MIP_Nerf_360/{scene}/images/"
                
                image_folder = f"{args.source_path}/images/"
                camera_num = len([file for file in os.listdir(image_folder) if file.lower().endswith(".jpg")])
                poison_num = round(args.poison_ratio * camera_num)
                clean_num = camera_num - poison_num
                ref_indices = np.random.choice(np.arange(camera_num), size=clean_num, replace=False)
                clean_indices = ref_indices
                poison_indices = np.setdiff1d(np.arange(camera_num), clean_indices)
                # record the mix poison data id
                mix_source_path = f'{poison_dataset}_mix/poison_ratio_{poison_ratio_round}/{scene}/exp_run_{exp_run}/'
                poison_cam_log = open(f'{mix_source_path}poison_cam.txt', 'w')
                poison_cam_log.write(f'poison_indices:\n{poison_indices}:\n\n')
                poison_cam_log.write(f'clean_indices:\n{clean_indices}:\n')
                poison_cam_log.close()
            else:
                ref_folder = None
                ref_indices = None
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, ref_path=ref_folder, ref_indices=ref_indices)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            if args.poison_ratio > 0.0:
                poison_mix = True
            else:
                poison_mix = False
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, poison_mix=poison_mix)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]