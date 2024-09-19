#
# Copyright (C) 2024, Jiahao Lu @ Skywork AI
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use.
#
# For inquiries contact jiahao.lu@u.nus.edu
import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt
from gpuinfo import GPUInfo
from datetime import datetime

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

def record_decoy_model_stats(path):
    plot_record(f'{path}/gaussian_num_record.npy', 'Number of Gaussians')
    #plot_record(f'{path}/iter_elapse_record.npy', 'Iteration Elapse Time [ms]', 'Time')
    plot_record(f'{path}/psnr_record.npy', 'PSNR')
    plot_record(f'{path}/l1_record.npy', 'L1 Loss')
    plot_record(f'{path}/ssim_record.npy', 'SSIM')
    plot_record(f'{path}/decoy_render_tv_record.npy', 'TV from decoy render')
    plot_record(f'{path}/poison_data_tv_record.npy', 'TV from poisoned data')

    gpu_log = open(f'{path}/gpu.log', 'r')
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
    plt.xlabel('Poisoning time')
    plt.ylabel('GPU memory cost [MB]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{path}/gpu_mem_cost.png')
    plt.close()

    gaussian_num_record = np.load(f'{path}/gaussian_num_record.npy')
    training_start_timestamp = timestamps[0]
    training_end_timestamp = timestamps[-1]
    training_start_time = datetime.strptime(training_start_timestamp, "%Y-%m-%d %H:%M:%S")
    training_end_time = datetime.strptime(training_end_timestamp, "%Y-%m-%d %H:%M:%S")
    training_time_diff = training_end_time - training_start_time
    training_time = training_time_diff.seconds / 60

    max_gaussian_nums = max(gaussian_num_record) / 1000 / 1000
    max_GPU_mem = max(gpu_mem_cost)
    decoy_render_tv_record = np.load(f'{path}/decoy_render_tv_record.npy')
    poison_data_tv_record = np.load(f'{path}/poison_data_tv_record.npy')
    max_decoy_tv = max(decoy_render_tv_record) / 1000 / 1000
    max_poison_data_tv = max(poison_data_tv_record) / 1000 / 1000

    result_log = open(f'{path}/decoy_model.log', 'w')
    result_str = ''
    result_str += f"Max Gaussian Number: {max_gaussian_nums:.3f} M\n"
    result_str += f"Max GPU memory: {int(max_GPU_mem)} MB\n"
    result_str += f"Poisoning time: {training_time:.3f} min\n"
    result_str += f"Max Decoy Render TV: {max_decoy_tv:.5f} M\n"
    result_str += f"Max Poison Data TV: {max_poison_data_tv:.5f} M\n"
    print(result_str)
    result_log.write(result_str)
    result_log.flush()
    result_log.close()