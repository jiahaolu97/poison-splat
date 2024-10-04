import re

nerf_synthetic_scenes = [
    'chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship'
]
mip_nerf_360_scenes = [
    'bicycle', 'bonsai', 'counter', 'flowers', 'garden',
    'kitchen', 'room', 'stump'
]

def read_benchmark_result(log_path, read_std=False):
    logfile = open(log_path)
    content = logfile.read()
    numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', content)
    for i, number in enumerate(numbers):
        numbers[i] = float(number)
        if i == 2 or i == 3 or i == 8 or i == 9:
            numbers[i] = int(numbers[i])
        if i == 5:
            numbers[i] = round(numbers[i], 2)
    if read_std:
        return numbers
    return [numbers[0], numbers[2], numbers[4]]

def write_latex_wo_std(dataset='nerf_synthetic', setting='eps16'):
    assert dataset in ['nerf_synthetic', 'mip_nerf_360']
    assert setting in ['eps8', 'eps16', 'eps24', 'unbounded']
    clean_root_dir = f'log/02_blackbox_generalize/victim_scaffold_{dataset}_clean'
    poison_root_dir = f'log/02_blackbox_generalize/victim_scaffold_{dataset}_{setting}'
    latex_block = ""
    dataset_abbr = ""
    if dataset == 'nerf_synthetic':
        scenes = nerf_synthetic_scenes
        dataset_abbr = "NS"
    elif dataset == 'mip_nerf_360':
        scenes = mip_nerf_360_scenes
        dataset_abbr = "MIP"
    for scene in scenes:
        latex_line = f"{dataset_abbr}-{scene} & "
        # latex_line = f"{scene} & "
        clean_scene_result = f'{clean_root_dir}/{scene}/benchmark_result.log'
        poison_scene_result = f'{poison_root_dir}/{scene}/benchmark_result.log'
        # clean_gaussian_num, clean_GPU_mem, clean_train_time, clean_FPS = read_benchmark_result(clean_scene_result, read_std=False)
        # poison_gaussian_num, poison_GPU_mem, poison_train_time, poison_FPS = read_benchmark_result(poison_scene_result, read_std=False)
        clean_gaussian_num, clean_GPU_mem, clean_train_time = read_benchmark_result(clean_scene_result, read_std=False)
        poison_gaussian_num, poison_GPU_mem, poison_train_time = read_benchmark_result(poison_scene_result, read_std=False)
        ratio_gaussian_num = round(poison_gaussian_num / clean_gaussian_num, 2)
        ratio_GPU_mem = round(poison_GPU_mem / clean_GPU_mem, 2)
        ratio_train_time = round(poison_train_time / clean_train_time, 2)
        # ratio_FPS = round(clean_FPS / poison_FPS, 2)
        poison_GPU_mem_str = f'{poison_GPU_mem}' if poison_GPU_mem < 24000 else f'\\textbf{{{poison_GPU_mem}$^*$}}'
        latex_line += f"{clean_gaussian_num:.3f} M & {poison_gaussian_num:.3f} M \\redbf{{{ratio_gaussian_num:.2f}x}} & "
        latex_line += f"{clean_GPU_mem} & {poison_GPU_mem_str} \\redbf{{{ratio_GPU_mem:.2f}x}} & "
        latex_line += f"{clean_train_time:.2f} & {poison_train_time:.2f} \\redbf{{{ratio_train_time:.2f}x}} \\\\\n\n"
        # latex_line += f"{clean_FPS} & {poison_FPS} \\redbfd{{{ratio_FPS:.2f}x}} \\\\\n\n"
        latex_block += latex_line
        # print(latex_line)
    print(latex_block)

write_latex_wo_std(dataset='nerf_synthetic', setting='eps16')