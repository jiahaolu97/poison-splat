import re

nerf_synthetic_scenes = [
    'chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship'
]
tanks_and_temples_scenes = [
    'Auditorium', 'Ballroom', 'Barn', 'Caterpillar', 'Church', 'Courthouse',
    'Courtroom', 'Family', 'Francis', 'Horse', 'Ignatius', 'Lighthouse',
    'M60', 'Meetingroom', 'Museum', 'Palace', 'Panther', 'Playground',
    'Temple', 'Train', 'Truck'
]
mip_nerf_360_scenes = [
    'bicycle', 'bonsai', 'counter', 'flowers', 'garden',
    'kitchen', 'room', 'stump', 
    #'treehill'
]

def read_benchmark_result(log_path, read_std=False):
    logfile = open(log_path)
    content = logfile.read()
    numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', content)
    for i, number in enumerate(numbers):
        numbers[i] = float(number)
        if i == 2 or i == 3:
            numbers[i] = int(numbers[i])
        if i == 5:
            numbers[i] = round(numbers[i], 2)
    if read_std:
        return numbers[:6]
    return [numbers[0], numbers[2], numbers[4]]

def write_latex_w_std(dataset='nerf_synthetic'):
    assert dataset in ['nerf_synthetic', 'mip_nerf_360']
    latex_block = ""
    if dataset == 'nerf_synthetic':
        scenes = nerf_synthetic_scenes
    elif dataset == 'mip_nerf_360':
        scenes = mip_nerf_360_scenes
    # elif dataset == 'tanks_and_temples':
    #     scenes = tanks_and_temples_scenes
    for scene in scenes:
        latex_line = f"\multirow{{5}}{{*}}{{{scene}}}\n"
        for setting, latex_setting in zip(['clean', 'eps8', 'eps16', 'eps24', 'unbounded'], ['clean', '$\\epsilon=8/255$', '$\\epsilon=16/255$', '$\\epsilon=24/255$', 'unconstrained']):
            setting_root_dir = f'log/01_main_exp/victim_gs_{dataset}_{setting}/{scene}'
            setting_result_log = f'{setting_root_dir}/benchmark_result.log'
            gaussian_num, gaussian_num_std, gpu_mem, gpu_mem_std, training_time, training_time_std = read_benchmark_result(setting_result_log, read_std=True)
            latex_line += f" & {latex_setting} & {gaussian_num:.3f} M $\\pm$ {gaussian_num_std:.3f} M & {gpu_mem} MB $\\pm$ {gpu_mem_std} MB & {training_time:.2f} min $\\pm$ {training_time_std:.2f} min \\\\\n"
        latex_line += '\midrule\n'
        latex_block += latex_line
    print(latex_block)

write_latex_w_std(dataset='mip_nerf_360')