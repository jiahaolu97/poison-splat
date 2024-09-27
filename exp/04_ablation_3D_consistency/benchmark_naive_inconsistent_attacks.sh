# python exp/04_ablation_3D_consistency/naive_attack_gauss_noise.py --gpu 0 --adv_epsilon 16 \
#     --data_path dataset/Nerf_Synthetic/chair/ --decoy_log_path log/04_ablation_3D_consistency/naive_gauss_noise/decoy/chair/ --data_output_path dataset/exp_3D_consistency/naive_gauss_noise/chair/ &
# python exp/04_ablation_3D_consistency/naive_attack_gauss_noise.py --gpu 1 --adv_epsilon 16 \
#     --data_path dataset/Nerf_Synthetic/drums/ --decoy_log_path log/04_ablation_3D_consistency/naive_gauss_noise/decoy/drums/ --data_output_path dataset/exp_3D_consistency/naive_gauss_noise/drums/ &
# python exp/04_ablation_3D_consistency/naive_attack_gauss_noise.py --gpu 2 --adv_epsilon 16 \
#     --data_path dataset/Nerf_Synthetic/ficus/ --decoy_log_path log/04_ablation_3D_consistency/naive_gauss_noise/decoy/ficus/ --data_output_path dataset/exp_3D_consistency/naive_gauss_noise/ficus/ &
# python exp/04_ablation_3D_consistency/naive_attack_gauss_noise.py --gpu 3 --adv_epsilon 16 \
#     --data_path dataset/Nerf_Synthetic/hotdog/ --decoy_log_path log/04_ablation_3D_consistency/naive_gauss_noise/decoy/hotdog/ --data_output_path dataset/exp_3D_consistency/naive_gauss_noise/hotdog/ &
# python exp/04_ablation_3D_consistency/naive_attack_gauss_noise.py --gpu 4 --adv_epsilon 16 \
#     --data_path dataset/Nerf_Synthetic/lego/ --decoy_log_path log/04_ablation_3D_consistency/naive_gauss_noise/decoy/lego/ --data_output_path dataset/exp_3D_consistency/naive_gauss_noise/lego/ &
# python exp/04_ablation_3D_consistency/naive_attack_gauss_noise.py --gpu 5 --adv_epsilon 16 \
#     --data_path dataset/Nerf_Synthetic/materials/ --decoy_log_path log/04_ablation_3D_consistency/naive_gauss_noise/decoy/materials/ --data_output_path dataset/exp_3D_consistency/naive_gauss_noise/materials/ &
# python exp/04_ablation_3D_consistency/naive_attack_gauss_noise.py --gpu 6 --adv_epsilon 16 \
#     --data_path dataset/Nerf_Synthetic/mic/ --decoy_log_path log/04_ablation_3D_consistency/naive_gauss_noise/decoy/mic/ --data_output_path dataset/exp_3D_consistency/naive_gauss_noise/mic/ &
# python exp/04_ablation_3D_consistency/naive_attack_gauss_noise.py --gpu 7 --adv_epsilon 16 \
#     --data_path dataset/Nerf_Synthetic/ship/ --decoy_log_path log/04_ablation_3D_consistency/naive_gauss_noise/decoy/ship/ --data_output_path dataset/exp_3D_consistency/naive_gauss_noise/ship/
# wait
python exp/04_ablation_3D_consistency/naive_attack_tv_inconsistent.py --gpu 0 --adv_epsilon 8 --adv_image_search_iters 100 \
    --data_path dataset/Nerf_Synthetic/chair/ --decoy_log_path log/04_ablation_3D_consistency/naive_tv_max/decoy/chair/ --data_output_path dataset/exp_3D_consistency/naive_tv_max/chair/ &
python exp/04_ablation_3D_consistency/naive_attack_tv_inconsistent.py --gpu 1 --adv_epsilon 8 --adv_image_search_iters 100 \
    --data_path dataset/Nerf_Synthetic/drums/ --decoy_log_path log/04_ablation_3D_consistency/naive_tv_max/decoy/drums/ --data_output_path dataset/exp_3D_consistency/naive_tv_max/drums/ &
python exp/04_ablation_3D_consistency/naive_attack_tv_inconsistent.py --gpu 2 --adv_epsilon 8 --adv_image_search_iters 100 \
    --data_path dataset/Nerf_Synthetic/ficus/ --decoy_log_path log/04_ablation_3D_consistency/naive_tv_max/decoy/ficus/ --data_output_path dataset/exp_3D_consistency/naive_tv_max/ficus/ &
python exp/04_ablation_3D_consistency/naive_attack_tv_inconsistent.py --gpu 3 --adv_epsilon 8 --adv_image_search_iters 100 \
    --data_path dataset/Nerf_Synthetic/hotdog/ --decoy_log_path log/04_ablation_3D_consistency/naive_tv_max/decoy/hotdog/ --data_output_path dataset/exp_3D_consistency/naive_tv_max/hotdog/ &
python exp/04_ablation_3D_consistency/naive_attack_tv_inconsistent.py --gpu 4 --adv_epsilon 8 --adv_image_search_iters 100 \
    --data_path dataset/Nerf_Synthetic/lego/ --decoy_log_path log/04_ablation_3D_consistency/naive_tv_max/decoy/lego/ --data_output_path dataset/exp_3D_consistency/naive_tv_max/lego/ &
python exp/04_ablation_3D_consistency/naive_attack_tv_inconsistent.py --gpu 5 --adv_epsilon 8 --adv_image_search_iters 100 \
    --data_path dataset/Nerf_Synthetic/materials/ --decoy_log_path log/04_ablation_3D_consistency/naive_tv_max/decoy/materials/ --data_output_path dataset/exp_3D_consistency/naive_tv_max/materials/ &
python exp/04_ablation_3D_consistency/naive_attack_tv_inconsistent.py --gpu 6 --adv_epsilon 8 --adv_image_search_iters 100 \
    --data_path dataset/Nerf_Synthetic/mic/ --decoy_log_path log/04_ablation_3D_consistency/naive_tv_max/decoy/mic/ --data_output_path dataset/exp_3D_consistency/naive_tv_max/mic/ &
python exp/04_ablation_3D_consistency/naive_attack_tv_inconsistent.py --gpu 7 --adv_epsilon 8 --adv_image_search_iters 100 \
    --data_path dataset/Nerf_Synthetic/ship/ --decoy_log_path log/04_ablation_3D_consistency/naive_tv_max/decoy/ship/ --data_output_path dataset/exp_3D_consistency/naive_tv_max/ship/
wait
# python victim/gaussian-splatting/benchmark.py --gpu 0\
#     -s dataset/exp_3D_consistency/naive_gauss_noise/chair/ -m log/04_ablation_3D_consistency/naive_gauss_noise/victim/chair/ &
# python victim/gaussian-splatting/benchmark.py --gpu 1\
#     -s dataset/exp_3D_consistency/naive_gauss_noise/drums/ -m log/04_ablation_3D_consistency/naive_gauss_noise/victim/drums/ &
# python victim/gaussian-splatting/benchmark.py --gpu 2\
#     -s dataset/exp_3D_consistency/naive_gauss_noise/ficus/ -m log/04_ablation_3D_consistency/naive_gauss_noise/victim/ficus/ &
# python victim/gaussian-splatting/benchmark.py --gpu 3\
#     -s dataset/exp_3D_consistency/naive_gauss_noise/hotdog/ -m log/04_ablation_3D_consistency/naive_gauss_noise/victim/hotdog/ &
# python victim/gaussian-splatting/benchmark.py --gpu 4\
#     -s dataset/exp_3D_consistency/naive_gauss_noise/lego/ -m log/04_ablation_3D_consistency/naive_gauss_noise/victim/lego/ &
# python victim/gaussian-splatting/benchmark.py --gpu 5\
#     -s dataset/exp_3D_consistency/naive_gauss_noise/materials/ -m log/04_ablation_3D_consistency/naive_gauss_noise/victim/materials/ &
# python victim/gaussian-splatting/benchmark.py --gpu 6\
#     -s dataset/exp_3D_consistency/naive_gauss_noise/mic/ -m log/04_ablation_3D_consistency/naive_gauss_noise/victim/mic/ &
# python victim/gaussian-splatting/benchmark.py --gpu 7\
#     -s dataset/exp_3D_consistency/naive_gauss_noise/ship/ -m log/04_ablation_3D_consistency/naive_gauss_noise/victim/ship/
# wait
python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/exp_3D_consistency/naive_tv_max/chair/ -m log/04_ablation_3D_consistency/naive_tv_max/victim/chair/ &
python victim/gaussian-splatting/benchmark.py --gpu 1\
    -s dataset/exp_3D_consistency/naive_tv_max/drums/ -m log/04_ablation_3D_consistency/naive_tv_max/victim/drums/ &
python victim/gaussian-splatting/benchmark.py --gpu 2\
    -s dataset/exp_3D_consistency/naive_tv_max/ficus/ -m log/04_ablation_3D_consistency/naive_tv_max/victim/ficus/ &
python victim/gaussian-splatting/benchmark.py --gpu 3\
    -s dataset/exp_3D_consistency/naive_tv_max/hotdog/ -m log/04_ablation_3D_consistency/naive_tv_max/victim/hotdog/ &
python victim/gaussian-splatting/benchmark.py --gpu 4\
    -s dataset/exp_3D_consistency/naive_tv_max/lego/ -m log/04_ablation_3D_consistency/naive_tv_max/victim/lego/ &
python victim/gaussian-splatting/benchmark.py --gpu 5\
    -s dataset/exp_3D_consistency/naive_tv_max/materials/ -m log/04_ablation_3D_consistency/naive_tv_max/victim/materials/ &
python victim/gaussian-splatting/benchmark.py --gpu 6\
    -s dataset/exp_3D_consistency/naive_tv_max/mic/ -m log/04_ablation_3D_consistency/naive_tv_max/victim/mic/ &
python victim/gaussian-splatting/benchmark.py --gpu 7\
    -s dataset/exp_3D_consistency/naive_tv_max/ship/ -m log/04_ablation_3D_consistency/naive_tv_max/victim/ship/
wait