python attacker/poison_splat_bounded.py --gpu 0 --adv_epsilon 16 --adv_iters 36000 --adv_image_search_iters 15 \
    --data_path dataset/Nerf_Synthetic/chair/ --decoy_log_path log/01_main_exp/attacker_eps16_nerf_synthetic/chair/ --data_output_path dataset/Nerf_Synthetic_eps16/chair/ &
python attacker/poison_splat_bounded.py --gpu 1 --adv_epsilon 16 --adv_iters 36000 --adv_image_search_iters 15 \
    --data_path dataset/Nerf_Synthetic/drums/ --decoy_log_path log/01_main_exp/attacker_eps16_nerf_synthetic/drums/ --data_output_path dataset/Nerf_Synthetic_eps16/drums/ &
python attacker/poison_splat_bounded.py --gpu 2 --adv_epsilon 16 --adv_iters 36000 --adv_image_search_iters 15 \
    --data_path dataset/Nerf_Synthetic/ficus/ --decoy_log_path log/01_main_exp/attacker_eps16_nerf_synthetic/ficus/ --data_output_path dataset/Nerf_Synthetic_eps16/ficus/ &
python attacker/poison_splat_bounded.py --gpu 3 --adv_epsilon 16 --adv_iters 36000 --adv_image_search_iters 15 \
    --data_path dataset/Nerf_Synthetic/hotdog/ --decoy_log_path log/01_main_exp/attacker_eps16_nerf_synthetic/hotdog/ --data_output_path dataset/Nerf_Synthetic_eps16/hotdog/ &
python attacker/poison_splat_bounded.py --gpu 4 --adv_epsilon 16 --adv_iters 36000 --adv_image_search_iters 15 \
    --data_path dataset/Nerf_Synthetic/lego/ --decoy_log_path log/01_main_exp/attacker_eps16_nerf_synthetic/lego/ --data_output_path dataset/Nerf_Synthetic_eps16/lego/ &
python attacker/poison_splat_bounded.py --gpu 5 --adv_epsilon 16 --adv_iters 36000 --adv_image_search_iters 15 \
    --data_path dataset/Nerf_Synthetic/materials/ --decoy_log_path log/01_main_exp/attacker_eps16_nerf_synthetic/materials/ --data_output_path dataset/Nerf_Synthetic_eps16/materials/ &
python attacker/poison_splat_bounded.py --gpu 6 --adv_epsilon 16 --adv_iters 36000 --adv_image_search_iters 15 \
    --data_path dataset/Nerf_Synthetic/mic/ --decoy_log_path log/01_main_exp/attacker_eps16_nerf_synthetic/mic/ --data_output_path dataset/Nerf_Synthetic_eps16/mic/ &
python attacker/poison_splat_bounded.py --gpu 7 --adv_epsilon 16 --adv_iters 36000 --adv_image_search_iters 15 \
    --data_path dataset/Nerf_Synthetic/ship/ --decoy_log_path log/01_main_exp/attacker_eps16_nerf_synthetic/ship/ --data_output_path dataset/Nerf_Synthetic_eps16/ship/
wait
python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/Nerf_Synthetic_eps16/chair/ -m log/01_main_exp/victim_gs_nerf_synthetic_eps16/chair/ &
python victim/gaussian-splatting/benchmark.py --gpu 1\
    -s dataset/Nerf_Synthetic_eps16/drums/ -m log/01_main_exp/victim_gs_nerf_synthetic_eps16/drums/ &
python victim/gaussian-splatting/benchmark.py --gpu 2\
    -s dataset/Nerf_Synthetic_eps16/ficus/ -m log/01_main_exp/victim_gs_nerf_synthetic_eps16/ficus/ &
python victim/gaussian-splatting/benchmark.py --gpu 3\
    -s dataset/Nerf_Synthetic_eps16/hotdog/ -m log/01_main_exp/victim_gs_nerf_synthetic_eps16/hotdog/ &
python victim/gaussian-splatting/benchmark.py --gpu 4\
    -s dataset/Nerf_Synthetic_eps16/lego/ -m log/01_main_exp/victim_gs_nerf_synthetic_eps16/lego/ &
python victim/gaussian-splatting/benchmark.py --gpu 5\
    -s dataset/Nerf_Synthetic_eps16/materials/ -m log/01_main_exp/victim_gs_nerf_synthetic_eps16/materials/ &
python victim/gaussian-splatting/benchmark.py --gpu 6\
    -s dataset/Nerf_Synthetic_eps16/mic/ -m log/01_main_exp/victim_gs_nerf_synthetic_eps16/mic/ &
python victim/gaussian-splatting/benchmark.py --gpu 7\
    -s dataset/Nerf_Synthetic_eps16/ship/ -m log/01_main_exp/victim_gs_nerf_synthetic_eps16/ship/
wait