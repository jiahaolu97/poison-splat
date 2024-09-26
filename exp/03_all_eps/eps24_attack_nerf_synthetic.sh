python attacker/poison_splat_bounded.py --gpu 0 --adv_epsilon 24 --adv_iters 24000 \
    --data_path dataset/Nerf_Synthetic/chair/ --decoy_log_path log/01_main_exp/attacker_eps24_nerf_synthetic/chair/ --data_output_path dataset/Nerf_Synthetic_eps24/chair/ &
python attacker/poison_splat_bounded.py --gpu 1 --adv_epsilon 24 --adv_iters 24000 \
    --data_path dataset/Nerf_Synthetic/drums/ --decoy_log_path log/01_main_exp/attacker_eps24_nerf_synthetic/drums/ --data_output_path dataset/Nerf_Synthetic_eps24/drums/ &
python attacker/poison_splat_bounded.py --gpu 2 --adv_epsilon 24 --adv_iters 24000 \
    --data_path dataset/Nerf_Synthetic/ficus/ --decoy_log_path log/01_main_exp/attacker_eps24_nerf_synthetic/ficus/ --data_output_path dataset/Nerf_Synthetic_eps24/ficus/ &
python attacker/poison_splat_bounded.py --gpu 3 --adv_epsilon 24 --adv_iters 16000 \
    --data_path dataset/Nerf_Synthetic/hotdog/ --decoy_log_path log/01_main_exp/attacker_eps24_nerf_synthetic/hotdog/ --data_output_path dataset/Nerf_Synthetic_eps24/hotdog/ &
python attacker/poison_splat_bounded.py --gpu 4 --adv_epsilon 24 --adv_iters 24000 \
    --data_path dataset/Nerf_Synthetic/lego/ --decoy_log_path log/01_main_exp/attacker_eps24_nerf_synthetic/lego/ --data_output_path dataset/Nerf_Synthetic_eps24/lego/ &
python attacker/poison_splat_bounded.py --gpu 5 --adv_epsilon 24 --adv_iters 24000 \
    --data_path dataset/Nerf_Synthetic/materials/ --decoy_log_path log/01_main_exp/attacker_eps24_nerf_synthetic/materials/ --data_output_path dataset/Nerf_Synthetic_eps24/materials/ &
python attacker/poison_splat_bounded.py --gpu 6 --adv_epsilon 24 --adv_iters 24000 \
    --data_path dataset/Nerf_Synthetic/mic/ --decoy_log_path log/01_main_exp/attacker_eps24_nerf_synthetic/mic/ --data_output_path dataset/Nerf_Synthetic_eps24/mic/ &
python attacker/poison_splat_bounded.py --gpu 7 --adv_epsilon 24 --adv_iters 24000 \
    --data_path dataset/Nerf_Synthetic/ship/ --decoy_log_path log/01_main_exp/attacker_eps24_nerf_synthetic/ship/ --data_output_path dataset/Nerf_Synthetic_eps24/ship/
wait
python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/Nerf_Synthetic_eps24/chair/ -m log/01_main_exp/victim_gs_nerf_synthetic_eps24/chair/ &
python victim/gaussian-splatting/benchmark.py --gpu 1\
    -s dataset/Nerf_Synthetic_eps24/drums/ -m log/01_main_exp/victim_gs_nerf_synthetic_eps24/drums/ &
python victim/gaussian-splatting/benchmark.py --gpu 2\
    -s dataset/Nerf_Synthetic_eps24/ficus/ -m log/01_main_exp/victim_gs_nerf_synthetic_eps24/ficus/ &
python victim/gaussian-splatting/benchmark.py --gpu 3\
    -s dataset/Nerf_Synthetic_eps24/hotdog/ -m log/01_main_exp/victim_gs_nerf_synthetic_eps24/hotdog/ &
python victim/gaussian-splatting/benchmark.py --gpu 4\
    -s dataset/Nerf_Synthetic_eps24/lego/ -m log/01_main_exp/victim_gs_nerf_synthetic_eps24/lego/ &
python victim/gaussian-splatting/benchmark.py --gpu 5\
    -s dataset/Nerf_Synthetic_eps24/materials/ -m log/01_main_exp/victim_gs_nerf_synthetic_eps24/materials/ &
python victim/gaussian-splatting/benchmark.py --gpu 6\
    -s dataset/Nerf_Synthetic_eps24/mic/ -m log/01_main_exp/victim_gs_nerf_synthetic_eps24/mic/ &
python victim/gaussian-splatting/benchmark.py --gpu 7\
    -s dataset/Nerf_Synthetic_eps24/ship/ -m log/01_main_exp/victim_gs_nerf_synthetic_eps24/ship/
wait