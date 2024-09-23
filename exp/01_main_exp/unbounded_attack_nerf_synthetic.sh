python attacker/poison_splat_unbounded.py --gpu 0 --adv_decoy_update_interval 10 \
    --data_path dataset/Nerf_Synthetic/chair/ --decoy_log_path log/01_main_exp/attacker_unbounded_nerf_synthetic/chair/ --data_output_path dataset/Nerf_Synthetic_unbounded/chair/ &
python attacker/poison_splat_unbounded.py --gpu 1 --adv_decoy_update_interval 10 \
    --data_path dataset/Nerf_Synthetic/drums/ --decoy_log_path log/01_main_exp/attacker_unbounded_nerf_synthetic/drums/ --data_output_path dataset/Nerf_Synthetic_unbounded/drums/ &
python attacker/poison_splat_unbounded.py --gpu 2 --adv_decoy_update_interval 10 \
    --data_path dataset/Nerf_Synthetic/ficus/ --decoy_log_path log/01_main_exp/attacker_unbounded_nerf_synthetic/ficus/ --data_output_path dataset/Nerf_Synthetic_unbounded/ficus/ &
python attacker/poison_splat_unbounded.py --gpu 3 --adv_decoy_update_interval 10 \
    --data_path dataset/Nerf_Synthetic/hotdog/ --decoy_log_path log/01_main_exp/attacker_unbounded_nerf_synthetic/hotdog/ --data_output_path dataset/Nerf_Synthetic_unbounded/hotdog/ &
python attacker/poison_splat_unbounded.py --gpu 4 --adv_decoy_update_interval 10 \
    --data_path dataset/Nerf_Synthetic/lego/ --decoy_log_path log/01_main_exp/attacker_unbounded_nerf_synthetic/lego/ --data_output_path dataset/Nerf_Synthetic_unbounded/lego/ &
python attacker/poison_splat_unbounded.py --gpu 5 --adv_decoy_update_interval 10 \
    --data_path dataset/Nerf_Synthetic/materials/ --decoy_log_path log/01_main_exp/attacker_unbounded_nerf_synthetic/materials/ --data_output_path dataset/Nerf_Synthetic_unbounded/materials/ &
python attacker/poison_splat_unbounded.py --gpu 6 --adv_decoy_update_interval 10 \
    --data_path dataset/Nerf_Synthetic/mic/ --decoy_log_path log/01_main_exp/attacker_unbounded_nerf_synthetic/mic/ --data_output_path dataset/Nerf_Synthetic_unbounded/mic/ &
python attacker/poison_splat_unbounded.py --gpu 7 --adv_decoy_update_interval 10 \
    --data_path dataset/Nerf_Synthetic/ship/ --decoy_log_path log/01_main_exp/attacker_unbounded_nerf_synthetic/ship/ --data_output_path dataset/Nerf_Synthetic_unbounded/ship/
wait
python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/Nerf_Synthetic_unbounded/chair/ -m log/01_main_exp/victim_gs_nerf_synthetic_unbounded/chair/ &
python victim/gaussian-splatting/benchmark.py --gpu 1\
    -s dataset/Nerf_Synthetic_unbounded/drums/ -m log/01_main_exp/victim_gs_nerf_synthetic_unbounded/drums/ &
python victim/gaussian-splatting/benchmark.py --gpu 2\
    -s dataset/Nerf_Synthetic_unbounded/ficus/ -m log/01_main_exp/victim_gs_nerf_synthetic_unbounded/ficus/ &
python victim/gaussian-splatting/benchmark.py --gpu 3\
    -s dataset/Nerf_Synthetic_unbounded/hotdog/ -m log/01_main_exp/victim_gs_nerf_synthetic_unbounded/hotdog/ &
python victim/gaussian-splatting/benchmark.py --gpu 4\
    -s dataset/Nerf_Synthetic_unbounded/lego/ -m log/01_main_exp/victim_gs_nerf_synthetic_unbounded/lego/ &
python victim/gaussian-splatting/benchmark.py --gpu 5\
    -s dataset/Nerf_Synthetic_unbounded/materials/ -m log/01_main_exp/victim_gs_nerf_synthetic_unbounded/materials/ &
python victim/gaussian-splatting/benchmark.py --gpu 6\
    -s dataset/Nerf_Synthetic_unbounded/mic/ -m log/01_main_exp/victim_gs_nerf_synthetic_unbounded/mic/ &
python victim/gaussian-splatting/benchmark.py --gpu 7\
    -s dataset/Nerf_Synthetic_unbounded/ship/ -m log/01_main_exp/victim_gs_nerf_synthetic_unbounded/ship/
wait