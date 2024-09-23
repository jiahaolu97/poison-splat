python attacker/poison_splat_unbounded.py --gpu 0 \
    --data_path dataset/MIP_Nerf_360/bicycle/ --decoy_log_path log/01_main_exp/attacker_unbounded_mip_nerf_360/bicycle/ --data_output_path dataset/MIP_Nerf_360_unbounded/bicycle/ &
python attacker/poison_splat_unbounded.py --gpu 1 \
    --data_path dataset/MIP_Nerf_360/bonsai/ --decoy_log_path log/01_main_exp/attacker_unbounded_mip_nerf_360/bonsai/ --data_output_path dataset/MIP_Nerf_360_unbounded/bonsai/ &
python attacker/poison_splat_unbounded.py --gpu 2 \
    --data_path dataset/MIP_Nerf_360/counter/ --decoy_log_path log/01_main_exp/attacker_unbounded_mip_nerf_360/counter/ --data_output_path dataset/MIP_Nerf_360_unbounded/counter/ &
python attacker/poison_splat_unbounded.py --gpu 3 \
    --data_path dataset/MIP_Nerf_360/flowers/ --decoy_log_path log/01_main_exp/attacker_unbounded_mip_nerf_360/flowers/ --data_output_path dataset/MIP_Nerf_360_unbounded/flowers/ &
python attacker/poison_splat_unbounded.py --gpu 4 \
    --data_path dataset/MIP_Nerf_360/garden/ --decoy_log_path log/01_main_exp/attacker_unbounded_mip_nerf_360/garden/ --data_output_path dataset/MIP_Nerf_360_unbounded/garden/ &
python attacker/poison_splat_unbounded.py --gpu 5 \
    --data_path dataset/MIP_Nerf_360/kitchen/ --decoy_log_path log/01_main_exp/attacker_unbounded_mip_nerf_360/kitchen/ --data_output_path dataset/MIP_Nerf_360_unbounded/kitchen/ &
python attacker/poison_splat_unbounded.py --gpu 6 \
    --data_path dataset/MIP_Nerf_360/room/ --decoy_log_path log/01_main_exp/attacker_unbounded_mip_nerf_360/room/ --data_output_path dataset/MIP_Nerf_360_unbounded/room/ &
python attacker/poison_splat_unbounded.py --gpu 7 \
    --data_path dataset/MIP_Nerf_360/stump/ --decoy_log_path log/01_main_exp/attacker_unbounded_mip_nerf_360/stump/ --data_output_path dataset/MIP_Nerf_360_unbounded/stump/
wait
python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/MIP_Nerf_360_unbounded/bicycle/ -m log/01_main_exp/victim_gs_mip_nerf_360_unbounded/bicycle/ &
python victim/gaussian-splatting/benchmark.py --gpu 1\
    -s dataset/MIP_Nerf_360_unbounded/bonsai/ -m log/01_main_exp/victim_gs_mip_nerf_360_unbounded/bonsai/ &
python victim/gaussian-splatting/benchmark.py --gpu 2\
    -s dataset/MIP_Nerf_360_unbounded/counter/ -m log/01_main_exp/victim_gs_mip_nerf_360_unbounded/counter/ &
python victim/gaussian-splatting/benchmark.py --gpu 3\
    -s dataset/MIP_Nerf_360_unbounded/flowers/ -m log/01_main_exp/victim_gs_mip_nerf_360_unbounded/flowers/ &
python victim/gaussian-splatting/benchmark.py --gpu 4\
    -s dataset/MIP_Nerf_360_unbounded/garden/ -m log/01_main_exp/victim_gs_mip_nerf_360_unbounded/garden/ &
python victim/gaussian-splatting/benchmark.py --gpu 5\
    -s dataset/MIP_Nerf_360_unbounded/kitchen/ -m log/01_main_exp/victim_gs_mip_nerf_360_unbounded/kitchen/ &
python victim/gaussian-splatting/benchmark.py --gpu 6\
    -s dataset/MIP_Nerf_360_unbounded/room/ -m log/01_main_exp/victim_gs_mip_nerf_360_unbounded/room/ &
python victim/gaussian-splatting/benchmark.py --gpu 7\
    -s dataset/MIP_Nerf_360_unbounded/stump/ -m log/01_main_exp/victim_gs_mip_nerf_360_unbounded/stump/
wait
python attacker/poison_splat_unbounded.py --gpu 0 \
    --data_path dataset/MIP_Nerf_360/treehill/ --decoy_log_path log/01_main_exp/attacker_unbounded_mip_nerf_360/treehill/ --data_output_path dataset/MIP_Nerf_360_unbounded/treehill/ 
python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/MIP_Nerf_360_unbounded/treehill/ -m log/01_main_exp/victim_gs_mip_nerf_360_unbounded/treehill/