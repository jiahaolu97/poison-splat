python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/MIP_Nerf_360/bicycle/ -m log/01_main_exp/victim_gs_mip_nerf_360_clean/bicycle/ &
python victim/gaussian-splatting/benchmark.py --gpu 1\
    -s dataset/MIP_Nerf_360/bonsai/ -m log/01_main_exp/victim_gs_mip_nerf_360_clean/bonsai/ &
python victim/gaussian-splatting/benchmark.py --gpu 2\
    -s dataset/MIP_Nerf_360/counter/ -m log/01_main_exp/victim_gs_mip_nerf_360_clean/counter/ &
python victim/gaussian-splatting/benchmark.py --gpu 3\
    -s dataset/MIP_Nerf_360/flowers/ -m log/01_main_exp/victim_gs_mip_nerf_360_clean/flowers/ &
python victim/gaussian-splatting/benchmark.py --gpu 4\
    -s dataset/MIP_Nerf_360/garden/ -m log/01_main_exp/victim_gs_mip_nerf_360_clean/garden/ &
python victim/gaussian-splatting/benchmark.py --gpu 5\
    -s dataset/MIP_Nerf_360/kitchen/ -m log/01_main_exp/victim_gs_mip_nerf_360_clean/kitchen/ &
python victim/gaussian-splatting/benchmark.py --gpu 6\
    -s dataset/MIP_Nerf_360/room/ -m log/01_main_exp/victim_gs_mip_nerf_360_clean/room/ &
python victim/gaussian-splatting/benchmark.py --gpu 7\
    -s dataset/MIP_Nerf_360/stump/ -m log/01_main_exp/victim_gs_mip_nerf_360_clean/stump/
wait
python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/MIP_Nerf_360/treehill/ -m log/01_main_exp/victim_gs_mip_nerf_360_clean/treehill/