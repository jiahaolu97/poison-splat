python victim/Scaffold-GS/benchmark.py --gpu 0\
    -s dataset/MIP_Nerf_360/bicycle/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_clean/bicycle/ &
python victim/Scaffold-GS/benchmark.py --gpu 1\
    -s dataset/MIP_Nerf_360/bonsai/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_clean/bonsai/ &
python victim/Scaffold-GS/benchmark.py --gpu 2\
    -s dataset/MIP_Nerf_360/counter/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_clean/counter/ &
python victim/Scaffold-GS/benchmark.py --gpu 3\
    -s dataset/MIP_Nerf_360/flowers/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_clean/flowers/ &
python victim/Scaffold-GS/benchmark.py --gpu 4\
    -s dataset/MIP_Nerf_360/garden/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_clean/garden/ &
python victim/Scaffold-GS/benchmark.py --gpu 5\
    -s dataset/MIP_Nerf_360/kitchen/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_clean/kitchen/ &
python victim/Scaffold-GS/benchmark.py --gpu 6\
    -s dataset/MIP_Nerf_360/room/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_clean/room/ &
python victim/Scaffold-GS/benchmark.py --gpu 7\
    -s dataset/MIP_Nerf_360/stump/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_clean/stump/
wait