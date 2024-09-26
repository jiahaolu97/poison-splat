python victim/Scaffold-GS/benchmark.py --gpu 0\
    -s dataset/MIP_Nerf_360_eps16/bicycle/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_eps16/bicycle/ &
python victim/Scaffold-GS/benchmark.py --gpu 1\
    -s dataset/MIP_Nerf_360_eps16/bonsai/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_eps16/bonsai/ &
python victim/Scaffold-GS/benchmark.py --gpu 2\
    -s dataset/MIP_Nerf_360_eps16/counter/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_eps16/counter/ &
python victim/Scaffold-GS/benchmark.py --gpu 3\
    -s dataset/MIP_Nerf_360_eps16/flowers/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_eps16/flowers/ &
python victim/Scaffold-GS/benchmark.py --gpu 4\
    -s dataset/MIP_Nerf_360_eps16/garden/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_eps16/garden/ &
python victim/Scaffold-GS/benchmark.py --gpu 5\
    -s dataset/MIP_Nerf_360_eps16/kitchen/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_eps16/kitchen/ &
python victim/Scaffold-GS/benchmark.py --gpu 6\
    -s dataset/MIP_Nerf_360_eps16/room/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_eps16/room/ &
python victim/Scaffold-GS/benchmark.py --gpu 7\
    -s dataset/MIP_Nerf_360_eps16/stump/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_eps16/stump/
wait