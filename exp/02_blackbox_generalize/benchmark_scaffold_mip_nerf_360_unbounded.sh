python victim/Scaffold-GS/benchmark.py --gpu 0\
    -s dataset/MIP_Nerf_360_unbounded/bicycle/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_unbounded/bicycle/ &
python victim/Scaffold-GS/benchmark.py --gpu 1\
    -s dataset/MIP_Nerf_360_unbounded/bonsai/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_unbounded/bonsai/ &
python victim/Scaffold-GS/benchmark.py --gpu 2\
    -s dataset/MIP_Nerf_360_unbounded/counter/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_unbounded/counter/ &
python victim/Scaffold-GS/benchmark.py --gpu 3\
    -s dataset/MIP_Nerf_360_unbounded/flowers/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_unbounded/flowers/ &
python victim/Scaffold-GS/benchmark.py --gpu 4\
    -s dataset/MIP_Nerf_360_unbounded/garden/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_unbounded/garden/ &
python victim/Scaffold-GS/benchmark.py --gpu 5\
    -s dataset/MIP_Nerf_360_unbounded/kitchen/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_unbounded/kitchen/ &
python victim/Scaffold-GS/benchmark.py --gpu 6\
    -s dataset/MIP_Nerf_360_unbounded/room/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_unbounded/room/ &
python victim/Scaffold-GS/benchmark.py --gpu 7\
    -s dataset/MIP_Nerf_360_unbounded/stump/ -m log/02_blackbox_generalize/victim_scaffold_mip_nerf_360_unbounded/stump/
wait