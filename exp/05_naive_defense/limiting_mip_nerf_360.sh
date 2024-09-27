python victim/gaussian-splatting/defense/naive_limiting_gaussian_number.py --gpu 0 --max_gaussian_num 6000000\
    -s dataset/MIP_Nerf_360_eps16/bicycle/ -m log/05_naive_defense/limit_gaussian/mip_360_eps16/bicycle/ &
python victim/gaussian-splatting/defense/naive_limiting_gaussian_number.py --gpu 1 --max_gaussian_num 2000000\
    -s dataset/MIP_Nerf_360_eps16/bonsai/ -m log/05_naive_defense/limit_gaussian/mip_360_eps16/bonsai/ &
python victim/gaussian-splatting/defense/naive_limiting_gaussian_number.py --gpu 2 --max_gaussian_num 2000000\
    -s dataset/MIP_Nerf_360_eps16/counter/ -m log/05_naive_defense/limit_gaussian/mip_360_eps16/counter/ &
python victim/gaussian-splatting/defense/naive_limiting_gaussian_number.py --gpu 3 --max_gaussian_num 2000000\
    -s dataset/MIP_Nerf_360_eps16/kitchen/ -m log/05_naive_defense/limit_gaussian/mip_360_eps16/kitchen/ &
python victim/gaussian-splatting/defense/naive_limiting_gaussian_number.py --gpu 4 --max_gaussian_num 2000000\
    -s dataset/MIP_Nerf_360_eps16/room/ -m log/05_naive_defense/limit_gaussian/mip_360_eps16/room/ &
python victim/gaussian-splatting/defense/naive_limiting_gaussian_number.py --gpu 5 --max_gaussian_num 5000000\
    -s dataset/MIP_Nerf_360_eps16/stump/ -m log/05_naive_defense/limit_gaussian/mip_360_eps16/stump/ &
python victim/gaussian-splatting/defense/naive_limiting_gaussian_number.py --gpu 6 --max_gaussian_num 4000000\
    -s dataset/MIP_Nerf_360_eps16/treehill/ -m log/05_naive_defense/limit_gaussian/mip_360_eps16/treehill/ &
python victim/gaussian-splatting/defense/naive_limiting_gaussian_number.py --gpu 7 --max_gaussian_num 4000000\
    -s dataset/MIP_Nerf_360_eps16/flowers/ -m log/05_naive_defense/limit_gaussian/mip_360_eps16/flowers/
wait