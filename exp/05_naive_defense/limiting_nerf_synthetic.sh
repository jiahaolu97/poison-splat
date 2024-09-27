python victim/gaussian-splatting/defense/naive_limiting_gaussian_number.py --gpu 0 --max_gaussian_num 500000\
    -s dataset/Nerf_Synthetic_eps16/chair/ -m log/05_naive_defense/limit_gaussian/nerf_synthetic_eps16/chair/ &
python victim/gaussian-splatting/defense/naive_limiting_gaussian_number.py --gpu 1 --max_gaussian_num 500000\
    -s dataset/Nerf_Synthetic_eps16/hotdog/ -m log/05_naive_defense/limit_gaussian/nerf_synthetic_eps16/hotdog/ &
python victim/gaussian-splatting/defense/naive_limiting_gaussian_number.py --gpu 2 --max_gaussian_num 500000\
    -s dataset/Nerf_Synthetic_eps16/lego/ -m log/05_naive_defense/limit_gaussian/nerf_synthetic_eps16/lego/ &
python victim/gaussian-splatting/defense/naive_limiting_gaussian_number.py --gpu 3 --max_gaussian_num 500000\
    -s dataset/Nerf_Synthetic_eps16/ship/ -m log/05_naive_defense/limit_gaussian/nerf_synthetic_eps16/ship/
wait