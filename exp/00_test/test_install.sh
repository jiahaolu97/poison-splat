# go to the root directory of this project
# python attacker/poison_splat_bounded.py --gpu 0 --adv_epsilon 16  --adv_image_search_iters 25 --adv_iters 600\
#                     --data_path dataset/Nerf_Synthetic/chair/ \
#                     --data_output_path dataset/Nerf_Synthetic_eps16/chair/ \
#                     --decoy_log_path log/test_install/attacker-bounded/testrun/ \
#                     --adv_proxy_model_path /mnt/data/jiahaolu/dev-poison-splat/rebuttal_exp_output/nerf_synthetic_clean/chair/victim_model.ply

python victim/gaussian-splatting/benchmark.py --gpu 0 --iterations 300\
    -s dataset/Nerf_Synthetic/chair/ -m log/test_install/victim_gs/testrun/

python victim/Scaffold-GS/benchmark.py --gpu 0 --iterations 300\
    -s dataset/Nerf_Synthetic/chair/ -m log/test_install/victim_scaffold/testrun/

echo SUCCESS