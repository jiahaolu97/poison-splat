python attacker/poison_splat_bounded.py --gpu 0 --adv_epsilon 16 --adv_iters 36000\
    --data_path dataset/Tanks_and_Temples/Panther/ --decoy_log_path log/01_main_exp/attacker_eps16_tanks_and_temples/Panther/ --data_output_path dataset/Tanks_and_Temples_eps16/Panther/ &
python attacker/poison_splat_bounded.py --gpu 1 --adv_epsilon 16 --adv_iters 36000\
    --data_path dataset/Tanks_and_Temples/Playground/ --decoy_log_path log/01_main_exp/attacker_eps16_tanks_and_temples/Playground/ --data_output_path dataset/Tanks_and_Temples_eps16/Playground/ &
python attacker/poison_splat_bounded.py --gpu 2 --adv_epsilon 16 --adv_iters 36000\
    --data_path dataset/Tanks_and_Temples/Temple/ --decoy_log_path log/01_main_exp/attacker_eps16_tanks_and_temples/Temple/ --data_output_path dataset/Tanks_and_Temples_eps16/Temple/ &
python attacker/poison_splat_bounded.py --gpu 3 --adv_epsilon 16 --adv_iters 36000\
    --data_path dataset/Tanks_and_Temples/Train/ --decoy_log_path log/01_main_exp/attacker_eps16_tanks_and_temples/Train/ --data_output_path dataset/Tanks_and_Temples_eps16/Train/ &
python attacker/poison_splat_bounded.py --gpu 4 --adv_epsilon 16 --adv_iters 36000\
    --data_path dataset/Tanks_and_Temples/Truck/ --decoy_log_path log/01_main_exp/attacker_eps16_tanks_and_temples/Truck/ --data_output_path dataset/Tanks_and_Temples_eps16/Truck/ 
wait
python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/Tanks_and_Temples_eps16/Panther/ -m log/01_main_exp/victim_gs_tanks_and_temples_eps16/Panther/ &
python victim/gaussian-splatting/benchmark.py --gpu 1\
    -s dataset/Tanks_and_Temples_eps16/Playground/ -m log/01_main_exp/victim_gs_tanks_and_temples_eps16/Playground/ &
python victim/gaussian-splatting/benchmark.py --gpu 2\
    -s dataset/Tanks_and_Temples_eps16/Temple/ -m log/01_main_exp/victim_gs_tanks_and_temples_eps16/Temple/ &
python victim/gaussian-splatting/benchmark.py --gpu 3\
    -s dataset/Tanks_and_Temples_eps16/Train/ -m log/01_main_exp/victim_gs_tanks_and_temples_eps16/Train/ &
python victim/gaussian-splatting/benchmark.py --gpu 4\
    -s dataset/Tanks_and_Temples_eps16/Truck/ -m log/01_main_exp/victim_gs_tanks_and_temples_eps16/Truck/ 
wait