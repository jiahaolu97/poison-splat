python attacker/poison_splat_unbounded.py --gpu 0 --adv_iters 6000 --adv_decoy_update_interval 10\
    --data_path dataset/Tanks_and_Temples/Panther/ --decoy_log_path log/01_main_exp/attacker_unbounded_tanks_and_temples/Panther/ --data_output_path dataset/Tanks_and_Temples_unbounded/Panther/ &
python attacker/poison_splat_unbounded.py --gpu 1 --adv_iters 6000 --adv_decoy_update_interval 10\
    --data_path dataset/Tanks_and_Temples/Playground/ --decoy_log_path log/01_main_exp/attacker_unbounded_tanks_and_temples/Playground/ --data_output_path dataset/Tanks_and_Temples_unbounded/Playground/ &
python attacker/poison_splat_unbounded.py --gpu 2 --adv_iters 6000 --adv_decoy_update_interval 10\
    --data_path dataset/Tanks_and_Temples/Temple/ --decoy_log_path log/01_main_exp/attacker_unbounded_tanks_and_temples/Temple/ --data_output_path dataset/Tanks_and_Temples_unbounded/Temple/ &
python attacker/poison_splat_unbounded.py --gpu 3 --adv_iters 6000 --adv_decoy_update_interval 10\
    --data_path dataset/Tanks_and_Temples/Train/ --decoy_log_path log/01_main_exp/attacker_unbounded_tanks_and_temples/Train/ --data_output_path dataset/Tanks_and_Temples_unbounded/Train/ &
python attacker/poison_splat_unbounded.py --gpu 4 --adv_iters 6000 --adv_decoy_update_interval 10\
    --data_path dataset/Tanks_and_Temples/Truck/ --decoy_log_path log/01_main_exp/attacker_unbounded_tanks_and_temples/Truck/ --data_output_path dataset/Tanks_and_Temples_unbounded/Truck/ 
wait
python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/Tanks_and_Temples_unbounded/Panther/ -m log/01_main_exp/victim_gs_tanks_and_temples_unbounded/Panther/ &
python victim/gaussian-splatting/benchmark.py --gpu 1\
    -s dataset/Tanks_and_Temples_unbounded/Playground/ -m log/01_main_exp/victim_gs_tanks_and_temples_unbounded/Playground/ &
python victim/gaussian-splatting/benchmark.py --gpu 2\
    -s dataset/Tanks_and_Temples_unbounded/Temple/ -m log/01_main_exp/victim_gs_tanks_and_temples_unbounded/Temple/ &
python victim/gaussian-splatting/benchmark.py --gpu 3\
    -s dataset/Tanks_and_Temples_unbounded/Train/ -m log/01_main_exp/victim_gs_tanks_and_temples_unbounded/Train/ &
python victim/gaussian-splatting/benchmark.py --gpu 4\
    -s dataset/Tanks_and_Temples_unbounded/Truck/ -m log/01_main_exp/victim_gs_tanks_and_temples_unbounded/Truck/ 
wait