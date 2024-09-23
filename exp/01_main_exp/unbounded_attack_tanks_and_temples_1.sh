python attacker/poison_splat_unbounded.py --gpu 0 --adv_iters 6000 --adv_decoy_update_interval 10\
    --data_path dataset/Tanks_and_Temples/Auditorium/ --decoy_log_path log/01_main_exp/attacker_unbounded_tanks_and_temples/Auditorium/ --data_output_path dataset/Tanks_and_Temples_unbounded/Auditorium/ &
python attacker/poison_splat_unbounded.py --gpu 1 --adv_iters 6000 --adv_decoy_update_interval 10\
    --data_path dataset/Tanks_and_Temples/Ballroom/ --decoy_log_path log/01_main_exp/attacker_unbounded_tanks_and_temples/Ballroom/ --data_output_path dataset/Tanks_and_Temples_unbounded/Ballroom/ &
python attacker/poison_splat_unbounded.py --gpu 2 --adv_iters 6000 --adv_decoy_update_interval 10\
    --data_path dataset/Tanks_and_Temples/Barn/ --decoy_log_path log/01_main_exp/attacker_unbounded_tanks_and_temples/Barn/ --data_output_path dataset/Tanks_and_Temples_unbounded/Barn/ &
python attacker/poison_splat_unbounded.py --gpu 3 --adv_iters 6000 --adv_decoy_update_interval 10\
    --data_path dataset/Tanks_and_Temples/Caterpillar/ --decoy_log_path log/01_main_exp/attacker_unbounded_tanks_and_temples/Caterpillar/ --data_output_path dataset/Tanks_and_Temples_unbounded/Caterpillar/ &
python attacker/poison_splat_unbounded.py --gpu 4 --adv_iters 6000 --adv_decoy_update_interval 10\
    --data_path dataset/Tanks_and_Temples/Church/ --decoy_log_path log/01_main_exp/attacker_unbounded_tanks_and_temples/Church/ --data_output_path dataset/Tanks_and_Temples_unbounded/Church/ &
python attacker/poison_splat_unbounded.py --gpu 5 --adv_iters 6000 --adv_decoy_update_interval 10\
    --data_path dataset/Tanks_and_Temples/Courthouse/ --decoy_log_path log/01_main_exp/attacker_unbounded_tanks_and_temples/Courthouse/ --data_output_path dataset/Tanks_and_Temples_unbounded/Courthouse/ &
python attacker/poison_splat_unbounded.py --gpu 6 --adv_iters 6000 --adv_decoy_update_interval 10\
    --data_path dataset/Tanks_and_Temples/Courtroom/ --decoy_log_path log/01_main_exp/attacker_unbounded_tanks_and_temples/Courtroom/ --data_output_path dataset/Tanks_and_Temples_unbounded/Courtroom/ &
python attacker/poison_splat_unbounded.py --gpu 7 --adv_iters 6000 --adv_decoy_update_interval 10\
    --data_path dataset/Tanks_and_Temples/Family/ --decoy_log_path log/01_main_exp/attacker_unbounded_tanks_and_temples/Family/ --data_output_path dataset/Tanks_and_Temples_unbounded/Family/
wait
python victim/gaussian-splatting/benchmark.py --gpu 0 \
    -s dataset/Tanks_and_Temples_unbounded/Auditorium/ -m log/01_main_exp/victim_gs_tanks_and_temples_unbounded/Auditorium/ &
python victim/gaussian-splatting/benchmark.py --gpu 1\
    -s dataset/Tanks_and_Temples_unbounded/Ballroom/ -m log/01_main_exp/victim_gs_tanks_and_temples_unbounded/Ballroom/ &
python victim/gaussian-splatting/benchmark.py --gpu 2\
    -s dataset/Tanks_and_Temples_unbounded/Barn/ -m log/01_main_exp/victim_gs_tanks_and_temples_unbounded/Barn/ &
python victim/gaussian-splatting/benchmark.py --gpu 3\
    -s dataset/Tanks_and_Temples_unbounded/Caterpillar/ -m log/01_main_exp/victim_gs_tanks_and_temples_unbounded/Caterpillar/ &
python victim/gaussian-splatting/benchmark.py --gpu 4\
    -s dataset/Tanks_and_Temples_unbounded/Church/ -m log/01_main_exp/victim_gs_tanks_and_temples_unbounded/Church/ &
python victim/gaussian-splatting/benchmark.py --gpu 5\
    -s dataset/Tanks_and_Temples_unbounded/Courthouse/ -m log/01_main_exp/victim_gs_tanks_and_temples_unbounded/Courthouse/ &
python victim/gaussian-splatting/benchmark.py --gpu 6\
    -s dataset/Tanks_and_Temples_unbounded/Courtroom/ -m log/01_main_exp/victim_gs_tanks_and_temples_unbounded/Courtroom/ &
python victim/gaussian-splatting/benchmark.py --gpu 7\
    -s dataset/Tanks_and_Temples_unbounded/Family/ -m log/01_main_exp/victim_gs_tanks_and_temples_unbounded/Family/
wait