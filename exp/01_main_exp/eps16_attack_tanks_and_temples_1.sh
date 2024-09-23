python attacker/poison_splat_bounded.py --gpu 0 --adv_epsilon 16 --adv_iters 36000\
    --data_path dataset/Tanks_and_Temples/Auditorium/ --decoy_log_path log/01_main_exp/attacker_eps16_tanks_and_temples/Auditorium/ --data_output_path dataset/Tanks_and_Temples_eps16/Auditorium/ &
python attacker/poison_splat_bounded.py --gpu 1 --adv_epsilon 16 --adv_iters 36000\
    --data_path dataset/Tanks_and_Temples/Ballroom/ --decoy_log_path log/01_main_exp/attacker_eps16_tanks_and_temples/Ballroom/ --data_output_path dataset/Tanks_and_Temples_eps16/Ballroom/ &
python attacker/poison_splat_bounded.py --gpu 2 --adv_epsilon 16 --adv_iters 36000\
    --data_path dataset/Tanks_and_Temples/Barn/ --decoy_log_path log/01_main_exp/attacker_eps16_tanks_and_temples/Barn/ --data_output_path dataset/Tanks_and_Temples_eps16/Barn/ &
python attacker/poison_splat_bounded.py --gpu 3 --adv_epsilon 16 --adv_iters 36000\
    --data_path dataset/Tanks_and_Temples/Caterpillar/ --decoy_log_path log/01_main_exp/attacker_eps16_tanks_and_temples/Caterpillar/ --data_output_path dataset/Tanks_and_Temples_eps16/Caterpillar/ &
python attacker/poison_splat_bounded.py --gpu 4 --adv_epsilon 16 --adv_iters 36000\
    --data_path dataset/Tanks_and_Temples/Church/ --decoy_log_path log/01_main_exp/attacker_eps16_tanks_and_temples/Church/ --data_output_path dataset/Tanks_and_Temples_eps16/Church/ &
python attacker/poison_splat_bounded.py --gpu 5 --adv_epsilon 16 --adv_iters 36000\
    --data_path dataset/Tanks_and_Temples/Courthouse/ --decoy_log_path log/01_main_exp/attacker_eps16_tanks_and_temples/Courthouse/ --data_output_path dataset/Tanks_and_Temples_eps16/Courthouse/ &
python attacker/poison_splat_bounded.py --gpu 6 --adv_epsilon 16 --adv_iters 36000\
    --data_path dataset/Tanks_and_Temples/Courtroom/ --decoy_log_path log/01_main_exp/attacker_eps16_tanks_and_temples/Courtroom/ --data_output_path dataset/Tanks_and_Temples_eps16/Courtroom/ &
python attacker/poison_splat_bounded.py --gpu 7 --adv_epsilon 16 --adv_iters 36000\
    --data_path dataset/Tanks_and_Temples/Family/ --decoy_log_path log/01_main_exp/attacker_eps16_tanks_and_temples/Family/ --data_output_path dataset/Tanks_and_Temples_eps16/Family/
wait
python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/Tanks_and_Temples_eps16/Auditorium/ -m log/01_main_exp/victim_gs_tanks_and_temples_eps16/Auditorium/ &
python victim/gaussian-splatting/benchmark.py --gpu 1\
    -s dataset/Tanks_and_Temples_eps16/Ballroom/ -m log/01_main_exp/victim_gs_tanks_and_temples_eps16/Ballroom/ &
python victim/gaussian-splatting/benchmark.py --gpu 2\
    -s dataset/Tanks_and_Temples_eps16/Barn/ -m log/01_main_exp/victim_gs_tanks_and_temples_eps16/Barn/ &
python victim/gaussian-splatting/benchmark.py --gpu 3\
    -s dataset/Tanks_and_Temples_eps16/Caterpillar/ -m log/01_main_exp/victim_gs_tanks_and_temples_eps16/Caterpillar/ &
python victim/gaussian-splatting/benchmark.py --gpu 4\
    -s dataset/Tanks_and_Temples_eps16/Church/ -m log/01_main_exp/victim_gs_tanks_and_temples_eps16/Church/ &
python victim/gaussian-splatting/benchmark.py --gpu 5\
    -s dataset/Tanks_and_Temples_eps16/Courthouse/ -m log/01_main_exp/victim_gs_tanks_and_temples_eps16/Courthouse/ &
python victim/gaussian-splatting/benchmark.py --gpu 6\
    -s dataset/Tanks_and_Temples_eps16/Courtroom/ -m log/01_main_exp/victim_gs_tanks_and_temples_eps16/Courtroom/ &
python victim/gaussian-splatting/benchmark.py --gpu 7\
    -s dataset/Tanks_and_Temples_eps16/Family/ -m log/01_main_exp/victim_gs_tanks_and_temples_eps16/Family/
wait
