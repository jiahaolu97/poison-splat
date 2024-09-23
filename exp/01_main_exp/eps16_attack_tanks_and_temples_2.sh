python attacker/poison_splat_bounded.py --gpu 0 --adv_epsilon 16 --adv_iters 36000\
    --data_path dataset/Tanks_and_Temples/Francis/ --decoy_log_path log/01_main_exp/attacker_eps16_tanks_and_temples/Francis/ --data_output_path dataset/Tanks_and_Temples_eps16/Francis/ &
python attacker/poison_splat_bounded.py --gpu 1 --adv_epsilon 16 --adv_iters 36000\
    --data_path dataset/Tanks_and_Temples/Horse/ --decoy_log_path log/01_main_exp/attacker_eps16_tanks_and_temples/Horse/ --data_output_path dataset/Tanks_and_Temples_eps16/Horse/ &
python attacker/poison_splat_bounded.py --gpu 2 --adv_epsilon 16 --adv_iters 36000\
    --data_path dataset/Tanks_and_Temples/Ignatius/ --decoy_log_path log/01_main_exp/attacker_eps16_tanks_and_temples/Ignatius/ --data_output_path dataset/Tanks_and_Temples_eps16/Ignatius/ &
python attacker/poison_splat_bounded.py --gpu 3 --adv_epsilon 16 --adv_iters 36000\
    --data_path dataset/Tanks_and_Temples/Lighthouse/ --decoy_log_path log/01_main_exp/attacker_eps16_tanks_and_temples/Lighthouse/ --data_output_path dataset/Tanks_and_Temples_eps16/Lighthouse/ &
python attacker/poison_splat_bounded.py --gpu 4 --adv_epsilon 16 --adv_iters 36000\
    --data_path dataset/Tanks_and_Temples/M60/ --decoy_log_path log/01_main_exp/attacker_eps16_tanks_and_temples/M60/ --data_output_path dataset/Tanks_and_Temples_eps16/M60/ &
python attacker/poison_splat_bounded.py --gpu 5 --adv_epsilon 16 --adv_iters 36000\
    --data_path dataset/Tanks_and_Temples/Meetingroom/ --decoy_log_path log/01_main_exp/attacker_eps16_tanks_and_temples/Meetingroom/ --data_output_path dataset/Tanks_and_Temples_eps16/Meetingroom/ &
python attacker/poison_splat_bounded.py --gpu 6 --adv_epsilon 16 --adv_iters 36000\
    --data_path dataset/Tanks_and_Temples/Museum/ --decoy_log_path log/01_main_exp/attacker_eps16_tanks_and_temples/Museum/ --data_output_path dataset/Tanks_and_Temples_eps16/Museum/ &
python attacker/poison_splat_bounded.py --gpu 7 --adv_epsilon 16 --adv_iters 36000\
    --data_path dataset/Tanks_and_Temples/Palace/ --decoy_log_path log/01_main_exp/attacker_eps16_tanks_and_temples/Palace/ --data_output_path dataset/Tanks_and_Temples_eps16/Palace/ 
wait
python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/Tanks_and_Temples_eps16/Francis/ -m log/01_main_exp/victim_gs_tanks_and_temples_eps16/Francis/ &
python victim/gaussian-splatting/benchmark.py --gpu 1\
    -s dataset/Tanks_and_Temples_eps16/Horse/ -m log/01_main_exp/victim_gs_tanks_and_temples_eps16/Horse/ &
python victim/gaussian-splatting/benchmark.py --gpu 2\
    -s dataset/Tanks_and_Temples_eps16/Ignatius/ -m log/01_main_exp/victim_gs_tanks_and_temples_eps16/Ignatius/ &
python victim/gaussian-splatting/benchmark.py --gpu 3\
    -s dataset/Tanks_and_Temples_eps16/Lighthouse/ -m log/01_main_exp/victim_gs_tanks_and_temples_eps16/Lighthouse/ &
python victim/gaussian-splatting/benchmark.py --gpu 4\
    -s dataset/Tanks_and_Temples_eps16/M60/ -m log/01_main_exp/victim_gs_tanks_and_temples_eps16/M60/ &
python victim/gaussian-splatting/benchmark.py --gpu 5\
    -s dataset/Tanks_and_Temples_eps16/Meetingroom/ -m log/01_main_exp/victim_gs_tanks_and_temples_eps16/Meetingroom/ &
python victim/gaussian-splatting/benchmark.py --gpu 6\
    -s dataset/Tanks_and_Temples_eps16/Museum/ -m log/01_main_exp/victim_gs_tanks_and_temples_eps16/Museum/ &
python victim/gaussian-splatting/benchmark.py --gpu 7\
    -s dataset/Tanks_and_Temples_eps16/Palace/ -m log/01_main_exp/victim_gs_tanks_and_temples_eps16/Palace/ 
wait
