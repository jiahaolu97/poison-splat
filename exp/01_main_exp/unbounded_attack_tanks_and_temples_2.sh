python attacker/poison_splat_unbounded.py --gpu 0 --adv_iters 6000 --adv_decoy_update_interval 10\
    --data_path dataset/Tanks_and_Temples/Francis/ --decoy_log_path log/01_main_exp/attacker_unbounded_tanks_and_temples/Francis/ --data_output_path dataset/Tanks_and_Temples_unbounded/Francis/ &
python attacker/poison_splat_unbounded.py --gpu 1 --adv_iters 6000 --adv_decoy_update_interval 10\
    --data_path dataset/Tanks_and_Temples/Horse/ --decoy_log_path log/01_main_exp/attacker_unbounded_tanks_and_temples/Horse/ --data_output_path dataset/Tanks_and_Temples_unbounded/Horse/ &
python attacker/poison_splat_unbounded.py --gpu 2 --adv_iters 6000 --adv_decoy_update_interval 10\
    --data_path dataset/Tanks_and_Temples/Ignatius/ --decoy_log_path log/01_main_exp/attacker_unbounded_tanks_and_temples/Ignatius/ --data_output_path dataset/Tanks_and_Temples_unbounded/Ignatius/ &
python attacker/poison_splat_unbounded.py --gpu 3 --adv_iters 6000 --adv_decoy_update_interval 10\
    --data_path dataset/Tanks_and_Temples/Lighthouse/ --decoy_log_path log/01_main_exp/attacker_unbounded_tanks_and_temples/Lighthouse/ --data_output_path dataset/Tanks_and_Temples_unbounded/Lighthouse/ &
python attacker/poison_splat_unbounded.py --gpu 4 --adv_iters 6000 --adv_decoy_update_interval 10\
    --data_path dataset/Tanks_and_Temples/M60/ --decoy_log_path log/01_main_exp/attacker_unbounded_tanks_and_temples/M60/ --data_output_path dataset/Tanks_and_Temples_unbounded/M60/ &
python attacker/poison_splat_unbounded.py --gpu 5 --adv_iters 6000 --adv_decoy_update_interval 10\
    --data_path dataset/Tanks_and_Temples/Meetingroom/ --decoy_log_path log/01_main_exp/attacker_unbounded_tanks_and_temples/Meetingroom/ --data_output_path dataset/Tanks_and_Temples_unbounded/Meetingroom/ &
python attacker/poison_splat_unbounded.py --gpu 6 --adv_iters 6000 --adv_decoy_update_interval 10\
    --data_path dataset/Tanks_and_Temples/Museum/ --decoy_log_path log/01_main_exp/attacker_unbounded_tanks_and_temples/Museum/ --data_output_path dataset/Tanks_and_Temples_unbounded/Museum/ &
python attacker/poison_splat_unbounded.py --gpu 7 --adv_iters 6000 --adv_decoy_update_interval 10\
    --data_path dataset/Tanks_and_Temples/Palace/ --decoy_log_path log/01_main_exp/attacker_unbounded_tanks_and_temples/Palace/ --data_output_path dataset/Tanks_and_Temples_unbounded/Palace/ 
wait
python victim/gaussian-splatting/benchmark.py --gpu 0\
    -s dataset/Tanks_and_Temples_unbounded/Francis/ -m log/01_main_exp/victim_gs_tanks_and_temples_unbounded/Francis/ &
python victim/gaussian-splatting/benchmark.py --gpu 1\
    -s dataset/Tanks_and_Temples_unbounded/Horse/ -m log/01_main_exp/victim_gs_tanks_and_temples_unbounded/Horse/ &
python victim/gaussian-splatting/benchmark.py --gpu 2\
    -s dataset/Tanks_and_Temples_unbounded/Ignatius/ -m log/01_main_exp/victim_gs_tanks_and_temples_unbounded/Ignatius/ &
python victim/gaussian-splatting/benchmark.py --gpu 3\
    -s dataset/Tanks_and_Temples_unbounded/Lighthouse/ -m log/01_main_exp/victim_gs_tanks_and_temples_unbounded/Lighthouse/ &
python victim/gaussian-splatting/benchmark.py --gpu 4\
    -s dataset/Tanks_and_Temples_unbounded/M60/ -m log/01_main_exp/victim_gs_tanks_and_temples_unbounded/M60/ &
python victim/gaussian-splatting/benchmark.py --gpu 5\
    -s dataset/Tanks_and_Temples_unbounded/Meetingroom/ -m log/01_main_exp/victim_gs_tanks_and_temples_unbounded/Meetingroom/ &
python victim/gaussian-splatting/benchmark.py --gpu 6\
    -s dataset/Tanks_and_Temples_unbounded/Museum/ -m log/01_main_exp/victim_gs_tanks_and_temples_unbounded/Museum/ &
python victim/gaussian-splatting/benchmark.py --gpu 7\
    -s dataset/Tanks_and_Temples_unbounded/Palace/ -m log/01_main_exp/victim_gs_tanks_and_temples_unbounded/Palace/ 
wait