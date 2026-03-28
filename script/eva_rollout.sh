aria auth pair

export LD_LIBRARY_PATH=/root/.local/share/mamba/envs/arx-py310/lib:$LD_LIBRARY_PATH
cd /home/robot/robot_ws/egomimic/robot

python3 rollout.py --cartesian --policy-path /home/robot/robot_ws/egomimic/robot/models/cup_saucer/hpt_eva_2026-03-26_21-48-57/epoch_epoch=1399.ckpt --arms both --frequency 30 --query_frequency 30 --resampled-action-len 45


python3 collect_demo.py --arms both --auto-episode-start 46 --demo-dir ./demos/white_cup_white_saucer
python3 collect_demo.py --arms both --auto-episode-start 49 --demo-dir ./demos/white_cup_yellow_saucer
python3 collect_demo.py --arms both --auto-episode-start 47 --demo-dir ./demos/white_cup_green_square_saucer

python3 collect_demo.py --arms both --auto-episode-start 50 --demo-dir ./demos/white_ceramic_cup_white_saucer
python3 collect_demo.py --arms both --auto-episode-start 51 --demo-dir ./demos/white_ceramic_cup_yellow_saucer
python3 collect_demo.py --arms both --auto-episode-start 42 --demo-dir ./demos/white_ceramic_cup_green_square_saucer

python3 collect_demo.py --arms both --auto-episode-start 46 --demo-dir ./demos/pink_cup_white_saucer
python3 collect_demo.py --arms both --auto-episode-start 46 --demo-dir ./demos/pink_cup_yellow_saucer
python3 collect_demo.py --arms both --auto-episode-start 47 --demo-dir ./demos/pink_cup_green_square_saucer