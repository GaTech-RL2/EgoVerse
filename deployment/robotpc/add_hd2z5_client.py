"""2026-11-07 egoverse-29: robot-client support for the hd2z5 policies (RBY1_fridge_open1106_hd2z5 = fridge OPEN v2, RBY1_cloth_basket1106_hd2z5;
user 11-06: "loosen the depth axis cropping ... upper bound much higher, like 5 meters ... let's not do like lateral cropping"). Run on the
LAPTOP (or on a copy of the client code), idempotent:
preset ABL=TGZ10HD2Z5 on port 8157 = TGZ10HD2 (the hd2 preset, add_hd2_client.py) with ONLY the global crop changed, as the training builder
(fridge_open_1106/build_hd2z5.sbatch: TOY_ZMAX=5.0 TOY_LAT=100):
  global 2048: Z-slab 0.25-5.0 m (was 2.5) AND a 1.0 m gaze ball, NO lateral crop (no lateral_max key -> None -> no sqrt(x^2+y^2) test),
               hand-dense (512 + 512 within 0.30 m of each TCP + 1024 FPS over the rest);
  locals 1024 each: 1.0 m ball around each eef, hand-dense local (unchanged);  grey: 0.30 m around either TCP (unchanged).
Run with the same VV_MODEL / gaze predictor as the fridge hd2 policies.
Edits, only if missing (asserts stop it if the client is not the TGZ10HD2-patched one):
  .../projects/rby1_teleop/utils/fastfs_policy_perception.py   + preset "TGZ10HD2Z5" (right after "TGZ10HD2")
  .../projects/rby1_teleop/run_rollout_pcd_policy.sh          + [TGZ10HD2Z5]=8157
usage: python3 add_hd2z5_client.py [SEW/projects/rby1_teleop]   (default: the laptop path; after add_hd2_client.py)"""
import sys
from pathlib import Path

SEW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/aloha/RB_Y1_workspace/SEW-Geometric-Teleop/projects/rby1_teleop')
perc, run = SEW / 'utils/fastfs_policy_perception.py', SEW / 'run_rollout_pcd_policy.sh'
s = perc.read_text()
if '"TGZ10HD2Z5"' in s:
    print('perception already has TGZ10HD2Z5')
else:
    a = ('    "TGZ10HD2": {"num_points_global": 2048, "num_points_local": 1024, "left_stream": True,\n'
         '                 "z_max": 2.5, "lateral_max": 1.0, "global_mode": "slab_gaze_ball", "gaze_radius": 1.0,\n'
         '                 "excision_radius": 0.30, "hand_dense": True, "ball_radius": 1.0,\n'
         '                 "excision_at_tcp": True, "hand_dense_local": True},\n')
    assert s.count(a) == 1, 'TGZ10HD2 preset not found as expected: run add_hd2_client.py first'
    s = s.replace(a, a + (
        '    # 2026-11-07 hd2z5 (EgoVerse RBY1_fridge_open1106_hd2z5 / RBY1_cloth_basket1106_hd2z5; builder fridge_open_1106/build_hd2z5.sbatch):\n'
        '    # TGZ10HD2 with the Z-slab up to 5.0 m and NO lateral crop (training TOY_LAT=100); gaze ball 1.0 m, hand-dense, grey at the TCPs.\n'
        '    "TGZ10HD2Z5": {"num_points_global": 2048, "num_points_local": 1024, "left_stream": True,\n'
        '                   "z_max": 5.0, "global_mode": "slab_gaze_ball", "gaze_radius": 1.0,\n'
        '                   "excision_radius": 0.30, "hand_dense": True, "ball_radius": 1.0,\n'
        '                   "excision_at_tcp": True, "hand_dense_local": True},\n'))
    perc.write_text(s); print('added preset TGZ10HD2Z5 to', perc)
r = run.read_text()
if '[TGZ10HD2Z5]=' in r:
    print('run script already has a TGZ10HD2Z5 port')
else:
    assert r.count('[TGZ10HD2]=8156') == 1, 'PORTS entry [TGZ10HD2]=8156 not found: run add_hd2_client.py first'
    assert '=8157' not in r, 'port 8157 already used in run_rollout_pcd_policy.sh: pick another port'
    run.write_text(r.replace('[TGZ10HD2]=8156', '[TGZ10HD2]=8156 [TGZ10HD2Z5]=8157')); print('added [TGZ10HD2Z5]=8157 to', run)
print('ADD_HD2Z5_CLIENT_OK')
