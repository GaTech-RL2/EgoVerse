import time, inspect, numpy as np
from Tsimulation.pushshapes import PushShapesEnv
env = PushShapesEnv(object_shape="T", pusher_shape="circle", obstacle_level=0, image_size=96)
o = env.reset(seed=0)
print("step sig   :", inspect.signature(env.step))
print("set_state  :", inspect.signature(env.set_state))
def act(): return np.array([np.random.uniform(20,190), np.random.uniform(20,190)], dtype=np.float32)
# warmup + discover API
for _ in range(10):
    try: env.step(act())
    except Exception as e: print("step ERR:", repr(e)); raise
N=300
t0=time.time()
for _ in range(N): env.step(act())
phys=(time.time()-t0)/N*1000
t0=time.time()
for _ in range(N):
    env.step(act()); env._get_obs()
full=(time.time()-t0)/N*1000
print(f"ms_per_step_phys_only = {phys:.3f}")
print(f"ms_per_step_with_render = {full:.3f}")
try:
    env.set_state(agent_pos=(100,100), object_pose=(150,150,0.5), goal_pose=(180,180,0.0))
    print("set_state(arbitrary) = OK")
except Exception as e:
    print("set_state ERR:", repr(e))
# Extrapolate closed-loop training cost (sim only, single env, sequential within window).
print(f"EXTRAP: K=32 window (1 env) = {full*32:.0f} ms of sim; with B parallel windows over C cores, ~{full*32:.0f}*ceil(B/C) ms/window-batch")
