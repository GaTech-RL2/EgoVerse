"""Train one RL policy for a single (algorithm, effector, control gap).

Unlike the BC sweep this needs NO dataset: the policy generates its own
experience, so a node runs it with no R2 pull at all.
"""
import os, sys, time, warnings, numpy as np
warnings.filterwarnings("ignore")
os.environ.setdefault("SDL_VIDEODRIVER","dummy")
import torch
# One torch thread per process. The policy is a tiny MLP, but SB3 defaults to
# many intra-op threads; with several trainers x N SubprocVecEnv workers on one
# node they fight over cores. Observed: 4 jobs each pinning ~14 cores, load 110
# on 128 CPUs, and throughput of ~105 steps/s against 1100 steps/s for a single
# 8-env job on a 10-core laptop.
torch.set_num_threads(1)
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from Tsimulation.sim_v2.rl.env import PushShapesRLEnv

ALGO=sys.argv[1]; EMB=sys.argv[2]; GAP=sys.argv[3]
STEPS=int(sys.argv[4]); HORIZON=int(sys.argv[5]); NENV=int(sys.argv[6])

def mk(i):
    def _f(): return PushShapesRLEnv(pusher_shape=EMB, control_gap=GAP, max_steps=HORIZON)
    return _f

def evaluate(model, venv, n=15):
    e = PushShapesRLEnv(pusher_shape=EMB, control_gap=GAP, max_steps=HORIZON)
    cov=[]; succ=0; lens=[]
    for k in range(n):
        o,_=e.reset(seed=1000+k); best=0.0; steps=0
        for _ in range(HORIZON):
            oo = venv.normalize_obs(o) if venv is not None else o
            a,_=model.predict(oo, deterministic=True)
            o,r,term,trunc,info=e.step(a); best=max(best,info["coverage"]); steps+=1
            if term: succ+=1; break
            if trunc: break
        cov.append(best); lens.append(steps)
    return float(np.mean(cov)), float(np.max(cov)), succ/n, float(np.mean(lens))

if __name__ == "__main__":
    venv = SubprocVecEnv([mk(i) for i in range(NENV)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    if ALGO == "sac":
        model = SAC("MlpPolicy", venv, verbose=0, batch_size=512, learning_starts=5000,
                    train_freq=8, gradient_steps=8, buffer_size=500_000,
                    ent_coef=0.01, learning_rate=3e-4, device="cpu")
    else:
        model = PPO("MlpPolicy", venv, verbose=0, n_steps=512, batch_size=1024,
                    n_epochs=10, gae_lambda=0.95, gamma=0.99, ent_coef=0.005,
                    learning_rate=3e-4, clip_range=0.2, device="cpu")
    t0=time.time()
    CHUNKS = 12          # evaluate often: a 4-chunk cadence meant no output at
                         # all for 750k steps, indistinguishable from a hang
    for c in range(CHUNKS):
        model.learn(total_timesteps=STEPS//CHUNKS, reset_num_timesteps=False,
                    progress_bar=False)
        mc,xc,sr,ln = evaluate(model, venv)
        print(f"  {ALGO} h{HORIZON} steps={(c+1)*STEPS//CHUNKS:>8} mean_cov={mc:.4f} "
              f"max_cov={xc:.4f} SR={sr:.2f} ep_len={ln:.0f} ({time.time()-t0:.0f}s)", flush=True)
    out = os.environ.get("RL_OUT", "/tmp")
    os.makedirs(out, exist_ok=True)
    tag = f"{ALGO}_{EMB}_{GAP}_h{HORIZON}"
    model.save(f"{out}/rl_{tag}")
    venv.save(f"{out}/rl_{tag}_vecnorm.pkl")
    print("DONE", flush=True)
