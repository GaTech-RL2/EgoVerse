"""BC-pretrain a SAC actor on demonstrations, then continue with RL.

From-scratch RL does not solve this task: across three reward variants, two
algorithms, two horizons and two effectors, every run plateaued near 0.03 mean
coverage with SR exactly 0 and no episode ever terminating early -- the terminal
bonus was never once observed. The task needs approach, contact, then a push
that aligns ANGLE to reach 0.95 IoU, and undirected exploration essentially
never produces that sequence, so the agent only ever optimises shaping.

BC puts the policy inside the basin first. RL then only has to improve on a
working policy rather than find one.
"""
from __future__ import annotations
import os, sys, glob, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import torch
torch.set_num_threads(1)
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from Tsimulation.sim_v2.rl.env import PushShapesRLEnv

import zarr


def load_demo_transitions(emb, gap, root, n_eps, horizon):
    """Replay recorded episodes through the RL env so obs/reward match exactly
    what the agent will see, and record normalised-absolute actions."""
    paths = sorted(glob.glob(f"{root}/{gap}/{emb}/T/episode_*"))[:n_eps]
    env = PushShapesRLEnv(pusher_shape=emb, control_gap=gap,
                          max_steps=horizon, action_mode="residual")
    O, A, R, O2, D = [], [], [], [], []
    ok = 0
    for p in paths:
        g = zarr.open(p, mode="r")
        ei = json.loads(g.attrs["episode_init"])
        acts = np.asarray(g["actions"][:])
        env.reset(seed=ei.get("reset_seed", 0))
        env._env.set_state(agent_pos=ei["agent_pos"], agent_angle=ei.get("agent_angle"),
                           object_pose=ei["object_pose"], goal_pose=ei.get("goal_pose"))
        env._prev_phi = env._phi()
        o = env._obs()
        for a in acts[:horizon]:
            # Encode with the SAME mode the policy acts in, and step with the
            # ENCODED action. Collecting targets through a native-mode env took
            # to_norm's absolute branch while evaluation decoded them as
            # residuals: the BC target and the decoder disagreed, and BC scored
            # exactly 0.0000 coverage while its MSE fell normally.
            a_norm = env.to_norm(a)
            o2, r, term, trunc, info = env.step(a_norm)
            O.append(o); A.append(a_norm); R.append(r); O2.append(o2); D.append(term)
            o = o2
            if term or trunc:
                ok += int(term)
                break
    return (np.array(O, np.float32), np.array(A, np.float32), np.array(R, np.float32),
            np.array(O2, np.float32), np.array(D, bool), ok, len(paths))


def bc_fit(model, O, A, epochs, batch=512, lr=1e-3):
    actor = model.policy.actor
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    Ot = torch.as_tensor(O); At = torch.as_tensor(A)
    n = len(Ot)
    for ep in range(epochs):
        perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            mean_actions, _log_std, _kw = actor.get_action_dist_params(Ot[idx])
            loss = torch.nn.functional.mse_loss(torch.tanh(mean_actions), At[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    bc epoch {ep:>3} mse={tot/n:.5f}", flush=True)
    return model


def evaluate(model, emb, gap, horizon, n=15):
    e = PushShapesRLEnv(pusher_shape=emb, control_gap=gap, max_steps=horizon,
                        action_mode="residual")
    cov, succ, lens = [], 0, []
    for k in range(n):
        o, _ = e.reset(seed=1000 + k); best = 0.0; steps = 0
        for _ in range(horizon):
            a, _ = model.predict(o, deterministic=True)
            o, r, term, trunc, info = e.step(a)
            best = max(best, info["coverage"]); steps += 1
            if term: succ += 1; break
            if trunc: break
        cov.append(best); lens.append(steps)
    return float(np.mean(cov)), float(np.max(cov)), succ / n, float(np.mean(lens))


if __name__ == "__main__":
    EMB, GAP = sys.argv[1], sys.argv[2]
    RL_STEPS, HORIZON, NENV = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    N_EPS, BC_EPOCHS = int(sys.argv[6]), int(sys.argv[7])
    ROOT = os.environ.get("DEMO_ROOT", "/workspace/pushshapes")

    t0 = time.time()
    O, A, R, O2, D, ok, npaths = load_demo_transitions(EMB, GAP, ROOT, N_EPS, HORIZON)
    print(f"  demos: {npaths} episodes -> {len(O)} transitions, {ok} reached success "
          f"({time.time()-t0:.0f}s)", flush=True)

    venv = SubprocVecEnv([(lambda: PushShapesRLEnv(pusher_shape=EMB, control_gap=GAP,
                                                   max_steps=HORIZON,
                                                   action_mode="residual"))
                          for _ in range(NENV)])
    model = SAC("MlpPolicy", venv, verbose=0, batch_size=512, learning_starts=0,
                train_freq=8, gradient_steps=8, buffer_size=1_000_000,
                ent_coef=0.01, learning_rate=3e-4, device="cpu")

    print("  --- BC pretrain ---", flush=True)
    bc_fit(model, O, A, BC_EPOCHS)
    mc, xc, sr, ln = evaluate(model, EMB, GAP, HORIZON)
    print(f"  BC-ONLY  mean_cov={mc:.4f} max_cov={xc:.4f} SR={sr:.2f} ep_len={ln:.0f}",
          flush=True)

    # Seed the buffer so the critic sees successful trajectories from step 0.
    ne = model.replay_buffer.n_envs
    for i in range(len(O)):
        model.replay_buffer.add(
            np.tile(O[i], (ne, 1)), np.tile(O2[i], (ne, 1)), np.tile(A[i], (ne, 1)),
            np.array([R[i]] * ne), np.array([D[i]] * ne), [{} for _ in range(ne)])
    print(f"  seeded replay buffer with {len(O)} demo transitions", flush=True)

    print("  --- RL finetune ---", flush=True)
    CH = 10
    for c in range(CH):
        model.learn(total_timesteps=RL_STEPS // CH, reset_num_timesteps=False,
                    progress_bar=False)
        mc, xc, sr, ln = evaluate(model, EMB, GAP, HORIZON)
        print(f"  bcrl steps={(c+1)*RL_STEPS//CH:>8} mean_cov={mc:.4f} max_cov={xc:.4f} "
              f"SR={sr:.2f} ep_len={ln:.0f} ({time.time()-t0:.0f}s)", flush=True)
    out = os.environ.get("RL_OUT", "/tmp"); os.makedirs(out, exist_ok=True)
    model.save(f"{out}/bcrl_{EMB}_{GAP}_h{HORIZON}")
    print("DONE", flush=True)
