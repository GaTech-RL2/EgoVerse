"""Diagnostic: does H-Net's AR inference_step actually use the observation?

Loads a trained H-Net ckpt and calls inference_step(t=0) with several very
different synthetic observations. If the returned action barely changes across
wildly different obs, the obs is NOT reaching the model (eval/plumbing bug). If
it changes substantially, the model genuinely (under)uses obs -> real failure.
"""
import argparse, sys
import numpy as np
import torch

sys.path.insert(0, "scripts")
from smoke_sim_eval import load_algo_from_ckpt
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


def make_obs(agent, obj, img_val, device):
    state5 = np.array([agent[0], agent[1], obj[0], obj[1], obj[2]], dtype=np.float32)
    img = np.full((3, 96, 96), img_val, dtype=np.float32)
    return {
        "state_agent_obj": torch.from_numpy(state5).unsqueeze(0).to(device),
        "front_img_1": torch.from_numpy(img).unsqueeze(0).to(device),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-path", required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algo, _ = load_algo_from_ckpt(args.ckpt, args.config_path)
    algo.nets = algo.nets.to(device)
    algo.device = device
    algo.nets.eval()
    emb_id = get_embodiment_id("pushshapes_sim")

    cases = [
        ("A agent[100,100] obj[200,200,0.0] img0.2", [100, 100], [200, 200, 0.0], 0.2),
        ("B agent[400,400] obj[50,300,1.5]  img0.8", [400, 400], [50, 300, 1.5], 0.8),
        ("C agent[250,80]  obj[400,420,-2.] img0.5", [250, 80], [400, 420, -2.0], 0.5),
        ("D agent[60,450]  obj[300,120,2.9] img0.1", [60, 450], [300, 120, 2.9], 0.1),
    ]
    acts = []
    for name, agent, obj, iv in cases:
        obs = make_obs(agent, obj, iv, device)
        a = np.asarray(algo.inference_step(obs, 0, emb_id)).reshape(-1)  # t=0 resets state
        acts.append(a)
        print(f"[t0] {name}  -> action={np.round(a,2)}", flush=True)

    acts = np.stack(acts, 0)
    spread = acts.max(0) - acts.min(0)
    print(f"\n[SPREAD across 4 very-different obs] per-dim max-min = {np.round(spread,3)}")
    print(f"[VERDICT] {'OBS IGNORED (spread<1px -> plumbing bug)' if spread.max() < 1.0 else 'obs DOES change output -> model uses obs'}")

    # Also: does action move over an AR rollout from a single obs, or freeze?
    obs = make_obs([200, 200], [300, 300, 0.0], 0.5, device)
    traj = [np.asarray(algo.inference_step(obs, t, emb_id)).reshape(-1) for t in range(8)]
    traj = np.stack(traj, 0)
    print(f"\n[AR traj from fixed obs, 8 steps] step-to-step move = {np.round(np.linalg.norm(np.diff(traj,axis=0),axis=1),2)}")


if __name__ == "__main__":
    main()
