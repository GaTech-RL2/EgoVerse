"""Multiprocessing env pool for the closed-loop windowed trainer (Path B).

A persistent pool of `n_envs` PushShapesEnv spread across `n_workers` CPU worker
processes (the GPU model stays in the main process). Each rollout step the main
process sends the W actions, the workers step their envs in parallel, and return
the W observations. This is the ~Nx speedup over the single-env loop — the sim
render (~10.8 ms/step) is the bottleneck, so parallelizing it across cores wins.

W (active windows) <= n_envs each batch (the trainer caps it via max_windows).
Uses the 'spawn' start method so the CUDA-using parent doesn't fork into workers.
"""
import os
import multiprocessing as mp


def _worker(conn, n_envs, image_size):
    os.environ.setdefault("MUJOCO_GL", "egl")
    from Tsimulation.pushshapes import PushShapesEnv
    envs = [
        PushShapesEnv(object_shape="T", pusher_shape="circle",
                      obstacle_level=0, image_size=image_size)
        for _ in range(n_envs)
    ]
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "close":
                break
            if cmd == "set":
                out = []
                for j, ap, op, gp in data:           # (local_idx, agent, obj, goal)
                    envs[j].reset(seed=0)
                    envs[j].set_state(agent_pos=ap, object_pose=op, goal_pose=gp)
                    out.append((j, envs[j]._get_obs()))
                conn.send(out)
            elif cmd == "step":
                out = []
                for j, a in data:                    # (local_idx, action_xy)
                    o, _, _, _, _ = envs[j].step(a)
                    out.append((j, o))
                conn.send(out)
    except (EOFError, KeyboardInterrupt):
        pass


class VecPushShapes:
    """Fixed pool of n_envs across n_workers; step/set_state the first W each call."""

    def __init__(self, n_envs: int, n_workers: int, image_size: int = 96):
        self.n_envs = int(n_envs)
        self.n_workers = max(1, min(int(n_workers), self.n_envs))
        # global env i -> (worker w, local index within w), round-robin.
        self._assign = []
        counts = [0] * self.n_workers
        for i in range(self.n_envs):
            w = i % self.n_workers
            self._assign.append((w, counts[w]))
            counts[w] += 1
        self._loc2glob = {wl: i for i, wl in enumerate(self._assign)}
        ctx = mp.get_context("spawn")
        self._conns, self._procs = [], []
        for w in range(self.n_workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, counts[w], image_size), daemon=True)
            p.start()
            self._conns.append(parent)
            self._procs.append(p)

    def _scatter(self, cmd, items):
        """items: list (len W) -> send per-worker, gather back in global order."""
        W = len(items)
        per = [[] for _ in range(self.n_workers)]
        for i in range(W):
            w, loc = self._assign[i]
            per[w].append((loc,) + tuple(items[i]) if isinstance(items[i], tuple) else (loc, items[i]))
        for w in range(self.n_workers):
            self._conns[w].send((cmd, per[w]))
        out = [None] * W
        for w in range(self.n_workers):
            for res in self._conns[w].recv():
                loc, o = res[0], res[1]
                out[self._loc2glob[(w, loc)]] = o
        return out

    def set_states(self, states):
        """states: list (len W<=n_envs) of (agent_pos, object_pose, goal_pose)."""
        return self._scatter("set", states)

    def step(self, actions):
        """actions: list (len W) of action_xy np arrays. Returns list of obs dicts."""
        return self._scatter("step", actions)

    def close(self):
        for c in self._conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self._procs:
            try:
                p.terminate()        # force-kill so the main process can exit cleanly
            except Exception:
                pass
        for p in self._procs:
            try:
                p.join(timeout=1)
            except Exception:
                pass
        self._procs, self._conns = [], []
