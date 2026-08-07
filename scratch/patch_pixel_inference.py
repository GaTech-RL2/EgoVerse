"""Idempotent patch: add a closed-loop controller for the PIXEL obs+action
policy to egomimic/algo/dfot/algo.py.

  * adds an ``inference_mode == "pixel_policy"`` dispatch branch
  * adds ``DFoT._inference_step_pixel_policy`` — receding-horizon rollout that
    pins the last n_context OBSERVED frames (real RGB + executed-action planes)
    clean, denoises the next k frames' [RGB + action] jointly, and reads the
    committed frame's action planes via global-avg-pool -> action.
"""
import re

F = "egomimic/algo/dfot/algo.py"
src = open(F).read()

if "pixel_policy" in src:
    print("already patched")
    raise SystemExit(0)

# 1) dispatch branch (insert before the unknown-mode raise)
RAISE = '        raise ValueError(f"unknown inference_mode {self.inference_mode!r}")'
BRANCH = (
    '        if self.inference_mode == "pixel_policy":\n'
    '            return self._inference_step_pixel_policy(obs_zarr, t, emb_id)\n'
    + RAISE
)
assert RAISE in src, "dispatch anchor not found"
src = src.replace(RAISE, BRANCH, 1)

# 2) the method (insert before _inference_step_ar)
METHOD = '''    @torch.no_grad()
    def _inference_step_pixel_policy(self, obs_zarr, t, emb_id):
        """Closed-loop controller for the PIXEL obs+action policy. The action
        rides as broadcast channels inside the diffused frame. Pin the last
        n_context OBSERVED frames (real RGB + executed-action planes) clean,
        denoise the next k frames' [RGB + action] jointly, read the committed
        frame's action planes by global-avg-pool -> action. Receding horizon."""
        from egomimic.algo.dfot.sampling import vanilla_schedule
        import numpy as _np

        outer = self.outer_stage
        diff = self.diffusion
        device = next(self.backbone.parameters()).device
        ac_key = self.ac_keys[get_embodiment(emb_id).lower()]
        n_ctx = max(1, int(getattr(self, "sp_n_context", 1)))
        k = max(1, int(getattr(self, "sp_commit", 1)))
        n_steps = int(self.sampler_n_steps)
        Ci = int(outer._image_channels)
        Ca = int(outer._action_channels)
        A = int(outer.action_dim)
        H = W = int(outer._image_size)
        C = Ci + Ca

        if t == 0 or not hasattr(self, "_pp_rgb"):
            self._pp_rgb = []     # observed RGB, each (Ci,H,W) in [0,1]
            self._pp_act = []     # executed NORMALIZED actions, each (A,)
            self._pp_queue = []   # pending committed unnorm actions

        img = obs_zarr[outer.image_key].float().to(device)
        if img.max() > 1.5:
            img = img / 255.0
        if img.dim() == 4:
            img = img[0]
        self._pp_rgb.append(img)

        if self._pp_queue:
            return self._pp_queue.pop(0)

        def act_plane(a):
            return a[:Ca].reshape(Ca, 1, 1).expand(Ca, H, W)

        n_done = len(self._pp_act)
        ctx_idx = [max(0, n_done - n_ctx + i) for i in range(n_ctx)]
        T = n_ctx + k

        ctx_frames = []
        for i in ctx_idx:
            rgb = self._pp_rgb[min(i, len(self._pp_rgb) - 1)]
            a = self._pp_act[i] if i < len(self._pp_act) else torch.zeros(A, device=device)
            ctx_frames.append(torch.cat([rgb, act_plane(a)], dim=0))
        ctx_stack = torch.stack(ctx_frames, dim=0).unsqueeze(0)   # (1,n_ctx,C,H,W)

        dts = int(diff.timesteps) if isinstance(diff, DiscreteDiffusion) else None
        clean = -1 if dts is not None else 0.0
        sched = vanilla_schedule(n_steps, T, discrete_timesteps=dts).to(device).clone()
        sched[:, :n_ctx] = clean

        x = torch.randn(1, T, C, H, W, device=device)
        x[:, :n_ctx] = ctx_stack
        for s in range(sched.shape[0] - 1):
            klev = sched[s].clamp_min(0).long().unsqueeze(0)
            v = self.backbone(x, klev, external_cond=None)
            x = self._struct_ddim_step(diff, x, v, sched[s], sched[s + 1])
            x[:, :n_ctx] = ctx_stack

        pred_planes = x[0, n_ctx:n_ctx + k, Ci:Ci + Ca]          # (k,Ca,H,W)
        pred_norm = pred_planes.mean(dim=(2, 3))[:, :A]          # (k,A) global-avg-pool
        for j in range(k):
            self._pp_act.append(pred_norm[j].detach())
        unnorm = self.norm_stats.unnormalize({ac_key: pred_norm}, emb_id)[ac_key]
        unnorm_np = unnorm.detach().cpu().numpy()
        for row in unnorm_np[1:]:
            self._pp_queue.append(row.reshape(-1).astype(_np.float32))
        return unnorm_np[0].reshape(-1).astype(_np.float32)

'''
ANCHOR = "    def _inference_step_ar("
assert ANCHOR in src, "method anchor not found"
src = src.replace(ANCHOR, METHOD + ANCHOR, 1)

open(F, "w").write(src)
print("patched ok")
