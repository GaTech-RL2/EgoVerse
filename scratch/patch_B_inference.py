F = "egomimic/algo/dfot/algo.py"
s = open(F).read()
if "pixel_regress" in s:
    print("already"); raise SystemExit(0)

RAISE = '        raise ValueError(f"unknown inference_mode {self.inference_mode!r}")'
BRANCH = ('        if self.inference_mode == "pixel_regress":\n'
          '            return self._inference_step_pixel_regress(obs_zarr, t, emb_id)\n' + RAISE)
assert RAISE in s, "dispatch anchor missing"
s = s.replace(RAISE, BRANCH, 1)

old_wl = '{"ar", "chunk", "spatial_rh", "spatial_decoupled", "pixel_policy"}'
new_wl = '{"ar", "chunk", "spatial_rh", "spatial_decoupled", "pixel_policy", "pixel_regress"}'
assert old_wl in s, "whitelist anchor missing"
s = s.replace(old_wl, new_wl, 1)

METHOD = '''    @torch.no_grad()
    def _inference_step_pixel_regress(self, obs_zarr, t, emb_id):
        """Closed-loop controller for Design B (regression). Pin the last
        n_context observed RGB frames clean, denoise the next k RGB frames,
        then read the action off each predicted frame via the outer stage's
        conv ``action_head``. Receding horizon."""
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
        H = W = int(outer._image_size)

        if t == 0 or not hasattr(self, "_pr_rgb"):
            self._pr_rgb = []
            self._pr_queue = []

        img = obs_zarr[outer.image_key].float().to(device)
        if img.max() > 1.5:
            img = img / 255.0
        if img.dim() == 4:
            img = img[0]
        self._pr_rgb.append(img)

        if self._pr_queue:
            return self._pr_queue.pop(0)

        T = n_ctx + k
        nrgb = len(self._pr_rgb)
        ctx_stack = torch.stack(
            [self._pr_rgb[max(0, nrgb - n_ctx + j)] for j in range(n_ctx)], dim=0
        ).unsqueeze(0)

        dts = int(diff.timesteps) if isinstance(diff, DiscreteDiffusion) else None
        clean = -1 if dts is not None else 0.0
        sched = vanilla_schedule(n_steps, T, discrete_timesteps=dts).to(device).clone()
        sched[:, :n_ctx] = clean

        x = torch.randn(1, T, Ci, H, W, device=device)
        x[:, :n_ctx] = ctx_stack
        for sidx in range(sched.shape[0] - 1):
            klev = sched[sidx].clamp_min(0).long().unsqueeze(0)
            v = self.backbone(x, klev, external_cond=None)
            x = self._struct_ddim_step(diff, x, v, sched[sidx], sched[sidx + 1])
            x[:, :n_ctx] = ctx_stack

        pred_frames = x[0, n_ctx:n_ctx + k]               # (k, Ci, H, W) predicted clean frames
        pred_norm = outer.action_head(pred_frames)        # (k, A)
        unnorm = self.norm_stats.unnormalize({ac_key: pred_norm}, emb_id)[ac_key]
        unnorm_np = unnorm.detach().cpu().numpy()
        for row in unnorm_np[1:]:
            self._pr_queue.append(row.reshape(-1).astype(_np.float32))
        return unnorm_np[0].reshape(-1).astype(_np.float32)

'''
assert "    def _inference_step_ar(" in s, "method anchor missing"
s = s.replace("    def _inference_step_ar(", METHOD + "    def _inference_step_ar(", 1)
open(F, "w").write(s)
print("patched")
