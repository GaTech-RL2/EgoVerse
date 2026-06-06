"""Replace _inference_step_pixel_decoupled with a CHUNKING version (sp_commit>1):
append (sp_commit-1) future frames whose obs AND action are denoised (model
predicts future obs), pin the current+past obs clean, read the action at the
current frame + the future frames, commit the whole chunk open-loop. sp_commit=1
is identical to the per-step controller. Keeps batched sp_n_samples averaging."""
p = "egomimic/algo/dfot/algo.py"
s = open(p).read()

start_anchor = "    @torch.no_grad()\n    def _inference_step_pixel_decoupled(self, obs_zarr, t, emb_id):"
end_anchor = "    def _inference_step_spatial_rh("
start = s.index(start_anchor)
end = s.index(end_anchor)

new_method = '''    @torch.no_grad()
    def _inference_step_pixel_decoupled(self, obs_zarr, t, emb_id):
        """Closed-loop controller for the PIXEL DECOUPLED-action policy, with
        optional CHUNKING (sp_commit>1). Pin the current RGB (+ last n_context)
        CLEAN; for a chunk, append (sp_commit-1) FUTURE frames whose obs AND
        action are denoised (model predicts future obs as a world-model), read
        the action at the current frame + the future frames, commit the whole
        chunk open-loop. Action never a clean input (no copy), no offset.
        sp_n_samples averages K diffusion samples (variance reduction)."""
        from egomimic.algo.dfot.sampling import vanilla_schedule

        outer = self.outer_stage
        diff = self.diffusion
        device = next(self.backbone.parameters()).device
        ac_key = self.ac_keys[get_embodiment(emb_id).lower()]
        n_ctx = max(1, int(getattr(self, "sp_n_context", 1)))
        k = max(1, int(getattr(self, "sp_commit", 1)))
        n_steps = int(self.sampler_n_steps)
        n_samp = max(1, int(getattr(self, "sp_n_samples", 1)))
        A = int(outer.action_dim)
        Ci = int(outer._image_channels)
        H = W = int(outer._image_size)

        if t == 0 or not hasattr(self, "_pd_rgb"):
            self._pd_rgb = []
            self._pd_queue = []

        img = obs_zarr[outer.image_key].float().to(device)
        if img.max() > 1.5:
            img = img / 255.0
        if img.dim() == 4:
            img = img[0]
        self._pd_rgb.append(img)

        if self._pd_queue:
            return self._pd_queue.pop(0)

        L = len(self._pd_rgb)
        idx = [max(0, L - n_ctx + i) for i in range(n_ctx)]
        rgb_ctx = torch.stack([self._pd_rgb[i] for i in idx], dim=0).unsqueeze(0)

        T = n_ctx + (k - 1)
        dts = int(diff.timesteps) if isinstance(diff, DiscreteDiffusion) else None
        sched = vanilla_schedule(n_steps, T, discrete_timesteps=dts).to(device)
        obs_sched = sched.clone()
        obs_sched[:, :n_ctx] = 0  # context obs CLEAN (re-pinned each step)

        ctx_b = rgb_ctx.expand(n_samp, -1, -1, -1, -1)
        x_obs = torch.randn(n_samp, T, Ci, H, W, device=device)
        x_obs[:, :n_ctx] = ctx_b
        x_act = torch.randn(n_samp, T, A, device=device)
        for st in range(sched.shape[0] - 1):
            o_lev = obs_sched[st].unsqueeze(0).expand(n_samp, -1)
            a_lev = sched[st].unsqueeze(0).expand(n_samp, -1)
            v_img, v_act = self.backbone(
                x_obs, o_lev, external_cond=None, action=x_act, action_noise_levels=a_lev,
            )
            x_obs = self._struct_ddim_step(diff, x_obs, v_img, obs_sched[st], obs_sched[st + 1])
            x_obs[:, :n_ctx] = ctx_b
            x_act = self._struct_ddim_step(diff, x_act, v_act, sched[st], sched[st + 1])

        chunk = x_act[:, n_ctx - 1:].mean(0)  # (k, A): current + (k-1) future actions
        unnorm = self.norm_stats.unnormalize({ac_key: chunk}, emb_id)[ac_key]
        unnorm_np = unnorm.detach().cpu().numpy()
        for row in unnorm_np[1:]:
            self._pd_queue.append(row.reshape(-1).astype(np.float32))
        return unnorm_np[0].reshape(-1).astype(np.float32)

'''
s = s[:start] + new_method + s[end:]
open(p, "w").write(s)
print("REPLACED _inference_step_pixel_decoupled with chunking version")
