"""Add the pixel_decoupled closed-loop controller + dispatch to DFoT.
Pins the current RGB frame (and last n_context) CLEAN, denoises the action
token on its own schedule, reads the action at the CURRENT frame -> reactive,
no 1-frame offset, action never a clean input (cannot copy)."""
p = "egomimic/algo/dfot/algo.py"
s = open(p).read()

old_d = '''        if self.inference_mode == "pixel_regress":
            return self._inference_step_pixel_regress(obs_zarr, t, emb_id)'''
new_d = old_d + '''
        if self.inference_mode == "pixel_decoupled":
            return self._inference_step_pixel_decoupled(obs_zarr, t, emb_id)'''
assert old_d in s, "dispatch anchor not found"
if "return self._inference_step_pixel_decoupled" not in s:
    s = s.replace(old_d, new_d)

marker = "    def _inference_step_spatial_rh("
method = '''    def _inference_step_pixel_decoupled(self, obs_zarr, t, emb_id):
        """Closed-loop controller for the PIXEL DECOUPLED-action policy. The
        action token carries its own noise level, so we pin the real current RGB
        frame (and last n_context observed) CLEAN while denoising the action
        token, and read the action at the CURRENT frame -> reactive, NO offset,
        action never a clean input (cannot copy)."""
        from egomimic.algo.dfot.sampling import vanilla_schedule

        outer = self.outer_stage
        diff = self.diffusion
        device = next(self.backbone.parameters()).device
        ac_key = self.ac_keys[get_embodiment(emb_id).lower()]
        n_ctx = max(1, int(getattr(self, "sp_n_context", 1)))
        n_steps = int(self.sampler_n_steps)
        A = int(outer.action_dim)

        if t == 0 or not hasattr(self, "_pd_rgb"):
            self._pd_rgb = []

        img = obs_zarr[outer.image_key].float().to(device)
        if img.max() > 1.5:
            img = img / 255.0
        if img.dim() == 4:
            img = img[0]
        self._pd_rgb.append(img)

        L = len(self._pd_rgb)
        idx = [max(0, L - n_ctx + i) for i in range(n_ctx)]
        T = n_ctx
        rgb_ctx = torch.stack([self._pd_rgb[i] for i in idx], dim=0).unsqueeze(0)

        dts = int(diff.timesteps) if isinstance(diff, DiscreteDiffusion) else None
        clean = -1 if dts is not None else 0.0
        obs_levels = torch.full(
            (1, T), clean, device=device,
            dtype=torch.long if dts is not None else torch.float32,
        )
        act_sched = vanilla_schedule(n_steps, T, discrete_timesteps=dts).to(device)

        x_rgb = rgb_ctx
        x_act = torch.randn(1, T, A, device=device)
        for st in range(act_sched.shape[0] - 1):
            _, v_act = self.backbone(
                x_rgb, obs_levels, external_cond=None, action=x_act,
                action_noise_levels=act_sched[st].unsqueeze(0),
            )
            x_act = self._struct_ddim_step(diff, x_act, v_act, act_sched[st], act_sched[st + 1])

        pred_norm = x_act[0, -1]
        unnorm = self.norm_stats.unnormalize(
            {ac_key: pred_norm.unsqueeze(0)}, emb_id
        )[ac_key]
        return unnorm.squeeze(0).detach().cpu().numpy().reshape(-1).astype(np.float32)

'''
assert marker in s, "method marker not found"
if "def _inference_step_pixel_decoupled" not in s:
    s = s.replace(marker, method + marker, 1)

open(p, "w").write(s)
print("PATCHED decoupled inference OK")
