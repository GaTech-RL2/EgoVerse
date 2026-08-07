"""Replace the SEQUENTIAL K-sample averaging in _inference_step_pixel_decoupled
with a BATCHED version (run all K samples as the batch dim -> one forward per
denoise step instead of K) -> ~K x faster, makes the averaged sim practical."""
p = "egomimic/algo/dfot/algo.py"
s = open(p).read()
old = '''        x_rgb = rgb_ctx
        n_samp = max(1, int(getattr(self, "sp_n_samples", 1)))
        _preds = []
        for _samp in range(n_samp):
            x_act = torch.randn(1, T, A, device=device)
            for st in range(act_sched.shape[0] - 1):
                _, v_act = self.backbone(
                    x_rgb, obs_levels, external_cond=None, action=x_act,
                    action_noise_levels=act_sched[st].unsqueeze(0),
                )
                x_act = self._struct_ddim_step(diff, x_act, v_act, act_sched[st], act_sched[st + 1])
            _preds.append(x_act[0, -1])
        pred_norm = torch.stack(_preds).mean(0)'''
new = '''        n_samp = max(1, int(getattr(self, "sp_n_samples", 1)))
        x_rgb = rgb_ctx.expand(n_samp, -1, -1, -1, -1)
        o_lev = obs_levels.expand(n_samp, -1)
        x_act = torch.randn(n_samp, T, A, device=device)
        for st in range(act_sched.shape[0] - 1):
            a_lev = act_sched[st].unsqueeze(0).expand(n_samp, -1)
            _, v_act = self.backbone(
                x_rgb, o_lev, external_cond=None, action=x_act,
                action_noise_levels=a_lev,
            )
            x_act = self._struct_ddim_step(diff, x_act, v_act, act_sched[st], act_sched[st + 1])
        pred_norm = x_act[:, -1].mean(0)'''
assert old in s, "sequential averaging block not found (already batched?)"
s = s.replace(old, new)
open(p, "w").write(s)
print("PATCHED batched averaging")
