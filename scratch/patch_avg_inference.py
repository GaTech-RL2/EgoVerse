"""Average sp_n_samples diffusion samples per action in the pixel_decoupled
closed-loop controller (variance reduction). Only worth deploying if the
variance-vs-bias test shows averaging sharply lowers the clean obs->action MSE."""
p = "egomimic/algo/dfot/algo.py"
s = open(p).read()
old = '''        x_rgb = rgb_ctx
        x_act = torch.randn(1, T, A, device=device)
        for st in range(act_sched.shape[0] - 1):
            _, v_act = self.backbone(
                x_rgb, obs_levels, external_cond=None, action=x_act,
                action_noise_levels=act_sched[st].unsqueeze(0),
            )
            x_act = self._struct_ddim_step(diff, x_act, v_act, act_sched[st], act_sched[st + 1])

        pred_norm = x_act[0, -1]'''
new = '''        x_rgb = rgb_ctx
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
assert old in s, "pixel_decoupled sampling block not found"
if "n_samp = max(1, int(getattr(self, \"sp_n_samples\", 1)))" not in s:
    s = s.replace(old, new)
    open(p, "w").write(s)
    print("PATCHED averaging into pixel_decoupled inference")
else:
    print("already patched")
