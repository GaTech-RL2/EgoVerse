"""Fix the pinned-obs noise level in _inference_step_pixel_decoupled: it was
torch.full(..., clean=-1) (a schedule sentinel, OOD to the backbone's time
embedding) -> use level 0 (the cleanest TRAINED level). Anchored on the
pixel-decoupled-specific `x_rgb = rgb_ctx` so it does NOT touch other methods."""
p = "egomimic/algo/dfot/algo.py"
s = open(p).read()
old = """        obs_levels = torch.full(
            (1, T), clean, device=device,
            dtype=torch.long if dts is not None else torch.float32,
        )
        act_sched = vanilla_schedule(n_steps, T, discrete_timesteps=dts).to(device)

        x_rgb = rgb_ctx"""
new = """        obs_levels = torch.zeros(
            (1, T), device=device,
            dtype=torch.long if dts is not None else torch.float32,
        )
        act_sched = vanilla_schedule(n_steps, T, discrete_timesteps=dts).to(device)

        x_rgb = rgb_ctx"""
assert old in s, "pixel_decoupled obs_levels block not found"
if "obs_levels = torch.zeros(" not in s.split("x_rgb = rgb_ctx")[0]:
    s = s.replace(old, new)
    open(p, "w").write(s)
    print("FIXED obs_levels -> 0 in pixel_decoupled")
else:
    print("already fixed")
