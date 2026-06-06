f = "egomimic/algo/dfot/algo.py"
s = open(f).read()
old = '''        n_done = len(self._pp_act)
        ctx_idx = [max(0, n_done - n_ctx + i) for i in range(n_ctx)]
        T = n_ctx + k

        ctx_frames = []
        for i in ctx_idx:
            rgb = self._pp_rgb[min(i, len(self._pp_rgb) - 1)]
            a = self._pp_act[i] if i < len(self._pp_act) else torch.zeros(A, device=device)'''
new = '''        T = n_ctx + k
        nrgb = len(self._pp_rgb)
        nact = len(self._pp_act)

        ctx_frames = []
        for j in range(n_ctx):
            ri = max(0, nrgb - n_ctx + j)               # most-recent n_ctx OBSERVED frames
            rgb = self._pp_rgb[ri]
            ai = nact - n_ctx + j                        # the action that produced that obs
            a = self._pp_act[ai] if 0 <= ai < nact else torch.zeros(A, device=device)'''
if "nrgb = len(self._pp_rgb)" in s:
    print("already")
elif old in s:
    open(f, "w").write(s.replace(old, new, 1)); print("patched")
else:
    print("anchor not found")
