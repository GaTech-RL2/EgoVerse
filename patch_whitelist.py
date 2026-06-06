f = "egomimic/algo/dfot/algo.py"
s = open(f).read()
old = '{"ar", "chunk", "spatial_rh", "spatial_decoupled"}'
new = '{"ar", "chunk", "spatial_rh", "spatial_decoupled", "pixel_policy"}'
if new in s:
    print("already")
elif old in s:
    open(f, "w").write(s.replace(old, new, 1))
    print("patched")
else:
    print("anchor not found")
