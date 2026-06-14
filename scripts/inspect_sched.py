import torch, sys
p = sys.argv[1]
c = torch.load(p, map_location="cpu", weights_only=False)
print("epoch:", c.get("epoch"), "global_step:", c.get("global_step"))
lrs = c.get("lr_schedulers")
print("lr_schedulers type:", type(lrs))
if lrs:
    s = lrs[0]
    print("top keys:", list(s.keys()))
    for k, v in s.items():
        if k == "_schedulers":
            for i, sub in enumerate(v):
                print("  sub[%d]: T_max=%s last_epoch=%s eta_min=%s base_lrs=%s" % (
                    i, sub.get("T_max"), sub.get("last_epoch"), sub.get("eta_min"), sub.get("base_lrs")))
        else:
            print("  ", k, "=", v)
opt = c.get("optimizer_states")
if opt:
    print("optim param_group lr:", [pg.get("lr") for pg in opt[0]["param_groups"]])
