import re
base = open("egomimic/hydra_configs/model/bf_prdec_nopre.yaml").read()
assert base.count("decoder_layout: T4") == 2, base.count("decoder_layout: T4")
for name, e, d in (("bf_nopre_e6d2", 6, 2), ("bf_nopre_e7d1", 7, 1)):
    s = base.replace("decoder_layout: T4", "decoder_layout: T%d" % d)
    s = re.sub(r"n_layers: 4\n(      decoder_layout:)", "n_layers: %d\n\\1" % e, s)
    assert s.count("decoder_layout: T%d" % d) == 2, name
    assert s.count("n_layers: %d" % e) == 2, name
    open("egomimic/hydra_configs/model/%s.yaml" % name, "w").write(s)
    print(name, "OK: enc", e, "x2, dec T%d" % d, "x2")
