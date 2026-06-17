"""Verify the reorg preserved every leaf's resolved config (subprocess per
config so Hydra's global singleton never leaks between composes).

Usage: srun ... .venv/bin/python scripts/_reorg_verify.py model
"""
import os
import re
import subprocess
import sys

GROUP = sys.argv[1]
BASE = f"/coc/flash7/paphiwetsa3/reorg_verify/{GROUP}_base"

RULES = {
    "model": [
        ("bc_rnn", r"^_?bc_?rnn_", r"^(_?)bc_?rnn_"),
        ("dfot", r"^dfot_", r"^dfot_"),
        ("hnet", r"^_?hnet_", r"^(_?)hnet_"),
        ("hpt", r"^_?hpt_", r"^(_?)hpt_"),
        ("pi", r"^pi0\.5_", r"^pi0\.5_"),
        ("vae", r"^vae_", r"^vae_"),
    ],
    "data": [
        ("tsimulation", r"^_?tsim", r"^(_?)tsim(ulation)?_?"),
        ("gmm", r"^gmm_", r"^gmm_"),
        ("cotrain", r"^cotrain_", r"^cotrain_"),
        ("aria", r"^aria", r"^aria_?"),
        ("eva", r"^eva", r"^eva_?"),
        ("mecka", r"^mecka", r"^mecka_?"),
        ("scale", r"^scale", r"^scale_?"),
    ],
}[GROUP]
KEEP_FLAT = {
    "model": {"act", "egobridge"},
    "data": {"_pickplace_qwen_base", "bc_pickplace_eva_qwen", "industry_eva_pi", "video_clips"},
}[GROUP]


def newname(stem):
    if stem in KEEP_FLAT:
        return stem
    for f, m, s in RULES:
        if re.match(m, stem):
            has = "(_?)" in s
            n = re.sub(s, (r"\1" if has else ""), stem) or f
            return f"{f}/{n}"
    return stem


env = {**os.environ, "PYTHONPATH": "."}
ok = 0
mism = []
fails = []
for fn in sorted(os.listdir(BASE)):
    if not fn.endswith(".yaml"):
        continue
    old = fn[:-5]
    new = newname(old)
    r = subprocess.run(
        [".venv/bin/python", "scripts/_cfg_resolve.py", GROUP, new],
        capture_output=True, text=True, env=env,
    )
    if r.returncode != 0:
        tail = (r.stderr.strip().splitlines() or ["?"])[-1][:140]
        fails.append((old, new, tail))
        continue
    want = open(os.path.join(BASE, fn)).read()
    if r.stdout == want:
        ok += 1
    else:
        mism.append((old, new))

print(f"ok={ok}  mismatch={len(mism)}  fail={len(fails)}")
for o, n in mism[:40]:
    print("  MISMATCH", o, "->", n)
for o, n, e in fails[:40]:
    print("  FAIL", o, "->", n, "::", e)
