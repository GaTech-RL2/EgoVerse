"""Generate the DP-recipe-matched, ~260M HPT-transformer flow config (Option B).

Width cascade 128 -> 1280 across every trunk-tied dim; depth 6 -> 12; heads
4 -> 16; horizon 32 -> 16; head num_inference_steps 50 -> 100. obs_horizon
stays 1 and crop stays 86 (flagged deltas vs DP). Robust regex replace:
(?<![\\w]) excludes `modality_embed_dim`/`crossattn_heads`; (?![\\d]) excludes
matching 128 inside 1280. Every edit is count-asserted."""
import pathlib, re

CFG = pathlib.Path("egomimic/hydra_configs/model")
D = 1280
BLOCKS = 12
HEADS = 16
HORIZON = 16

def kv(s, key, old, new, n=1):
    """Replace `key: old` -> `key: new`, key not preceded by a word char, old
    not followed by a digit. Asserts exactly n replacements."""
    pat = re.compile(r'(?<![\w])' + re.escape(f"{key}: {old}") + r'(?![\d])')
    s2, cnt = pat.subn(f"{key}: {new}", s)
    assert cnt == n, f"{key}: {old} -> expected {n}, got {cnt}"
    return s2

def lit(s, old, new, n=1):
    assert s.count(old) == n, f"{old!r}: expected {n}, got {s.count(old)}"
    return s.replace(old, new)

# ---- 1) common trunk ------------------------------------------------------
c = (CFG / "_hpt_pushshapes_128_common.yaml").read_text()
c = kv(c, "embed_dim", 128, D)                 # trunk.embed_dim (not modality_embed_dim)
c = kv(c, "modality_embed_dim", 128, D)
c = kv(c, "num_blocks", 6, BLOCKS)
c = kv(c, "num_heads", 4, HEADS)               # trunk heads (not crossattn_heads)
c = kv(c, "action_horizon", 32, HORIZON)
c = kv(c, "input_dim", 128, D)                 # shared_stem input (state stem input_dim:5 untouched)
c = kv(c, "output_dim", 128, D, n=2)           # shared_stem + encoder ResNet
c = lit(c, "widths: [128]", f"widths: [{D}]")
(CFG / "_hpt_pushshapes_260m_common.yaml").write_text(c)
print("wrote _hpt_pushshapes_260m_common.yaml")

# ---- 2) circle model (state stem + flow head) -----------------------------
m = (CFG / "hpt_pushshapes_circle.yaml").read_text()
m = lit(m, "- _hpt_pushshapes_128_common", "- _hpt_pushshapes_260m_common")
m = kv(m, "output_dim", 128, D)                # state stem (input_dim:5 untouched)
m = kv(m, "modality_embed_dim", 128, D)
m = lit(m, "widths: [128]", f"widths: [{D}]")
m = kv(m, "action_horizon", 32, HORIZON)       # flow head
m = kv(m, "num_inference_steps", 50, 100)
m = kv(m, "cond_dim", 128, D)
m = kv(m, "act_seq", 32, HORIZON)
m = kv(m, "hidden_dim", 64, 256)
m = kv(m, "nblocks", 4, 6)
(CFG / "hpt_pushshapes_circle_260m.yaml").write_text(m)
print("wrote hpt_pushshapes_circle_260m.yaml")

# ---- 3) dpmatch flow config (GN ResNet@1280 + crop86 + EMA-ready) ----------
e = (CFG / "hpt_bc_flow_pushshapes_ema.yaml").read_text()
e = lit(e, "- hpt_pushshapes_circle", "- hpt_pushshapes_circle_260m")
e = kv(e, "output_dim", 128, D)                # GN ResNet output must match trunk width
(CFG / "hpt_bc_flow_pushshapes_dpmatch.yaml").write_text(e)
print("wrote hpt_bc_flow_pushshapes_dpmatch.yaml")
print("OK")
