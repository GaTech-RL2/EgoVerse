"""Compose-check the 7 new H-Net chunk leaves + the chunk8 base, asserting the
core stays HNetCore and the knobs resolve correctly.

For each model config: compose model=<leaf> under the real train config root,
resolve, and report:
  core (robomimic_model.core), core_net._target_, core_net.d_model,
  obs_stride, chunk_len, gmm_head.chunk_len, rnn_horizon, chunk_head,
  query_decoder._target_/d_model/chunk_len (for _q leaves).
"""
import sys
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

LEAVES = [
    ("bc_rnn_pushshapes_paperexact_hnet_chunk8",    "C8 (base, LINEAR)", 8,  "linear"),
    ("bc_rnn_pushshapes_paperexact_hnet_chunk4",    "C4 LINEAR",         4,  "linear"),
    ("bc_rnn_pushshapes_paperexact_hnet_chunk16",   "C16 LINEAR",        16, "linear"),
    ("bc_rnn_pushshapes_paperexact_hnet_chunk32",   "C32 LINEAR",        32, "linear"),
    ("bc_rnn_pushshapes_paperexact_hnet_chunk4_q",  "C4 QUERY",          4,  "queries"),
    ("bc_rnn_pushshapes_paperexact_hnet_chunk8_q",  "C8 QUERY",          8,  "queries"),
    ("bc_rnn_pushshapes_paperexact_hnet_chunk16_q", "C16 QUERY",         16, "queries"),
    ("bc_rnn_pushshapes_paperexact_hnet_chunk32_q", "C32 QUERY",         32, "queries"),
]

CFG_MODULE = "egomimic.hydra_configs"
ROOT_CFG = "train_zarr_cartesian"

all_ok = True
lines = []
with initialize_config_module(version_base=None, config_module=CFG_MODULE):
    for leaf, label, want_chunk, want_head in LEAVES:
        try:
            cfg = compose(config_name=ROOT_CFG, overrides=[f"model={leaf}"])
        except Exception as e:
            print(f"[{label}] COMPOSE ERROR: {type(e).__name__}: {e}")
            all_ok = False
            continue
        rm = cfg.model.robomimic_model
        core = OmegaConf.select(rm, "core")
        core_tgt = OmegaConf.select(rm, "core_net._target_")
        d_model = OmegaConf.select(rm, "core_net.d_model")
        obs_stride = OmegaConf.select(rm, "obs_stride")
        chunk_len = OmegaConf.select(rm, "chunk_len")
        gmm_cl = OmegaConf.select(rm, "gmm_head.chunk_len")
        rnn_h = OmegaConf.select(rm, "rnn_horizon")
        chunk_head = OmegaConf.select(rm, "chunk_head") or "linear"
        causal = OmegaConf.select(rm, "core_net.causal")

        is_hnet = (core == "hnet") and (core_tgt is not None) and core_tgt.endswith("HNetCore")
        d_ok = (d_model == 256)
        stride_ok = (obs_stride == want_chunk)
        chunk_ok = (chunk_len == want_chunk)
        gmm_ok = (gmm_cl == want_chunk)
        rnn_ok = (rnn_h == 10)
        head_ok = (chunk_head == want_head)
        causal_ok = (causal is True)

        qline = ""
        if want_head == "queries":
            qd_tgt = OmegaConf.select(rm, "query_decoder._target_")
            qd_dm = OmegaConf.select(rm, "query_decoder.d_model")
            qd_cl = OmegaConf.select(rm, "query_decoder.chunk_len")
            qd_ps = OmegaConf.select(rm, "query_decoder.per_step")
            qd_ok = (qd_tgt is not None and qd_tgt.endswith("QueryActionDecoder")
                     and qd_dm == 256 and qd_cl == want_chunk and qd_ps == 25)
            head_ok = head_ok and qd_ok
            qline = (f"  query_decoder: target={qd_tgt.split('.')[-1] if qd_tgt else None} "
                     f"d_model={qd_dm} chunk_len={qd_cl} per_step={qd_ps} -> {'OK' if qd_ok else 'FAIL'}")

        leaf_ok = all([is_hnet, d_ok, stride_ok, chunk_ok, gmm_ok, rnn_ok, head_ok, causal_ok])
        all_ok = all_ok and leaf_ok
        verdict = "PASS" if leaf_ok else "FAIL"
        tgt_short = core_tgt.split(".")[-1] if core_tgt else None
        line = (f"[{label:18s}] core={core}/{tgt_short} d_model={d_model} causal={causal} | "
                f"obs_stride={obs_stride} chunk_len={chunk_len} gmm.chunk_len={gmm_cl} "
                f"rnn_horizon={rnn_h} chunk_head={chunk_head} -> {verdict}")
        print(line)
        if qline:
            print(qline)
        lines.append(line)

print("=" * 72)
print("COMPOSE-CHECK RESULT:", "ALL PASS" if all_ok else "SOME FAILED")
sys.exit(0 if all_ok else 1)
