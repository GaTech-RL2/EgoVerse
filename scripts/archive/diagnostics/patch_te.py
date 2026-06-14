#!/usr/bin/env python3
"""Additively insert an env-gated CHUNK_TRAJ_DIR dump into _rollout_chunk_te,
mirroring the one already in _rollout_chunk_openloop. Idempotent."""
import io, sys

PATH = "egomimic/eval/eval_sim.py"
with io.open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

if "chunktraj_te_ep" in src:
    print("ALREADY_PATCHED")
    sys.exit(0)

# Anchor: the TE method's trailing debug block. Insert the dump just before it.
ANCHOR = '''        if actions_taken:
            _a = np.asarray(actions_taken)
            print(
                f"[HNET_CHUNKTE_DBG] ep={ep_idx} steps={len(actions_taken)} "
                f"act_x[{_a[:,0].min():.1f},{_a[:,0].max():.1f}] final_cov={last_coverage:.3f}",
                flush=True,
            )
        return last_coverage, frames, actions_taken'''

if ANCHOR not in src:
    print("ANCHOR_NOT_FOUND")
    sys.exit(2)

if src.count(ANCHOR) != 1:
    print("ANCHOR_NOT_UNIQUE count=%d" % src.count(ANCHOR))
    sys.exit(3)

DUMP = '''        # --- env-gated executed-trajectory dump (CHUNK_TRAJ_DIR); first ~4 eps ---
        # Mirrors _rollout_chunk_openloop's dump: guarded by an env var, wrapped
        # in try/except, additive; no behaviour change when unset.
        _ctd = os.environ.get("CHUNK_TRAJ_DIR")
        if _ctd and ep_idx < 4 and actions_taken:
            try:
                os.makedirs(_ctd, exist_ok=True)
                _acts = np.asarray(actions_taken, dtype=np.float64)  # (T,2) WORLD
                np.savez(
                    os.path.join(_ctd, f"chunktraj_te_ep{ep_idx}.npz"),
                    actions=_acts,
                    K=int(K),
                    coverage=float(last_coverage),
                )
                print(
                    f"[CHUNK_TRAJ_TE] ep{ep_idx} saved actions{_acts.shape} "
                    f"K={K} cov={last_coverage:.3f}",
                    flush=True,
                )
            except Exception as _ex:
                import traceback
                print(f"[CHUNK_TRAJ_TE_ERR] {_ex}\\n{traceback.format_exc()}", flush=True)
'''

new = src.replace(ANCHOR, DUMP + ANCHOR, 1)
with io.open(PATH, "w", encoding="utf-8") as f:
    f.write(new)
print("PATCHED_OK")
