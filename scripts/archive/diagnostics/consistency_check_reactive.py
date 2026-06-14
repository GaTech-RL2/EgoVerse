"""Forward-vs-generate consistency check for the reactive chunk path.

Isolates the part that matters for train/eval consistency: the backbone +
action head over the [c_t, BOS] 2-token sequence. We stub _encode_cond to
return a FIXED cond tensor so both paths see identical conditioning, then
compare:

  * forward_packed(reactive)  -- the TRAINING path (packed) -> (T_total, K, A)
  * generate (chunk_k>1)      -- the EVAL/rollout path      -> (B, K, A)

For the SAME obs/cond they must produce the same K-action chunk (up to
full-attn vs cached-step numerics). Prints max abs diff.
"""
import torch
from egomimic.algo.hnet import FlatFusedPolicy


class _StubCond:
    """Stand-in cond_encoder: returns a fixed per-frame fused cond."""

    def __init__(self, d_cond, fixed):
        self.d_cond = d_cond
        self._fixed = fixed  # (d_cond,)

    def encode(self, obs, T):
        # obs content is ignored; we only need a batch dim. forward_packed
        # passes obs as (1, T_total, ...) (already unsqueezed); generate passes
        # (B, T, ...). Either way obs["x"].shape[0] is the batch dim we want.
        any_v = next(iter(obs.values()))
        B = any_v.shape[0]
        c = self._fixed.view(1, 1, -1).expand(B, T, -1).clone()
        return {"fused_cond": c}


def main():
    torch.manual_seed(0)
    A = 4
    K = 32
    d_model = 256
    d_cond = 64
    AH = 1024

    fixed = torch.randn(d_cond)
    cond = _StubCond(d_cond, fixed)

    pol = FlatFusedPolicy(
        action_dim=A,
        action_horizon=AH,
        d_model=d_model,
        d_cond=d_cond,
        cond_encoder=cond,
        arch_layout="T8",
        num_heads=4,
        d_intermediate=512,
        action_head_cfg={"mode": "continuous", "chunk_k": K},
    )
    pol.reactive = True
    pol.token_dropout_p = 0.0
    pol = pol.float().eval()

    B = 1
    Ttot = 1  # single packed frame == single generate frame

    # forward_packed obs: per-frame (T_total, feat); generate/forward obs:
    # (B, T, feat). Content is ignored by the stub cond encoder.
    obs_packed = {"x": torch.zeros(Ttot, 3)}
    obs_btf = {"x": torch.zeros(B, 1, 3)}

    # TRAINING path: forward_packed with one packed frame.
    actions_packed = torch.zeros(Ttot, A)
    cu = torch.tensor([0, Ttot], dtype=torch.long)
    with torch.no_grad():
        pred_fp, _ = pol.forward_packed(actions_packed, obs_packed, cu, Ttot)  # (Ttot, K, A)
    chunk_train = pred_fp[0]  # (K, A)

    # EVAL path: generate (chunk_k>1 branch).
    with torch.no_grad():
        chunk_gen = pol.generate(obs_btf, batch_size=B, device="cpu", T=K)  # (B, K, A)
    chunk_eval = chunk_gen[0]  # (K, A)

    print("chunk_train shape", tuple(chunk_train.shape))
    print("chunk_eval  shape", tuple(chunk_eval.shape))
    diff = (chunk_train - chunk_eval).abs()
    print("MAX_ABS_DIFF", float(diff.max()))
    print("MEAN_ABS_DIFF", float(diff.mean()))
    ok = float(diff.max()) < 1e-4
    print("CONSISTENT", ok)

    # Sanity: also check the padded reactive forward() matches, single frame.
    with torch.no_grad():
        pred_pad, _ = pol.forward(actions_packed.view(1, 1, A), obs_btf)  # (1,1,K,A)
    chunk_pad = pred_pad[0, 0]
    d2 = (chunk_pad - chunk_eval).abs().max()
    print("PADDED_FWD_VS_GEN_MAX_ABS_DIFF", float(d2))


if __name__ == "__main__":
    main()
