"""Stride-alignment + HNet chunk-32 train-vs-rollout parity proof for the
7 new H-Net chunk-size leaves (bcrnnHnetC{4,16,32}[Q], C8Q).

Run on the A40 compute node inside the EgoVerse7 venv with PYTHONPATH=.

PART A: synthetic action[t]=t alignment through the REAL _cut_windows_strided
        for strides/chunks {4,8,16,32}. For obs-step k with start s:
          obs frame  f_k    = s + sigma*k
          act target [f_k, f_k+C)  ==  [sigma*k+s , sigma*k+s+C)
        With action[t]=t (scalar per frame), act_w[win,k,j] MUST equal sigma*k+s+j
        (clamped to L-1 in the repeat-pad tail). 0 mismatches required.
        Special focus: chunk-32 -> obs-step k targets [32k, 32k+32).

PART B: train-vs-rollout parity on the HNet chunk-32 CORE. forward(window)
        (teacher-forced) vs sequential step() over the same window with a fresh
        rolling buffer. The per-obs-step features h_k must match at ~1e-6 -- this
        is the property that makes train == rollout for the chunked AR readout.
"""
import sys
import torch

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float32

from egomimic.algo.bc_rnn import _cut_windows_strided
from egomimic.models.bc_rnn_nets import HNetCore

print("=" * 72)
print("PART A: SYNTHETIC action[t]=t ALIGNMENT through _cut_windows_strided")
print("=" * 72)

# Build a synthetic episode batch. action[t] = t (a single scalar feature D=1)
# so we can read the frame index straight out of the gathered chunk.
B = 2
L0, L1 = 300, 257          # two episode lengths (257 forces tail clamp at C=32)
T = max(L0, L1)
D = 1
seq_lens = torch.tensor([L0, L1], device=DEV)

# actions_padded[b, t, 0] = t  (the frame index itself)
actions_padded = torch.zeros(B, T, D, device=DEV, dtype=DT)
for b in range(B):
    actions_padded[b, :, 0] = torch.arange(T, device=DEV, dtype=DT)
# obs_padded: also carry the frame index so we can verify obs alignment too.
obs_padded = {"frame": actions_padded.clone()}  # (B,T,1)

H = 10  # rnn_horizon

all_ok = True
total_mismatch = 0
for sigma in (4, 8, 16, 32):
    C = sigma  # chunk_len == obs_stride for these runs
    # window starts: a few per episode, including s=0 and a tail-hitting start.
    pairs = [(0, 0), (0, 5), (1, 0), (1, 13)]
    obs_w, act_w, mask_w = _cut_windows_strided(
        obs_padded, actions_padded, seq_lens, pairs, H, sigma, C, repeat=True
    )
    # act_w: (Nw, H, C, D); expected[win,k,j] = clamp(sigma*k + s + j, max=L-1)
    n_mis = 0
    for wi, (b, s) in enumerate(pairs):
        L = int(seq_lens[b].item())
        last = L - 1
        for k in range(H):
            f_k = sigma * k + s
            # obs alignment: obs frame must be clamp(f_k, last)
            exp_obs = min(f_k, last)
            got_obs = int(round(obs_w["frame"][wi, k, 0].item()))
            if got_obs != exp_obs:
                n_mis += 1
            for j in range(C):
                exp = min(f_k + j, last)            # repeat-clamped target frame
                got = int(round(act_w[wi, k, j, 0].item()))
                if got != exp:
                    n_mis += 1
    total_mismatch += n_mis
    # explicit chunk-32 spot statement for k in {0,1,2}, start s=0, episode 0
    extra = ""
    if sigma == 32:
        b, s = pairs[0]
        rng = []
        for k in (0, 1, 2):
            lo = act_w[0, k, 0, 0].item()
            hi = act_w[0, k, C - 1, 0].item()
            rng.append(f"k={k}->[{int(lo)},{int(hi)}]")
        extra = "  chunk32 targets: " + ", ".join(rng) + " (i.e. [32k,32k+31])"
    status = "OK" if n_mis == 0 else f"FAIL({n_mis})"
    all_ok = all_ok and (n_mis == 0)
    print(f"  sigma={sigma:2d} chunk={C:2d}  windows={len(pairs)}  mismatches={n_mis:3d}  [{status}]{extra}")

print(f"  ---> TOTAL MISMATCHES = {total_mismatch}  ({'ALL ALIGNED' if all_ok else 'ALIGNMENT FAILURE'})")
assert all_ok, "STRIDE ALIGNMENT FAILED"

print("=" * 72)
print("PART B: HNet CHUNK-32 TRAIN-vs-ROLLOUT PARITY (core forward == step())")
print("=" * 72)
IN_DIM = 66
D_MODEL = 256
MAXW = 10  # rnn_horizon (obs-steps); chunk_len=32 does NOT change the obs window

torch.manual_seed(42)
core = HNetCore(input_dim=IN_DIM, d_model=D_MODEL, n_heads=8, d_intermediate=768,
                outer_layers=4, inner_layers=6, target_compression_ratio=2.0,
                ratio_loss_weight=0.03, max_window=MAXW, causal=True).to(DEV).to(DT).eval()
n_core = sum(p.numel() for p in core.parameters())
print(f"  HNetCore params = {n_core/1e6:.4f}M  (d_model={D_MODEL}, causal=True)")

Bp = 3
x = torch.randn(Bp, MAXW, IN_DIM, device=DEV, dtype=DT)
with torch.no_grad():
    full, _ = core(x)                       # (B, T, d_model) teacher-forced
    h = core.init_hidden(Bp, device=DEV, dtype=DT)
    step_outs = []
    for t in range(MAXW):
        h_t, h = core.step(x[:, t], h)      # rolling-buffer single step
        step_outs.append(h_t)
    step_stack = torch.stack(step_outs, dim=1)   # (B, T, d_model)
    maxdiff = (full - step_stack).abs().max().item()
print(f"  train-vs-rollout per-step feature maxdiff = {maxdiff:.3e}")
PASS_TOL = 1e-5
ok_parity = maxdiff < PASS_TOL
print(f"  ---> {'PARITY OK' if ok_parity else 'PARITY FAIL'} (tol {PASS_TOL:.0e})")
assert ok_parity, f"PARITY FAILED maxdiff={maxdiff}"

print("=" * 72)
print(f"PROOF RESULT: stride_mismatches={total_mismatch}  parity_maxdiff={maxdiff:.3e}")
print("ALL CHECKS PASSED" if (all_ok and ok_parity) else "FAILED")
print("=" * 72)
