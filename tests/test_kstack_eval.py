"""K action-chunk stacking (D07) — EVAL side.

Pure-torch CPU tests, no data, no env, no GPU:

  * ``decode_action_planes(chunk_k=K)`` — unstacks a keyframe's stacked
    K-action planes to ``(..., K, A)`` time-major (upstream
    ``df_pusht.py:117`` channel-split semantics); ``chunk_k=1`` is
    bit-identical to the legacy decode.
  * ``DFoTPixelPolicyOverlayEval`` — end-to-end on a stub stage/algo:
    at K>1 the metrics score the unrolled K*T action sequence (per-step
    world-px error, npz ``steps`` = ``keyframe*K + chunk_i``), GT pairing
    stays aligned, and unnormalize runs on per-underlying-dim (A-wide)
    rows (the (A,)-shaped norm stats would not broadcast otherwise);
    K=1 / a stage without the knob reproduces the legacy per-frame
    semantics exactly.
  * ``SimRolloutEval._rollout_one`` — a flat (K*2,) ``inference_step``
    return is consumed as a time-ordered open-loop chunk (model queried
    only at keyframes); (2,)-returns keep the legacy call-every-step
    contract bit-identically, odd sizes keep the historical [:2] cut.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from egomimic.eval.core.eval_sim import PackedSimEval
from egomimic.eval.dfot.eval_dfot_pixel_overlay import (
    DFoTPixelPolicyOverlayEval,
    decode_action_planes,
)
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


EMB_ID = get_embodiment_id("pushshapes_sim")


# --------------------------------------------------------------------- #
# decode_action_planes chunk_k
# --------------------------------------------------------------------- #


def test_decode_kchunk_unstacks_time_major():
    Ci, A, K, H, W = 3, 2, 2, 6, 6
    Ca = K * A
    # chunk values laid out time-major per keyframe: [x0, y0, x1, y1]
    vals = torch.tensor(
        [[0.1, -0.2, 0.3, -0.4], [0.5, 0.6, -0.7, -0.8]]
    )  # (T=2, K*A)
    bundle = torch.full((2, Ci + Ca, H, W), 7.0)  # poisoned image channels
    bundle[:, Ci:] = vals.reshape(2, Ca, 1, 1).expand(2, Ca, H, W)
    out = decode_action_planes(bundle, Ci, Ca, A, chunk_k=K)
    assert out.shape == (2, K, A)
    assert torch.allclose(out, vals.reshape(2, K, A))

    out5 = decode_action_planes(
        bundle.unsqueeze(0).expand(3, 2, Ci + Ca, H, W), Ci, Ca, A, chunk_k=K
    )
    assert out5.shape == (3, 2, K, A)
    assert torch.allclose(out5[1], vals.reshape(2, K, A))


def test_decode_kchunk_k1_bit_identical_to_legacy():
    torch.manual_seed(0)
    Ci, Ca, A = 3, 2, 2
    bundle = torch.randn(4, Ci + Ca, 5, 5)
    legacy = bundle[..., Ci : Ci + Ca, :, :].mean(dim=(-2, -1))[..., :A]
    assert torch.equal(decode_action_planes(bundle, Ci, Ca, A), legacy)
    assert torch.equal(decode_action_planes(bundle, Ci, Ca, A, chunk_k=1), legacy)


# --------------------------------------------------------------------- #
# overlay eval end-to-end on stubs
# --------------------------------------------------------------------- #


class _MinMaxNormStats:
    """Stub of MultiDataset's minmax unnormalize: (t+1)*0.5*(mx-mn+1e-6)+mn.

    Stats are (A,)-shaped per UNDERLYING action dim — a regression to
    unnormalizing un-unrolled (N, K*A) rows fails to broadcast here.
    """

    def __init__(self, mn, mx):
        self.mn = torch.as_tensor(mn, dtype=torch.float32)
        self.mx = torch.as_tensor(mx, dtype=torch.float32)

    def unnormalize(self, data: dict, emb_id: int) -> dict:
        return {
            k: (v + 1) * 0.5 * (self.mx - self.mn + 1e-6) + self.mn
            for k, v in data.items()
        }


class _StubOuter:
    pixel_mode = "policy"
    image_range = "01"

    def __init__(self, Ci, A, K=None):
        self._image_channels = Ci
        self.action_dim = A
        self._action_channels = (K or 1) * A
        if K is not None:
            self.action_chunk_k = K


class _StubAlgo:
    def __init__(self, outer, ac_key="actions"):
        self.outer_stage = outer
        self.resolved_ac_keys = {EMB_ID: ac_key}
        self.norm_stats = _MinMaxNormStats(mn=[0.0, 0.0], mx=[512.0, 512.0])


_ERR_NORM = 64.0 / (0.5 * (512.0 + 1e-6))  # 64 world px in normalized units


def _run_overlay(monkeypatch, tmp_path, K_attr, T=6, n_ctx=2, offset_x=True, commit_k=1):
    """Drive compute_metrics_and_viz on a scripted predictor that returns the
    GT chunk planes (+ a 64-px x-offset on every action when offset_x)."""
    Ci, A, H = 3, 2, 4
    K = K_attr or 1
    Ca = K * A
    torch.manual_seed(0)
    acts = torch.rand(1, T, Ca) * 0.8 - 0.4  # normalized GT, (1, T, K*A)
    imgs = torch.rand(1, T, Ci, H, H)

    ev = DFoTPixelPolicyOverlayEval(
        n_context_frames=n_ctx, commit_k=commit_k, max_episodes=1, max_videos=1
    )
    ev.trainer = SimpleNamespace(
        lightning_module=SimpleNamespace(device=torch.device("cpu")),
        current_epoch=0,
    )
    ev.model = _StubAlgo(_StubOuter(Ci, A, K=K_attr))
    monkeypatch.setattr(ev, "video_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        ev,
        "_render_overlay_panel",
        lambda *a, **kw: np.zeros((8, 8, 3), dtype=np.uint8),
    )

    state = {"a": n_ctx}

    def fake_predict(algo, ctx_bundle, k, device):
        a = state["a"]
        state["a"] += k
        rows = acts[0, a : a + k].clone()  # (k, K*A) GT chunk rows
        if offset_x:
            rows.reshape(k, K, A)[..., 0] += _ERR_NORM
        rgb = torch.zeros(k, Ci, H, H)
        planes = rows.reshape(k, Ca, 1, 1).expand(k, Ca, H, H)
        return torch.cat([rgb, planes], dim=1)

    monkeypatch.setattr(ev, "_predict_window", fake_predict)
    metrics, _ = ev.compute_metrics_and_viz({EMB_ID: {"front_img_1": imgs, "actions": acts}})
    npz = np.load(tmp_path / "epoch_0" / f"overlay_emb{EMB_ID}_ep0.npz")
    return metrics, npz


def test_overlay_k2_scores_unrolled_chunk_sequence(monkeypatch, tmp_path):
    T, n_ctx, K = 6, 2, 2
    metrics, npz = _run_overlay(monkeypatch, tmp_path, K_attr=K, T=T, n_ctx=n_ctx)
    n_unrolled = (T - n_ctx) * K
    assert metrics[f"Valid/emb{EMB_ID}_overlay_n_steps"].item() == n_unrolled
    # 64-px error on x only => per-step sq = 64^2/2, rmse = 64/sqrt(2)
    assert metrics[f"Valid/emb{EMB_ID}_overlay_action_rmse_px"].item() == pytest.approx(
        64.0 / np.sqrt(2.0), rel=1e-4
    )
    # npz steps are unrolled per-action indices: keyframe*K + chunk_i
    expected = [a * K + i for a in range(n_ctx, T) for i in range(K)]
    assert npz["steps"].tolist() == expected
    assert npz["pred_world"].shape == (n_unrolled, 2)
    assert np.allclose(npz["pred_world"][:, 0] - npz["gt_world"][:, 0], 64.0, rtol=1e-4)
    assert np.allclose(npz["pred_world"][:, 1], npz["gt_world"][:, 1], atol=1e-4)


def test_overlay_k2_perfect_prediction_zero_rmse(monkeypatch, tmp_path):
    metrics, _ = _run_overlay(monkeypatch, tmp_path, K_attr=4, offset_x=False)
    assert metrics[f"Valid/emb{EMB_ID}_overlay_action_rmse_px"].item() == pytest.approx(
        0.0, abs=1e-3
    )
    assert metrics[f"Valid/emb{EMB_ID}_overlay_pred_std_px"].item() > 0.0


def test_overlay_commit_k_composes_with_chunk_k(monkeypatch, tmp_path):
    # per model tick: commit_k keyframes x K actions each, all unrolled in
    # time order (the sp_commit x K composition).
    T, n_ctx, K, ck = 6, 2, 2, 2
    metrics, npz = _run_overlay(
        monkeypatch, tmp_path, K_attr=K, T=T, n_ctx=n_ctx, commit_k=ck
    )
    n_unrolled = (T - n_ctx) * K  # keyframes 2..5 covered in two ticks of 2
    assert metrics[f"Valid/emb{EMB_ID}_overlay_n_steps"].item() == n_unrolled
    expected = [a * K + i for a in range(n_ctx, T) for i in range(K)]
    assert npz["steps"].tolist() == expected
    assert metrics[f"Valid/emb{EMB_ID}_overlay_action_rmse_px"].item() == pytest.approx(
        64.0 / np.sqrt(2.0), rel=1e-4
    )


def test_overlay_stage_without_knob_is_legacy_k1(monkeypatch, tmp_path):
    # stage predates the action_chunk_k knob => getattr default 1 => legacy
    # per-frame semantics: steps = keyframe indices, one row per keyframe.
    T, n_ctx = 6, 2
    metrics, npz = _run_overlay(monkeypatch, tmp_path, K_attr=None, T=T, n_ctx=n_ctx)
    assert metrics[f"Valid/emb{EMB_ID}_overlay_n_steps"].item() == T - n_ctx
    assert npz["steps"].tolist() == list(range(n_ctx, T))
    assert metrics[f"Valid/emb{EMB_ID}_overlay_action_rmse_px"].item() == pytest.approx(
        64.0 / np.sqrt(2.0), rel=1e-4
    )


# --------------------------------------------------------------------- #
# sim eval consumption of K-chunk returns
# --------------------------------------------------------------------- #


class _RecordingEnv:
    def __init__(self, n_steps):
        self.n = n_steps
        self.t = 0
        self.actions = []

    def reset(self, seed=None):
        self.t = 0

    def _get_obs(self):
        return {
            "agent_pos": np.zeros(2, dtype=np.float32),
            "object_pose": np.zeros(3, dtype=np.float32),
            "image": np.zeros((8, 8, 3), dtype=np.uint8),
        }

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.t += 1
        return None, None, self.t >= self.n, None, {"coverage": 0.0}

    def render(self):
        return None


class _ChunkAlgo:
    """Returns a flat (K*2,) time-ordered chunk; values encode (tick, slot)."""

    def __init__(self, K):
        self.K = K
        self.calls = []

    def inference_step(self, obs_zarr, t, emb_id, T_max=None):
        self.calls.append(t)
        base = 100.0 * (len(self.calls) - 1)
        chunk = [[base + i, -(base + i)] for i in range(self.K)]
        return np.asarray(chunk, dtype=np.float32).reshape(-1)


class _FixedAlgo:
    def __init__(self, ret):
        self.ret = np.asarray(ret, dtype=np.float32)
        self.calls = []

    def inference_step(self, obs_zarr, t, emb_id, T_max=None):
        self.calls.append(t)
        return self.ret


def _make_sim_eval(algo, n_env_steps, max_steps):
    ev = PackedSimEval(init_mode="seeds", init_seeds=[0], max_steps=max_steps)
    ev.trainer = SimpleNamespace(
        lightning_module=SimpleNamespace(device=torch.device("cpu"))
    )
    ev.model = algo
    ev._env = _RecordingEnv(n_env_steps)
    return ev


def test_sim_eval_flat_kchunk_executes_open_loop():
    K = 3
    algo = _ChunkAlgo(K)
    ev = _make_sim_eval(algo, n_env_steps=6, max_steps=6)
    ev._rollout_one(None, emb_id=EMB_ID, ep_idx=0, render=False)
    # model queried only at keyframes (every K env steps)
    assert algo.calls == [0, 3]
    got = np.stack(ev._env.actions)
    expected = np.asarray(
        [[0, -0], [1, -1], [2, -2], [100, -100], [101, -101], [102, -102]],
        dtype=np.float32,
    )
    assert np.array_equal(got, expected)


def test_sim_eval_single_action_return_is_legacy_per_step():
    algo = _FixedAlgo([3.0, 4.0])
    ev = _make_sim_eval(algo, n_env_steps=5, max_steps=5)
    ev._rollout_one(None, emb_id=EMB_ID, ep_idx=0, render=False)
    assert algo.calls == list(range(5))  # re-queried every env step
    assert all(np.array_equal(a, [3.0, 4.0]) for a in ev._env.actions)


def test_sim_eval_odd_return_keeps_legacy_truncation():
    algo = _FixedAlgo([1.0, 2.0, 3.0])  # odd size: not a chunk, [:2] cut
    ev = _make_sim_eval(algo, n_env_steps=4, max_steps=4)
    ev._rollout_one(None, emb_id=EMB_ID, ep_idx=0, render=False)
    assert algo.calls == list(range(4))
    assert all(np.array_equal(a, [1.0, 2.0]) for a in ev._env.actions)


def test_sim_eval_termination_mid_chunk():
    algo = _ChunkAlgo(4)
    ev = _make_sim_eval(algo, n_env_steps=2, max_steps=10)  # env ends inside chunk
    cov, _ = ev._rollout_one(None, emb_id=EMB_ID, ep_idx=0, render=False)
    assert algo.calls == [0]
    assert len(ev._env.actions) == 2
