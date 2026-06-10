"""Parity-port eval additions (D06/D19): overlay action-decode math + sim knobs.

Pure-torch CPU tests, no data, no env, no GPU:

  * ``decode_action_planes`` — masked spatial-mean decode recovers constant
    action planes exactly, on BOTH (T,C,H,W) and (B,T,C,H,W) bundles, and
    provably slices the CHANNEL axis (the D02 failure mode sliced image WIDTH).
  * ``actions_to_planes`` -> ``decode_action_planes`` round-trip mirrors the
    outer stage's broadcast encoding.
  * ``overlay_metrics`` — world-px MSE/RMSE + pred-std collapse check vs a
    numpy reference; the full decode -> minmax-unnormalize -> metrics chain on
    a synthetic batch.
  * ``SimRolloutEval`` knobs — defaults reproduce legacy behavior
    (coverage_agg="final", n_eval_episodes=None => only rendered episodes
    rolled out); "best" aggregation and the explicit episode budget work.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from egomimic.eval.dfot.eval_dfot_pixel_overlay import (
    DFoTPixelPolicyOverlayEval,
    actions_to_planes,
    decode_action_planes,
    overlay_metrics,
)
from egomimic.eval.core.eval_sim import PackedSimEval


# --------------------------------------------------------------------- #
# decode math
# --------------------------------------------------------------------- #


def test_decode_recovers_constant_planes_4d_and_5d():
    Ci, Ca, A, H, W = 3, 2, 2, 8, 8
    vals = torch.tensor([[0.25, -0.5], [1.2, 0.0], [-0.75, 0.3]])  # (T=3, A)
    bundle = torch.full((3, Ci + Ca, H, W), 7.0)  # image channels = 7 everywhere
    bundle[:, Ci:] = vals.reshape(3, Ca, 1, 1).expand(3, Ca, H, W)
    out = decode_action_planes(bundle, Ci, Ca, A)
    assert out.shape == (3, A)
    assert torch.allclose(out, vals)

    bundle5 = bundle.unsqueeze(0).expand(2, 3, Ci + Ca, H, W)
    out5 = decode_action_planes(bundle5, Ci, Ca, A)
    assert out5.shape == (2, 3, A)
    assert torch.allclose(out5[0], vals) and torch.allclose(out5[1], vals)


def test_decode_slices_channel_axis_not_width():
    # D02 regression guard: ``t[..., 3:5]`` on (T,5,H,W) slices image WIDTH
    # columns, not channels. With image channels at 7.0 and W>=5, a wrong-axis
    # slice can only produce 7.0-contaminated values; the decode must return
    # the pure plane constants.
    Ci, Ca, A, H, W = 3, 2, 2, 6, 6
    bundle = torch.full((4, Ci + Ca, H, W), 7.0)
    bundle[:, Ci] = 0.111
    bundle[:, Ci + 1] = -0.222
    out = decode_action_planes(bundle, Ci, Ca, A)
    assert torch.allclose(out[:, 0], torch.full((4,), 0.111))
    assert torch.allclose(out[:, 1], torch.full((4,), -0.222))
    # the buggy width-slice would not even have the decoded shape
    wrong = bundle[..., Ci : Ci + Ca].mean(dim=(-2, -1))
    assert wrong.shape != out.shape


def test_actions_to_planes_roundtrip_mirrors_outer_stage():
    torch.manual_seed(0)
    Ci, Ca, A, H, W = 3, 2, 2, 5, 9
    actions = torch.randn(7, A)
    planes = actions_to_planes(actions, Ca, H, W)
    assert planes.shape == (7, Ca, H, W)
    bundle = torch.cat([torch.zeros(7, Ci, H, W), planes], dim=1)
    assert torch.allclose(decode_action_planes(bundle, Ci, Ca, A), actions, atol=1e-6)


# --------------------------------------------------------------------- #
# overlay metrics
# --------------------------------------------------------------------- #


def test_overlay_metrics_matches_numpy_reference():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 512, size=(40, 2))
    gt = rng.uniform(0, 512, size=(40, 2))
    m = overlay_metrics(pred, gt)
    ref_sq = ((pred - gt) ** 2).mean(axis=-1)
    assert np.allclose(m["per_step_sq_err"], ref_sq)
    assert np.isclose(float(m["mse_px"]), ref_sq.mean())
    assert np.isclose(float(m["rmse_px"]), np.sqrt(ref_sq.mean()))
    assert np.isclose(float(m["pred_std_px"]), pred.std(axis=0).mean())
    assert np.isclose(float(m["gt_std_px"]), gt.std(axis=0).mean())


def test_overlay_metrics_collapse_check():
    # constant prediction => pred_std exactly 0 (the collapse signature)
    pred = np.full((25, 2), 256.0)
    gt = np.linspace(0, 512, 50).reshape(25, 2)
    m = overlay_metrics(pred, gt)
    assert float(m["pred_std_px"]) == 0.0
    assert float(m["gt_std_px"]) > 0.0


def test_overlay_metrics_shape_mismatch_raises():
    with pytest.raises(ValueError):
        overlay_metrics(np.zeros((3, 2)), np.zeros((4, 2)))


class _MinMaxNormStats:
    """Stub of MultiDataset's minmax unnormalize: (t+1)*0.5*(mx-mn+1e-6)+mn."""

    def __init__(self, mn, mx):
        self.mn = torch.as_tensor(mn, dtype=torch.float32)
        self.mx = torch.as_tensor(mx, dtype=torch.float32)

    def unnormalize(self, data: dict, emb_id: int) -> dict:
        return {
            k: (v + 1) * 0.5 * (self.mx - self.mn + 1e-6) + self.mn
            for k, v in data.items()
        }


def test_decode_to_world_px_chain_on_synthetic_batch():
    # full chain on a synthetic bundle: planes -> decode (normalized) ->
    # minmax unnormalize -> world px -> RMSE vs GT.
    Ci, Ca, A, H, W = 3, 2, 2, 8, 8
    norm_stats = _MinMaxNormStats(mn=[0.0, 0.0], mx=[512.0, 512.0])
    gt_norm = torch.tensor([[-0.5, 0.0], [0.0, 0.5], [0.25, -0.25]])
    err_norm = 64.0 / (0.5 * (512.0 + 1e-6))  # 64 world px in normalized units
    pred_norm = gt_norm + torch.tensor([err_norm, 0.0])

    bundle = torch.zeros(3, Ci + Ca, H, W)
    bundle[:, Ci:] = pred_norm.reshape(3, Ca, 1, 1).expand(3, Ca, H, W)
    decoded = decode_action_planes(bundle, Ci, Ca, A)
    pred_world = norm_stats.unnormalize({"actions": decoded}, 0)["actions"].numpy()
    gt_world = norm_stats.unnormalize({"actions": gt_norm}, 0)["actions"].numpy()

    m = overlay_metrics(pred_world, gt_world)
    # error vector is (64, 0) px at every step -> per-dim-mean sq = 64^2/2
    assert np.allclose(m["per_step_sq_err"], 64.0**2 / 2.0, rtol=1e-4)
    assert np.isclose(float(m["rmse_px"]), 64.0 / np.sqrt(2.0), rtol=1e-4)


def test_overlay_eval_ctor_validates_action_history():
    ev = DFoTPixelPolicyOverlayEval()
    assert ev.action_history == "pred"
    assert ev.n_context_frames == 4 and ev.commit_k == 1
    with pytest.raises(ValueError):
        DFoTPixelPolicyOverlayEval(action_history="bogus")


# --------------------------------------------------------------------- #
# sim eval knobs (coverage_agg / n_eval_episodes)
# --------------------------------------------------------------------- #


def _cpu_trainer_stub():
    return SimpleNamespace(
        lightning_module=SimpleNamespace(device=torch.device("cpu"))
    )


class _ScriptedEnv:
    """Env stub emitting a fixed coverage sequence; terminates at the end."""

    def __init__(self, coverages):
        self.coverages = list(coverages)
        self.t = 0

    def reset(self, seed=None):
        self.t = 0

    def _get_obs(self):
        return {
            "agent_pos": np.zeros(2, dtype=np.float32),
            "object_pose": np.zeros(3, dtype=np.float32),
            "image": np.zeros((8, 8, 3), dtype=np.uint8),
        }

    def step(self, action):
        cov = self.coverages[self.t]
        self.t += 1
        terminated = self.t >= len(self.coverages)
        return None, None, terminated, None, {"coverage": cov}

    def render(self):
        return None


class _ZeroAlgo:
    def inference_step(self, obs_zarr, t, emb_id, T_max=None):
        return np.zeros(2, dtype=np.float32)


def _make_sim_eval(**kwargs):
    ev = PackedSimEval(
        init_mode="seeds", init_seeds=[0], max_steps=10, **kwargs
    )
    ev.trainer = _cpu_trainer_stub()
    ev.model = _ZeroAlgo()
    return ev


def test_sim_eval_defaults_are_legacy():
    ev = _make_sim_eval()
    assert ev.coverage_agg == "final"
    assert ev.n_eval_episodes is None
    assert ev.coverage_threshold == 0.7
    with pytest.raises(ValueError):
        _make_sim_eval(coverage_agg="median")


@pytest.mark.parametrize(
    "agg,expected", [("final", 0.3), ("best", 0.6)]
)
def test_sim_eval_coverage_agg(agg, expected):
    ev = _make_sim_eval(coverage_agg=agg)
    ev._env = _ScriptedEnv([0.1, 0.6, 0.3])
    cov, _ = ev._rollout_one(None, emb_id=0, ep_idx=0)
    assert cov == pytest.approx(expected)


def _seq_batch(n_eps=3):
    return {0: {"seq_lens": torch.full((n_eps,), 5, dtype=torch.long)}}


def test_sim_eval_legacy_rollout_count_is_render_capped(monkeypatch):
    # n_eval_episodes=None => only min(B, max_videos) episodes rolled out
    ev = _make_sim_eval(max_videos=2)
    calls = []
    monkeypatch.setattr(
        ev, "_rollout_one",
        lambda env_init, emb_id, ep_idx, render=True: (calls.append(ep_idx) or (0.4, [])),
    )
    metrics, _ = ev.compute_metrics_and_viz(_seq_batch(n_eps=3))
    assert calls == [0, 1]
    assert metrics["Valid/emb0_sim_coverage"].item() == pytest.approx(0.4)


def test_sim_eval_n_eval_episodes_overrides_count(monkeypatch):
    # explicit budget: rollouts = n_eval_episodes (seeds mode ignores the
    # batch's episode count); rendering would still cap at max_videos
    ev = _make_sim_eval(max_videos=2, n_eval_episodes=7, coverage_threshold=0.5)
    calls = []
    monkeypatch.setattr(
        ev, "_rollout_one",
        lambda env_init, emb_id, ep_idx, render=True: (
            calls.append(ep_idx) or (0.6 if ep_idx % 2 == 0 else 0.2, [])
        ),
    )
    metrics, _ = ev.compute_metrics_and_viz(_seq_batch(n_eps=3))
    assert calls == list(range(7))
    # 4 of 7 episodes hit >= 0.5
    assert metrics["Valid/emb0_sim_success_rate"].item() == pytest.approx(4 / 7)
