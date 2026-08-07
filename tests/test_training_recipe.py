"""
Unit tests for the upstream H-Net training recipe ports:

  1. ``apply_optimization_params``      — per-param ``_optim`` annotation
  2. ``HNet.init_weights``               — residual-stream-aware Linear init
                                            with ``1/sqrt(n_residuals)``
                                            scaling on ``out_proj`` / ``fc2``
  3. ``HNet.apply_lr_multiplier``        — per-stage LR scale annotated on
                                            every parameter, recursing
                                            through ``inner_stage``
  4. ``HNet.parameter_groups``           — builds AdamW-compatible
                                            ``list[dict]`` from annotations,
                                            with ``weight_decay=0`` on
                                            biases and norm weights

Each test exercises one feature end-to-end on a small but
representative 3-stage tree (EncoderDecoder → Chunker → Compute) so the
recursion + cumulative residual count is actually exercised.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from egomimic.models.hnet.hnet import HNet, apply_optimization_params
from egomimic.models.hnet.stages import (
    ChunkerStage,
    ComputeStage,
    EncoderDecoderStage,
)


def _seed(s: int = 0):
    torch.manual_seed(s)


def _build_tree(d_model: int = 32, d_inner: int = 32, with_cond: bool = False) -> HNet:
    d_cond = 16 if with_cond else 0
    return HNet(
        [
            EncoderDecoderStage(
                input_hidden_dim=d_model,
                output_hidden_dim=d_model,
                encoder=dict(
                    arch_layout="T4",
                    d_model=d_model,
                    d_intermediate=64,
                    num_heads=4,
                    cond=with_cond,
                ),
                decoder=dict(
                    arch_layout="T4",
                    d_model=d_model,
                    d_intermediate=64,
                    num_heads=4,
                    cond=with_cond,
                ),
                cond_key="fused_cond",
                d_cond=d_cond,
                causal=True,
            ),
            ChunkerStage(
                input_hidden_dim=d_model,
                output_hidden_dim=d_inner,
                target_compression_ratio=8.0,
                ratio_loss_weight=0.03,
            ),
            ComputeStage(
                input_hidden_dim=d_inner,
                output_hidden_dim=d_inner,
                main_network=dict(
                    arch_layout="T4",
                    d_model=d_inner,
                    d_intermediate=64,
                    num_heads=4,
                    cond=False,
                ),
                d_cond=0,
                causal=True,
            ),
        ]
    )


# =========================================================================== #
# apply_optimization_params
# =========================================================================== #


class TestApplyOptimizationParams:
    def test_first_call_creates_optim_dict(self):
        p = nn.Parameter(torch.zeros(4))
        assert not hasattr(p, "_optim")
        apply_optimization_params(p, lr_multiplier=3.0)
        assert p._optim == {"lr_multiplier": 3.0}

    def test_subsequent_call_merges(self):
        p = nn.Parameter(torch.zeros(4))
        apply_optimization_params(p, lr_multiplier=3.0)
        apply_optimization_params(p, weight_decay=0.0)
        assert p._optim == {"lr_multiplier": 3.0, "weight_decay": 0.0}

    def test_subsequent_call_overwrites_same_key(self):
        p = nn.Parameter(torch.zeros(4))
        apply_optimization_params(p, lr_multiplier=3.0)
        apply_optimization_params(p, lr_multiplier=1.7)
        assert p._optim == {"lr_multiplier": 1.7}


# =========================================================================== #
# init_weights — residual-stream-aware init
# =========================================================================== #


class TestInitWeights:
    """For a 3-stage tree with T4 (=height 8) at each Isotropic, the
    cumulative ``n_residuals`` should be:

      * Outer encoder/decoder Isotropic stacks    → encoder.height + decoder.height = 16
      * Inner compute Isotropic stack             → 16 + main_network.height       = 24

    out_proj (attention) and fc2 (MLP) weights should be initialized with
    ``std = initializer_range / sqrt(n_residuals)``; other Linears should
    use plain ``initializer_range``. Modules marked ``_no_reinit`` (routing
    q/k, residual_proj, AdaLN) must be skipped.
    """

    def test_height_tracking_correct(self):
        """Sanity check: every Isotropic in our T4-T4-T4 tree reports height=8."""
        _seed()
        net = _build_tree()
        assert net.stages[0].encoder.height == 8
        assert net.stages[0].decoder.height == 8
        assert net.stages[2].main_network.height == 8

    def test_out_proj_and_fc2_get_scaled_std(self):
        _seed()
        net = _build_tree()
        initializer_range = 0.02
        net.init_weights(initializer_range)

        # Outer encoder/decoder: n_residuals = 8 + 8 = 16.
        # Inner main_network: n_residuals = 16 + 8 = 24.
        n_outer = 16
        n_inner = 24
        expected_outer = initializer_range / math.sqrt(n_outer)
        expected_inner = initializer_range / math.sqrt(n_inner)

        # Outer encoder out_proj std should be near expected_outer.
        outer_enc = net.stages[0].encoder
        outer_outproj_stds = []
        outer_fc2_stds = []
        for name, m in outer_enc.named_modules():
            if not isinstance(m, nn.Linear):
                continue
            if "out_proj" in name:
                outer_outproj_stds.append(m.weight.std().item())
            elif "fc2" in name:
                outer_fc2_stds.append(m.weight.std().item())

        assert len(outer_outproj_stds) > 0, "no out_proj in encoder?"
        assert len(outer_fc2_stds) > 0, "no fc2 in encoder?"
        for s in outer_outproj_stds:
            assert (
                abs(s - expected_outer) / expected_outer < 0.25
            ), f"outer out_proj std {s} not near expected {expected_outer}"
        for s in outer_fc2_stds:
            assert (
                abs(s - expected_outer) / expected_outer < 0.25
            ), f"outer fc2 std {s} not near expected {expected_outer}"

        # Inner main_network: n_residuals = 24, smaller std.
        inner = net.stages[2].main_network
        inner_outproj_stds = []
        inner_fc2_stds = []
        for name, m in inner.named_modules():
            if not isinstance(m, nn.Linear):
                continue
            if "out_proj" in name:
                inner_outproj_stds.append(m.weight.std().item())
            elif "fc2" in name:
                inner_fc2_stds.append(m.weight.std().item())
        for s in inner_outproj_stds + inner_fc2_stds:
            assert (
                abs(s - expected_inner) / expected_inner < 0.25
            ), f"inner residual-contrib std {s} not near expected {expected_inner}"

    def test_other_linears_get_unscaled_std(self):
        """Non-residual Linears (qkv, fc1) get plain initializer_range."""
        _seed()
        net = _build_tree()
        net.init_weights(initializer_range=0.02)
        outer_enc = net.stages[0].encoder
        for name, m in outer_enc.named_modules():
            if not isinstance(m, nn.Linear):
                continue
            if "out_proj" in name or "fc2" in name:
                continue
            if getattr(m.weight, "_no_reinit", False):
                continue
            s = m.weight.std().item()
            # 0.02 ± 25%
            assert (
                abs(s - 0.02) / 0.02 < 0.25
            ), f"non-residual Linear {name} std {s} not near 0.02"

    def test_routing_qk_stays_identity(self):
        """RoutingModule's q_proj / k_proj are identity-init and marked
        ``_no_reinit`` — init_weights must leave them alone."""
        net = _build_tree()
        chunker = net.stages[1]
        # Capture before
        q_before = chunker.routing_module.q_proj_layer.weight.clone()
        k_before = chunker.routing_module.k_proj_layer.weight.clone()
        net.init_weights()
        assert torch.equal(chunker.routing_module.q_proj_layer.weight, q_before)
        assert torch.equal(chunker.routing_module.k_proj_layer.weight, k_before)
        # And they're still identity.
        d = chunker.input_hidden_dim
        assert torch.equal(chunker.routing_module.q_proj_layer.weight, torch.eye(d))
        assert torch.equal(chunker.routing_module.k_proj_layer.weight, torch.eye(d))

    def test_residual_proj_stays_zero(self):
        """ChunkerStage.residual_proj is zero-init and marked _no_reinit."""
        net = _build_tree()
        chunker = net.stages[1]
        assert torch.equal(
            chunker.residual_proj.weight, torch.zeros_like(chunker.residual_proj.weight)
        )
        net.init_weights()
        assert torch.equal(
            chunker.residual_proj.weight, torch.zeros_like(chunker.residual_proj.weight)
        )

    def test_adaln_stays_zero(self):
        """AdaLN proj is zero-init and marked _no_reinit."""
        net = _build_tree(with_cond=True)
        # Find an AdaLN module in the encoder.
        adalns = [
            m
            for n, m in net.stages[0].encoder.named_modules()
            if type(m).__name__ == "AdaLNModulation"
        ]
        assert len(adalns) > 0
        before = adalns[0].proj.weight.clone()
        net.init_weights()
        after = adalns[0].proj.weight
        assert torch.equal(before, after) and torch.equal(
            after, torch.zeros_like(after)
        )


# =========================================================================== #
# apply_lr_multiplier — per-stage scale
# =========================================================================== #


class TestApplyLrMultiplier:
    def test_each_stage_gets_its_multiplier(self):
        """For [3.0, 1.7, 0.9], every param in stage i should carry
        lr_multiplier = multipliers[i]. inner_stage params get the NEXT
        multiplier (not the outer)."""
        net = _build_tree()
        multipliers = [3.0, 1.7, 0.9]
        net.apply_lr_multiplier(multipliers)

        # Stage 0 (EncoderDecoder): encoder + decoder params.
        for p in net.stages[0].encoder.parameters():
            assert p._optim["lr_multiplier"] == 3.0
        for p in net.stages[0].decoder.parameters():
            assert p._optim["lr_multiplier"] == 3.0

        # Stage 1 (Chunker): routing_module + residual_proj + dechunk_layer +
        # (proj_in/proj_out only if dim-bridged). For d_model=d_inner=32 they
        # are None, so we just check routing + residual.
        for p in net.stages[1].routing_module.parameters():
            assert p._optim["lr_multiplier"] == 1.7
        for p in net.stages[1].residual_proj.parameters():
            assert p._optim["lr_multiplier"] == 1.7

        # Stage 2 (Compute): main_network.
        for p in net.stages[2].main_network.parameters():
            assert p._optim["lr_multiplier"] == 0.9

    def test_wrong_length_raises(self):
        net = _build_tree()
        # 3 stages in the tree but only 2 multipliers — should error before
        # the inner stage tries to dereference index 2.
        with pytest.raises(IndexError, match="2 entries.*index 2"):
            net.apply_lr_multiplier([3.0, 1.7])

    def test_can_overwrite(self):
        net = _build_tree()
        net.apply_lr_multiplier([3.0, 1.7, 0.9])
        net.apply_lr_multiplier([1.0, 1.0, 1.0])
        for p in net.stages[0].encoder.parameters():
            assert p._optim["lr_multiplier"] == 1.0


# =========================================================================== #
# parameter_groups — AdamW param groups
# =========================================================================== #


class TestParameterGroups:
    def test_default_single_group(self):
        """Without ``apply_lr_multiplier``, params either go in one group with
        the requested weight_decay, OR (for bias/norm) in a group with WD=0.
        At least one group exists."""
        net = _build_tree()
        groups = net.parameter_groups(weight_decay=0.01)
        assert len(groups) >= 1
        # All params accounted for.
        param_ids = {id(p) for g in groups for p in g["params"]}
        expected_ids = {id(p) for p in net.parameters() if p.requires_grad}
        assert param_ids == expected_ids

    def test_bias_and_norm_have_zero_weight_decay(self):
        net = _build_tree()
        groups = net.parameter_groups(weight_decay=0.01)
        # Locate every bias and every norm.weight in the net by name, then
        # confirm each ended up in a group with weight_decay=0.
        name_by_id = {id(p): n for n, p in net.named_parameters()}
        for g in groups:
            for p in g["params"]:
                name = name_by_id.get(id(p), "?")
                if (
                    name.endswith(".bias")
                    or ".norm" in name
                    or "rmsnorm" in name.lower()
                ):
                    assert (
                        g["weight_decay"] == 0.0
                    ), f"{name} ended up in group with WD={g['weight_decay']}"

    def test_with_lr_multiplier_groups_split_per_stage(self):
        """With per-stage LR multipliers, params split into per-multiplier
        groups (cross-cut by weight_decay). At minimum 3 distinct
        lr_multiplier values appear in the output."""
        net = _build_tree()
        net.apply_lr_multiplier([3.0, 1.7, 0.9])
        groups = net.parameter_groups(weight_decay=0.01)
        multipliers = sorted({g["lr_multiplier"] for g in groups})
        assert multipliers == [0.9, 1.7, 3.0], multipliers

    def test_all_params_in_exactly_one_group(self):
        """No double-counting; no missing params."""
        net = _build_tree()
        net.apply_lr_multiplier([3.0, 1.7, 0.9])
        groups = net.parameter_groups(weight_decay=0.01)
        seen = []
        for g in groups:
            for p in g["params"]:
                seen.append(id(p))
        assert len(seen) == len(set(seen)), "param appears in multiple groups"
        expected = {id(p) for p in net.parameters() if p.requires_grad}
        assert set(seen) == expected, "missing or extra params"

    def test_groups_consumable_by_adamw(self):
        """End-to-end: AdamW should accept the returned groups and run a step."""
        net = _build_tree()
        net.init_weights()
        net.apply_lr_multiplier([3.0, 1.7, 0.9])
        groups = net.parameter_groups(weight_decay=0.01)
        # AdamW expects each group to have ``params`` and ``lr``. The
        # ``lr_multiplier`` annotation is informational only — the user is
        # expected to multiply it into ``lr`` themselves. Set lr per group
        # here as a smoke.
        for g in groups:
            g["lr"] = 1e-4 * g["lr_multiplier"]
        opt = torch.optim.AdamW(groups)
        # Forward/backward through the net to populate grads.
        x = torch.randn(1, 4, 32)
        from egomimic.models.hnet.context import HNetContext

        y = net(x, HNetContext())
        y.sum().backward()
        opt.step()  # should not raise


# =========================================================================== #
# Flexibility — recipe must work for arbitrary stage compositions
# =========================================================================== #


class TestRecipeFitsFlexibleConfigs:
    """The new EgoVerse stage tree is a flat list of stages with arbitrary
    composition (single ComputeStage, ChunkerStage→ChunkerStage→...,
    EncoderDecoderStage→ComputeStage, etc.). The training recipe should
    work for any of these.
    """

    def _build_single_compute(self) -> HNet:
        return HNet(
            [
                ComputeStage(
                    input_hidden_dim=32,
                    output_hidden_dim=32,
                    main_network=dict(
                        arch_layout="T4",
                        d_model=32,
                        d_intermediate=64,
                        num_heads=4,
                        cond=False,
                    ),
                    d_cond=0,
                    causal=True,
                ),
            ]
        )

    def _build_compute_compute(self) -> HNet:
        return HNet(
            [
                ComputeStage(
                    input_hidden_dim=32,
                    output_hidden_dim=32,
                    main_network=dict(
                        arch_layout="T2",
                        d_model=32,
                        d_intermediate=64,
                        num_heads=4,
                        cond=False,
                    ),
                    d_cond=0,
                    causal=True,
                ),
                ComputeStage(
                    input_hidden_dim=32,
                    output_hidden_dim=32,
                    main_network=dict(
                        arch_layout="T4",
                        d_model=32,
                        d_intermediate=64,
                        num_heads=4,
                        cond=False,
                    ),
                    d_cond=0,
                    causal=True,
                ),
            ]
        )

    def _build_enc_compute(self) -> HNet:
        return HNet(
            [
                EncoderDecoderStage(
                    input_hidden_dim=32,
                    output_hidden_dim=32,
                    encoder=dict(
                        arch_layout="T2",
                        d_model=32,
                        d_intermediate=64,
                        num_heads=4,
                        cond=False,
                    ),
                    decoder=dict(
                        arch_layout="T2",
                        d_model=32,
                        d_intermediate=64,
                        num_heads=4,
                        cond=False,
                    ),
                    d_cond=0,
                    causal=True,
                ),
                ComputeStage(
                    input_hidden_dim=32,
                    output_hidden_dim=32,
                    main_network=dict(
                        arch_layout="T4",
                        d_model=32,
                        d_intermediate=64,
                        num_heads=4,
                        cond=False,
                    ),
                    d_cond=0,
                    causal=True,
                ),
            ]
        )

    def test_single_stage_init_and_lr(self):
        net = self._build_single_compute()
        net.init_weights()
        # n_residuals = 8 (T4); all out_proj/fc2 should have std = 0.02/sqrt(8).
        expected = 0.02 / math.sqrt(8)
        for name, m in net.stages[0].main_network.named_modules():
            if (
                isinstance(m, nn.Linear)
                and ("out_proj" in name or "fc2" in name)
                and not getattr(m.weight, "_no_reinit", False)
            ):
                assert abs(m.weight.std().item() - expected) / expected < 0.3
        net.apply_lr_multiplier([0.5])
        for p in net.parameters():
            assert p._optim["lr_multiplier"] == 0.5

    def test_two_compute_stages_lr(self):
        net = self._build_compute_compute()
        net.apply_lr_multiplier([2.0, 0.5])
        # Stage 0 params get 2.0
        for p in net.stages[0].main_network.parameters():
            assert p._optim["lr_multiplier"] == 2.0
        # Stage 1 (inner) params get 0.5
        for p in net.stages[1].main_network.parameters():
            assert p._optim["lr_multiplier"] == 0.5
        # parameter_groups has 2 multipliers.
        groups = net.parameter_groups(weight_decay=0.01)
        muls = sorted({g["lr_multiplier"] for g in groups})
        assert muls == [0.5, 2.0]

    def test_enc_compute_height_cumulative(self):
        """Encoder-Decoder (T2 each = height 4 each) + ComputeStage (T4 = 8).
        Outer enc/dec: n_residuals = 4 + 4 = 8.
        Inner compute: n_residuals = 8 + 8 = 16.
        """
        net = self._build_enc_compute()
        net.init_weights()

        exp_outer = 0.02 / math.sqrt(8)
        exp_inner = 0.02 / math.sqrt(16)
        # Outer encoder check.
        for name, m in net.stages[0].encoder.named_modules():
            if (
                isinstance(m, nn.Linear)
                and ("out_proj" in name or "fc2" in name)
                and not getattr(m.weight, "_no_reinit", False)
            ):
                assert abs(m.weight.std().item() - exp_outer) / exp_outer < 0.3
        # Inner compute check.
        for name, m in net.stages[1].main_network.named_modules():
            if (
                isinstance(m, nn.Linear)
                and ("out_proj" in name or "fc2" in name)
                and not getattr(m.weight, "_no_reinit", False)
            ):
                assert abs(m.weight.std().item() - exp_inner) / exp_inner < 0.3


# =========================================================================== #
# Algo wiring — opt-in via PackedAlgoBase.__init__ kwargs, disabled by default
# =========================================================================== #


class TestAlgoWiring:
    """The PackedAlgoBase algo accepts ``init_weights_range`` / ``lr_multipliers`` /
    ``use_parameter_groups`` / ``weight_decay`` as constructor kwargs.
    Defaults must reproduce "standard training": no init override, no LR
    multipliers stamped on params, no parameter_groups returned to the
    Lightning optimizer hook.
    """

    def _build_algo(
        self,
        norm_stats,
        init_weights_range=None,
        lr_multipliers=None,
        use_parameter_groups=False,
        weight_decay=0.0,
    ):
        from egomimic.algo.hnet import HNetOuterStage, PackedAlgoBase as HNetAlgo
        from egomimic.models.stems.cond_encoders import CondEncoderModule

        outer_stage = HNetOuterStage(
            action_dim=2,
            action_horizon=64,
            d_model=32,
            cond_encoder=CondEncoderModule(d_cond=0),
            hnet=_build_tree(),
        )
        return HNetAlgo(
            outer_stage=outer_stage,
            norm_stats=norm_stats,
            d_cond=0,
            domains=["pushshapes_sim"],
            ac_keys={"pushshapes_sim": "actions"},
            device=torch.device("cpu"),
            init_weights_range=init_weights_range,
            lr_multipliers=lr_multipliers,
            use_parameter_groups=use_parameter_groups,
            weight_decay=weight_decay,
        )

    def _fake_norm_stats(self):
        """Minimal MultiDataset stats holder with pushshapes_sim emb=15."""
        from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

        ms = MultiDataset(state={}, norm_mode="zscore")
        emb_id = 15
        ms.embodiments.add(emb_id)
        ms.key_types[emb_id] = {"actions": "action_keys"}
        ms.zarr_keys[emb_id] = {"actions": "actions"}
        ms.norm_stats[emb_id] = {}
        return ms

    def test_defaults_off_no_lr_stamp(self):
        """With defaults, params should NOT have ``_optim`` stamped."""
        ns = self._fake_norm_stats()
        algo = self._build_algo(ns)
        for p in algo.nets.parameters():
            assert not hasattr(
                p, "_optim"
            ), "params should not be annotated when lr_multipliers=None"

    def test_defaults_off_parameter_groups_returns_none(self):
        ns = self._fake_norm_stats()
        algo = self._build_algo(ns)
        assert algo.parameter_groups(base_lr=1e-4) is None

    def test_init_weights_range_applies_scaled_std(self):
        """With ``init_weights_range=0.02``, the inner ComputeStage's
        out_proj/fc2 should be near 0.02/sqrt(n_residuals) — same check as
        TestInitWeights, but applied via the algo constructor instead of
        ``hnet.init_weights()`` directly.
        """
        ns = self._fake_norm_stats()
        algo = self._build_algo(ns, init_weights_range=0.02)
        # The HNet tree built by _build_tree has T4 enc/dec/main → n_inner=24.
        expected_inner = 0.02 / math.sqrt(24)
        inner = algo._hnet_core.stages[2].main_network
        stds = []
        for name, m in inner.named_modules():
            if (
                isinstance(m, nn.Linear)
                and ("out_proj" in name or "fc2" in name)
                and not getattr(m.weight, "_no_reinit", False)
            ):
                stds.append(m.weight.std().item())
        assert len(stds) > 0
        for s in stds:
            assert abs(s - expected_inner) / expected_inner < 0.3

    def test_lr_multipliers_stamps_params(self):
        ns = self._fake_norm_stats()
        algo = self._build_algo(ns, lr_multipliers=[3.0, 1.7, 0.9])
        # Every HNet stage param should now carry ``lr_multiplier``.
        for p in algo._hnet_core.parameters():
            assert hasattr(p, "_optim")
            assert "lr_multiplier" in p._optim

    def test_use_parameter_groups_returns_adamw_ready_groups(self):
        ns = self._fake_norm_stats()
        algo = self._build_algo(
            ns,
            init_weights_range=0.02,
            lr_multipliers=[3.0, 1.7, 0.9],
            use_parameter_groups=True,
            weight_decay=0.01,
        )
        base_lr = 1e-4
        groups = algo.parameter_groups(base_lr=base_lr)
        assert groups is not None
        # Each group must have ``params``, ``lr``, ``weight_decay``.
        for g in groups:
            assert "params" in g
            assert "lr" in g
            assert "weight_decay" in g
        # lrs should reflect base_lr * lr_multiplier for at least one group.
        lrs = sorted({g["lr"] for g in groups})
        # Expect {1e-4, 0.9e-4, 1.7e-4, 3.0e-4} at minimum (the extras-bias
        # group gets multiplier=1.0).
        assert any(abs(lr - 3.0 * base_lr) < 1e-12 for lr in lrs), lrs
        assert any(abs(lr - 0.9 * base_lr) < 1e-12 for lr in lrs), lrs

    def test_use_parameter_groups_covers_every_param_exactly_once(self):
        """No double-counting between the HNet-tree groups and the extras
        groups; nothing missing."""
        ns = self._fake_norm_stats()
        algo = self._build_algo(
            ns,
            init_weights_range=0.02,
            lr_multipliers=[3.0, 1.7, 0.9],
            use_parameter_groups=True,
            weight_decay=0.01,
        )
        groups = algo.parameter_groups(base_lr=1e-4)
        seen_ids = []
        for g in groups:
            for p in g["params"]:
                seen_ids.append(id(p))
        assert len(seen_ids) == len(set(seen_ids)), "param double-counted"
        expected = {id(p) for p in algo.nets.parameters() if p.requires_grad}
        assert set(seen_ids) == expected, "missing or extra params"

    def test_extras_group_has_action_in_action_out_bos_pos_emb(self):
        """The policy-level params (action_in, action_out, bos, pos_emb,
        cond_encoder) live outside the HNet stage tree and must land in an
        extras group with lr_multiplier=1.0."""
        ns = self._fake_norm_stats()
        algo = self._build_algo(
            ns,
            lr_multipliers=[3.0, 1.7, 0.9],
            use_parameter_groups=True,
            weight_decay=0.01,
        )
        groups = algo.parameter_groups(base_lr=1e-4)
        hnet_ids = {id(p) for p in algo._hnet_core.parameters()}
        for g in groups:
            for p in g["params"]:
                if id(p) not in hnet_ids:
                    # This is a policy-level param. Its group must have
                    # lr_multiplier == 1.0.
                    assert g["lr_multiplier"] == 1.0
                    assert g["lr"] == 1e-4


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
