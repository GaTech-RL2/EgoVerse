"""TEI and post-block TLI using the frozen released RoboCasa JAX parameters.

Source latents are recomputed with the current real cameras/state. This is the
online-subgoal variant, not a bank extracted from training demonstrations.
The native action expert, normalization, noise and Euler sampler are retained.
"""

import time

import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma
from openpi.shared import nnx_utils

from astra_reversal.complex_manipulation.text_slots import (
    alignment_indices,
    instruction_slots,
)
from astra_reversal.records import digest


class RecordingBlock(gemma.Block):
    """Same parameter scope as upstream Block, with explicit post-block writes."""

    @nn.compact
    def __call__(self, xs, kv_cache, positions, mask, delta, edit_mask):
        xs, kv_cache = super().__call__(
            xs, kv_cache, positions, mask, [None] * len(xs), True
        )
        before = xs[0][:, -delta.shape[1] :]
        after = jnp.where(
            edit_mask[..., None],
            (before.astype(jnp.float32) + delta).astype(before.dtype),
            before,
        )
        xs = [xs[0].at[:, -delta.shape[1] :].set(after), *xs[1:]]
        difference = after.astype(jnp.float32) - before.astype(jnp.float32)
        metrics = jnp.stack([jnp.linalg.norm(difference), jnp.max(jnp.abs(difference))])
        return xs, (kv_cache, before, metrics)


class RecordingPrefix(nn.Module):
    configs: tuple
    embed_dtype: str

    @nn.compact
    def __call__(self, embedded, positions, mask, deltas, edit_mask):
        block = nn.scan(
            RecordingBlock,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, nn.broadcast, nn.broadcast, 0, nn.broadcast),
            length=self.configs[0].depth,
        )(configs=self.configs, name="layers")
        _, (cache, hidden, metrics) = block(
            [embedded.astype(self.embed_dtype), None],
            None,
            positions,
            mask[:, None],
            deltas,
            edit_mask,
        )
        return cache, hidden, metrics


def aligned(values, indices, valid):
    return jnp.where(valid[None, :, None], values[:, indices], 0)


class Interpolator(nnx.Module):
    def __init__(self, native):
        self.native = native

    def _record_prefix(self, tokens, mask, ar_mask, deltas, instruction_mask):
        from openpi.models import pi0

        llm = self.native.PaliGemma.llm
        # ToNNX stores the original Linen parameter tree without name changes.
        params = nnx.state(llm, nnx.Param).to_pure_dict()
        recorder = RecordingPrefix(
            configs=tuple(llm.module.configs),
            embed_dtype=llm.module.embed_dtype,
        )
        return recorder.apply(
            {"params": params},
            tokens,
            jnp.cumsum(mask, axis=1) - 1,
            pi0.make_attn_mask(mask, ar_mask),
            deltas,
            instruction_mask,
        )

    def prefix_probe(self, observation, instruction_mask):
        from openpi.models import model as model_lib, pi0

        observation = model_lib.preprocess_observation(None, observation, train=False)
        tokens, mask, ar_mask = self.native.embed_prefix(observation)
        depth = self.native.PaliGemma.llm.module.configs[0].depth
        zeros = jnp.zeros((depth, 1, 200, tokens.shape[-1]), dtype=jnp.float32)
        recorded, _, _ = self._record_prefix(
            tokens, mask, ar_mask, zeros, instruction_mask
        )
        _, original = self.native.PaliGemma.llm(
            [tokens, None],
            mask=pi0.make_attn_mask(mask, ar_mask),
            positions=jnp.cumsum(mask, axis=1) - 1,
        )
        return jnp.stack(
            [
                jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))
                for a, b in zip(recorded, original, strict=True)
            ]
        )

    def sample(
        self,
        rng,
        observation,
        source_observation,
        instruction_mask,
        source_indices,
        source_valid,
        alpha,
        *,
        operator,
        noise=None,
    ):
        from openpi.models import model as model_lib, pi0

        native = self.native
        observation = model_lib.preprocess_observation(None, observation, train=False)
        source_observation = model_lib.preprocess_observation(
            None, source_observation, train=False
        )
        tokens, prefix_mask, ar_mask = native.embed_prefix(observation)
        target_text = tokens[:, -200:]
        metrics = jnp.zeros((18, 2), jnp.float32)
        if operator == "native_copy":
            # Diagnostic only: isolate copied sampler numerics from text edits.
            _, cache = native.PaliGemma.llm(
                [tokens, None],
                mask=pi0.make_attn_mask(prefix_mask, ar_mask),
                positions=jnp.cumsum(prefix_mask, axis=1) - 1,
            )
        elif operator == "tei":
            source_text = native.PaliGemma.llm(
                source_observation.tokenized_prompt, method="embed"
            )
            source_text = aligned(source_text, source_indices, source_valid)
            mixed = (
                (1 - alpha) * target_text.astype(jnp.float32)
                + alpha * source_text.astype(jnp.float32)
            ).astype(target_text.dtype)
            edited = jnp.where(instruction_mask[..., None], mixed, target_text)
            delta = edited.astype(jnp.float32) - target_text.astype(jnp.float32)
            metrics = metrics.at[0].set(
                jnp.stack([jnp.linalg.norm(delta), jnp.max(jnp.abs(delta))])
            )
            tokens = tokens.at[:, -200:].set(edited)
            _, cache = native.PaliGemma.llm(
                [tokens, None],
                mask=pi0.make_attn_mask(prefix_mask, ar_mask),
                positions=jnp.cumsum(prefix_mask, axis=1) - 1,
            )
        elif operator == "tli":
            depth = native.PaliGemma.llm.module.configs[0].depth
            zeros = jnp.zeros((depth, 1, 200, tokens.shape[-1]), jnp.float32)
            _, target_hidden, _ = self._record_prefix(
                tokens, prefix_mask, ar_mask, zeros, instruction_mask
            )
            source_tokens, source_mask, source_ar = native.embed_prefix(
                source_observation
            )
            _, source_hidden, _ = self._record_prefix(
                source_tokens, source_mask, source_ar, zeros, instruction_mask
            )
            source_hidden = jax.vmap(aligned, in_axes=(0, None, None))(
                source_hidden, source_indices, source_valid
            )
            # T_A = subgoal, T_B = target; alpha=.5 is the native identity.
            deltas = (1 - 2 * alpha) * (
                source_hidden.astype(jnp.float32) - target_hidden.astype(jnp.float32)
            )
            # The final block already made its K/V: its output cannot steer actions.
            deltas = deltas.at[-1].set(0)
            cache, _, metrics = self._record_prefix(
                tokens, prefix_mask, ar_mask, deltas, instruction_mask
            )
        else:
            raise ValueError("Only TEI and TLI are registered")

        batch = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(
                rng, (batch, native.action_horizon, native.action_dim)
            )
        dt = -1.0 / 10

        def step(carry):
            actions, t = carry
            suffix, suffix_mask, suffix_ar, cond = native.embed_suffix(
                observation, actions, jnp.broadcast_to(t, batch)
            )
            full_mask = jnp.concatenate(
                [
                    jnp.broadcast_to(
                        prefix_mask[:, None],
                        (batch, suffix.shape[1], prefix_mask.shape[1]),
                    ),
                    pi0.make_attn_mask(suffix_mask, suffix_ar),
                ],
                axis=-1,
            )
            positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )
            (_, output), _ = native.PaliGemma.llm(
                [None, suffix],
                mask=full_mask,
                positions=positions,
                kv_cache=cache,
                adarms_cond=[None, cond],
            )
            velocity = native.action_out_proj(output[:, -native.action_horizon :])
            return actions + dt * velocity, t + dt

        result, _ = jax.lax.while_loop(
            lambda carry: carry[1] >= -dt / 2, step, (noise, 1.0)
        )
        return result, metrics


class TextInterpolationPolicy:
    def __init__(self, native):
        from openpi.models.tokenizer import PaligemmaTokenizer

        self.native = native
        self.tokenizer = PaligemmaTokenizer(200)._tokenizer
        self.interpolator = Interpolator(native._model)
        self._sample = nnx_utils.module_jit(
            self.interpolator.sample, static_argnames=("operator",)
        )
        self._prefix_probe = nnx_utils.module_jit(self.interpolator.prefix_probe)

    @property
    def _rng(self):
        return self.native._rng

    @_rng.setter
    def _rng(self, value):
        self.native._rng = value

    def _prepare(self, observation):
        from openpi.models import model as model_lib

        inputs = self.native._input_transform(jax.tree.map(lambda x: x, observation))
        slots = instruction_slots(
            self.tokenizer,
            observation["prompt"],
            np.asarray(inputs["state"]),
            inputs["tokenized_prompt"],
            inputs["tokenized_prompt_mask"],
        )
        batched = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
        return model_lib.Observation.from_dict(batched), slots

    def infer(self, observation, *, noise=None, intervention=None):
        if intervention is None or intervention["method"] == "native":
            return self.native.infer(observation, noise=noise)
        method, alpha = intervention["method"], intervention["alpha"]
        if method not in ("tei", "tli") or alpha not in (0, 0.25, 0.5, 0.75, 1):
            raise ValueError("Intervention is outside the registered choices")
        started = time.perf_counter()
        target, mask = self._prepare(observation)
        source, source_mask = self._prepare(
            {**observation, "prompt": intervention["subgoal"]}
        )
        indices, valid = alignment_indices(source_mask, mask)
        self.native._rng, sample_rng = jax.random.split(self.native._rng)
        if noise is not None:
            noise = jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None]
        actions, metrics = self._sample(
            sample_rng,
            target,
            source,
            jnp.asarray(mask)[None],
            jnp.asarray(indices),
            jnp.asarray(valid),
            jnp.asarray(alpha, jnp.float32),
            operator=method,
            noise=noise,
        )
        actions, metrics = np.asarray(actions[0]), np.asarray(metrics)
        if not np.isfinite(actions).all() or not np.isfinite(metrics).all():
            raise ValueError("Non-finite intervention output")
        result = self.native._output_transform(
            {"state": np.asarray(target.state[0]), "actions": actions}
        )
        result["interpolation"] = {
            "method": method,
            "alpha": alpha,
            "subgoal": intervention["subgoal"],
            "source": "same_current_real_cameras_and_state_online_subgoal",
            "target_instruction_positions": np.flatnonzero(mask).tolist(),
            "source_instruction_positions": np.flatnonzero(source_mask).tolist(),
            "target_tokens_sha256": digest(np.asarray(target.tokenized_prompt)),
            "source_tokens_sha256": digest(np.asarray(source.tokenized_prompt)),
            "layer_delta_frobenius_max_abs": metrics.tolist(),
            "has_effect": bool(np.any(metrics[:, 0] != 0)),
            "protected_slots_direct_writes": 0,
            "seconds": time.perf_counter() - started,
        }
        return result

    def preflight(self, observation, *, publish=None):
        """No environment steps: compare native sampling and inspect real cache edits."""
        original_key = self._rng
        receipt = {
            "status": "running",
            "operators": {},
            "failed_checks": [],
            "environment_resets": 0,
            "environment_actions": 0,
            "policy_updates": 0,
        }

        def save():
            if publish is not None:
                publish(receipt)

        save()
        try:
            target, mask = self._prepare(observation)
            noise = (
                np.random.default_rng(731).standard_normal((50, 32)).astype(np.float32)
            )
            native = self.native.infer(observation, noise=noise)["actions"]
            cache_error = np.asarray(
                self._prefix_probe(target, jnp.asarray(mask)[None])
            )
            receipt["native_cache_max_abs"] = cache_error.tolist()
            if not np.isfinite(cache_error).all() or np.max(cache_error) > 1e-5:
                receipt["failed_checks"].append("native_cache")
            save()
            # A third path separates Euler/suffix numerical drift from any edit.
            copied, _ = self._sample(
                jax.random.key(731),
                target,
                target,
                jnp.asarray(mask)[None],
                jnp.arange(200),
                jnp.asarray(mask),
                jnp.asarray(0, jnp.float32),
                operator="native_copy",
                noise=jnp.asarray(noise)[None],
            )
            copied = self.native._output_transform(
                {
                    "state": np.asarray(target.state[0]),
                    "actions": np.asarray(copied[0]),
                }
            )["actions"]
            receipt["native_copy_max_action_difference"] = float(
                np.max(np.abs(copied - native))
            )
            save()
            subgoal = "Lift the held object higher while keeping the gripper closed."
            for method, alpha in (
                ("tei", 0.0),
                ("tli", 0.5),
                ("tei", 0.5),
                ("tli", 0.25),
            ):
                result = self.infer(
                    observation,
                    noise=noise,
                    intervention={
                        "method": method,
                        "alpha": alpha,
                        "subgoal": subgoal,
                    },
                )
                error = float(np.max(np.abs(result["actions"] - native)))
                neutral = (method, alpha) in (("tei", 0.0), ("tli", 0.5))
                name = f"{method}_{alpha}"
                receipt["operators"][name] = {
                    "max_action_difference": error,
                    "max_action_difference_from_native_copy": float(
                        np.max(np.abs(result["actions"] - copied))
                    ),
                    **result["interpolation"],
                }
                if (neutral and error > 1e-5) or (not neutral and error <= 1e-6):
                    receipt["failed_checks"].append(name)
                save()
            after = self.native.infer(observation, noise=noise)["actions"]
            receipt["native_restored_exact"] = bool(np.array_equal(native, after))
            if not receipt["native_restored_exact"]:
                receipt["failed_checks"].append("native_restoration")
            receipt["status"] = "failed" if receipt["failed_checks"] else "passed"
            save()
            if receipt["failed_checks"]:
                raise RuntimeError(
                    "Preflight failed: " + ", ".join(receipt["failed_checks"])
                )
            return receipt
        except BaseException as exc:
            receipt["status"] = "failed"
            receipt["exception"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:500],
            }
            save()
            raise
        finally:
            self._rng = original_key
