"""Frozen OpenPI adapter with shared conditioning for both integration directions.

The backend-specific prefix preparation follows the pinned local OpenPI sampler.
No upstream modules or checkpoint weights are modified. Real-model parity must
be measured with ``diagnostics`` before interpreting experiments.
"""

import copy
import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import flow
from .checkpoint import inspect_checkpoint
from .records import digest, file_sha256, to_numpy


@dataclass
class Condition:
    condition_id: str
    observation_id: str
    prompt: str
    raw: dict
    state: Any
    velocity: Any
    preparation_seconds: float


def load_policy(
    checkpoint,
    config_name="pi05_libero",
    device="cpu",
    provenance="unknown",
    training_overlap="unknown",
    *,
    tokenizer_path=None,
    input_profile="checkpoint",
    reference_assets=None,
):
    """Select the implementation described by the checkpoint's own config."""
    if checkpoint and Path(checkpoint).expanduser().is_dir():
        if inspect_checkpoint(checkpoint)["format"] == "lerobot_pi05":
            from .lerobot_policy import FrozenLeRobotPI05

            policy = FrozenLeRobotPI05.load(
                checkpoint,
                device,
                provenance,
                training_overlap,
                tokenizer_path=tokenizer_path,
            )
            if input_profile == "openpi_libero":
                from .openpi_inputs import use_openpi_libero_inputs

                return use_openpi_libero_inputs(policy, reference_assets)
            if input_profile != "checkpoint" or reference_assets is not None:
                raise ValueError("Invalid LeRobot input profile/assets")
            return policy
    if input_profile != "checkpoint" or reference_assets is not None:
        raise ValueError("Input profile overrides are only for the LeRobot export")
    if tokenizer_path is not None:
        raise ValueError("tokenizer_path is only supported for LeRobot checkpoints")
    return FrozenOpenPI.load(
        checkpoint, config_name, device, provenance, training_overlap
    )


class FrozenOpenPI:
    observation_image_size = 224

    def __init__(self, policy, train_config, metadata: dict):
        self.policy = policy
        self.model = policy._model
        self.config = train_config
        self.metadata = metadata
        self.backend = "pytorch" if policy._is_pytorch_model else "jax"
        self.horizon = train_config.model.action_horizon
        self.action_dim = train_config.model.action_dim
        self.input_transform = policy._input_transform
        self.output_transform = policy._output_transform
        if not train_config.model.pi05:
            raise ValueError("These experiments require a frozen pi0.5 checkpoint")
        data_config = train_config.data.create(
            train_config.assets_dirs, train_config.model
        )
        tokenizers = [
            x
            for x in data_config.model_transforms.inputs
            if type(x).__name__ == "TokenizePrompt"
        ]
        if len(tokenizers) != 1 or tokenizers[0].discrete_state_input:
            raise ValueError(
                "This OpenPI prompt budgeter expects discrete_state_input=False"
            )
        self.tokenizer = tokenizers[0].tokenizer
        self.max_token_len = train_config.model.max_token_len
        if self.backend == "pytorch":
            self.model.eval()
            self.model.requires_grad_(False)

    @classmethod
    def load(
        cls,
        checkpoint: str,
        config_name="pi05_libero",
        device="cpu",
        provenance="unknown",
        training_overlap="unknown",
    ):
        if not checkpoint:
            raise ValueError("A checkpoint artifact must be explicitly configured")
        if Path(checkpoint).expanduser().is_dir():
            artifact = inspect_checkpoint(checkpoint)
            if artifact["format"] == "lerobot_pi05":
                raise ValueError(artifact["compatibility_note"])
        from openpi import transforms
        from openpi.policies import libero_policy, policy_config
        from openpi.shared import download
        from openpi.training import config

        path = Path(download.maybe_download(checkpoint))
        if not ((path / "model.safetensors").is_file() or (path / "params").is_dir()):
            raise ValueError(f"No OpenPI PyTorch or JAX weights found in {path}")
        stats = sorted((path / "assets").rglob("norm_stats.json"))
        if not stats:
            raise ValueError("Checkpoint normalization assets are required")
        normalization = {str(p.relative_to(path)): file_sha256(p) for p in stats}
        weights = (
            [path / "model.safetensors"]
            if (path / "model.safetensors").is_file()
            else sorted(p for p in (path / "params").rglob("*") if p.is_file())
        )
        if not weights:
            raise ValueError("Checkpoint weight artifact is empty")
        weight_hashes = {str(p.relative_to(path)): file_sha256(p) for p in weights}
        train_config = config.get_config(config_name)
        policy = policy_config.create_trained_policy(
            train_config, path, pytorch_device=device
        )
        metadata = {
            "artifact": str(path),
            "requested_artifact": checkpoint,
            "config_name": config_name,
            "normalization_sha256": normalization,
            "weights_sha256": weight_hashes,
            "model_source_sha256": digest(inspect.getsource(type(policy._model))),
            "adapter_source_sha256": file_sha256(__file__),
            "transforms_source_sha256": file_sha256(inspect.getfile(transforms)),
            "libero_adapter_source_sha256": file_sha256(inspect.getfile(libero_policy)),
            "config_source_sha256": file_sha256(inspect.getfile(config)),
            "provenance": provenance,
            "training_overlap": training_overlap,
            "frozen": True,
            "backend": "pytorch" if policy._is_pytorch_model else "jax",
            "horizon": train_config.model.action_horizon,
            "model_action_dim": train_config.model.action_dim,
        }
        return cls(policy, train_config, metadata)

    def tensor(self, array):
        if self.backend == "pytorch":
            import torch

            return torch.as_tensor(
                np.asarray(array).copy(),
                dtype=torch.float32,
                device=self.policy._pytorch_device,
            )
        import jax.numpy as jnp

        return jnp.asarray(array, dtype=jnp.float32)

    def noise(self, rng):
        # The saved tensor, not a backend-specific RNG seed, defines paired noise.
        return self.tensor(
            rng.standard_normal((1, self.horizon, self.action_dim)).astype(np.float32)
        )

    def prompt_length(self, prompt):
        text = prompt.strip().replace("_", " ").replace("\n", " ")
        tokenizer = self.tokenizer._tokenizer
        return len(tokenizer.encode(text, add_bos=True)) + len(tokenizer.encode("\n"))

    def assemble_prompt(self, original, subgoal, constraints, *, observation=None):
        if self.prompt_length(original) > self.max_token_len:
            raise ValueError(
                "Original task instruction exceeds the checkpoint token budget"
            )
        accepted = original
        omitted = []
        pieces = [
            f"Current subgoal: {subgoal}",
            *[f"Constraint: {x}" for x in constraints],
        ]
        for piece in pieces:
            candidate = accepted + "\n" + piece
            if self.prompt_length(candidate) <= self.max_token_len:
                accepted = candidate
            else:
                omitted.append(piece)
        return accepted, omitted

    def prepare(self, observation: dict, observation_id: str, prompt: str) -> Condition:
        import jax
        from openpi.models import model as model_module

        if self.prompt_length(prompt) > self.max_token_len:
            raise ValueError("Policy prompt would be silently truncated")
        raw = {**copy.deepcopy(observation), "prompt": prompt}
        started = time.perf_counter()
        inputs = self.input_transform(copy.deepcopy(raw))
        if self.backend == "pytorch":
            import torch

            batch = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self.policy._pytorch_device)[
                    None
                ],
                inputs,
            )
            obs = model_module.Observation.from_dict(batch)
            velocity = self._prepare_torch(obs)
        else:
            import jax.numpy as jnp

            batch = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
            obs = model_module.Observation.from_dict(batch)
            velocity = self._prepare_jax(obs)
        return Condition(
            digest(raw),
            observation_id,
            prompt,
            raw,
            batch["state"],
            velocity,
            time.perf_counter() - started,
        )

    def _prepare_torch(self, obs):
        import torch
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        model = self.model
        with torch.no_grad():
            images, masks, tokens, token_masks, state = model._preprocess_observation(
                obs, train=False
            )
            embeddings, padding, attention = model.embed_prefix(
                images, masks, tokens, token_masks
            )
            mask = model._prepare_attention_masks_4d(
                make_att_2d_masks(padding, attention)
            )
            positions = torch.cumsum(padding, dim=1) - 1
            model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
            _, cache = model.paligemma_with_expert.forward(
                attention_mask=mask,
                position_ids=positions,
                past_key_values=None,
                inputs_embeds=[embeddings, None],
                use_cache=True,
            )
            # Ensure prefix preparation latency includes the actual device work.
            to_numpy(state)

        def velocity(x, t):
            with torch.no_grad():
                times = torch.full(
                    (x.shape[0],), t, dtype=torch.float32, device=x.device
                )
                return model.denoise_step(state, padding, cache, x, times)

        return velocity

    def _prepare_jax(self, obs):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as model_module
        from openpi.models.pi0 import make_attn_mask

        model = self.model
        obs = model_module.preprocess_observation(None, obs, train=False)
        prefix, padding, attention = model.embed_prefix(obs)
        _, cache = model.PaliGemma.llm(
            [prefix, None],
            mask=make_attn_mask(padding, attention),
            positions=jnp.cumsum(padding, axis=1) - 1,
        )
        jax.block_until_ready(cache)

        # Capture only this condition; no prefix survives an observation/prompt change.
        def velocity(x, t):
            suffix, suffix_mask, suffix_ar_mask, adarms = model.embed_suffix(
                obs, x, jnp.full((x.shape[0],), t)
            )
            prefix_mask = jnp.broadcast_to(
                padding[:, None, :], (x.shape[0], suffix.shape[1], padding.shape[1])
            )
            mask = jnp.concatenate(
                [prefix_mask, make_attn_mask(suffix_mask, suffix_ar_mask)], axis=-1
            )
            positions = (
                jnp.sum(padding, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )
            (_, output), _ = model.PaliGemma.llm(
                [None, suffix],
                mask=mask,
                positions=positions,
                kv_cache=cache,
                adarms_cond=[None, adarms],
            )
            return model.action_out_proj(output[:, -self.horizon :])

        return velocity

    def sample(
        self,
        condition,
        noise,
        *,
        steps,
        solver="euler",
        save_trace=False,
        **solver_options,
    ):
        return flow.generate(
            condition.velocity,
            noise,
            steps=steps,
            solver=solver,
            save_trace=save_trace,
            **solver_options,
        )

    def invert(
        self,
        condition,
        actions,
        *,
        steps,
        solver="euler",
        save_trace=False,
        **solver_options,
    ):
        return flow.invert(
            condition.velocity,
            self.tensor(to_numpy(actions)),
            steps=steps,
            solver=solver,
            save_trace=save_trace,
            **solver_options,
        )

    def reference_actions(self, condition, noise, *, steps):
        """Upstream sampler for diagnostic parity; returns decoded seven-channel actions."""
        prior = self.policy._sample_kwargs
        try:
            self.policy._sample_kwargs = {**prior, "num_steps": steps}
            return self.policy.infer(
                copy.deepcopy(condition.raw), noise=to_numpy(noise)
            )["actions"]
        finally:
            self.policy._sample_kwargs = prior
