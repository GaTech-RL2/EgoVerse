"""Use LeRobot's existing PI05 velocity and sampler with frozen checkpoint weights.

The only model-specific flow work here is preparing the same prefix cache as
``PI05Pytorch.sample_actions`` and calling its ``denoise_step`` in either direction.
"""

import copy
import inspect
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from . import flow
from .checkpoint import inspect_checkpoint
from .policy_adapter import Condition
from .records import digest, file_sha256, to_numpy


def load_processors(path, device, tokenizer_path=None):
    # Import registers the state-to-language step in LeRobot's processor registry.
    from lerobot.policies.pi05 import processor_pi05  # noqa: F401
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        policy_action_to_transition,
        transition_to_policy_action,
    )

    overrides = {"device_processor": {"device": str(device)}}
    if tokenizer_path is not None:
        overrides["tokenizer_processor"] = {"tokenizer_name": str(tokenizer_path)}
    preprocess = PolicyProcessorPipeline.from_pretrained(
        path,
        config_filename="policy_preprocessor.json",
        overrides=overrides,
    )
    postprocess = PolicyProcessorPipeline.from_pretrained(
        path,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocess, postprocess


def load_native_model(checkpoint, device="cpu"):
    """Strictly load published weights into LeRobot's unmodified PI05 model."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05 import modeling_pi05
    from safetensors.torch import load_file
    from transformers.modeling_utils import no_init_weights

    path = Path(checkpoint).expanduser().resolve()
    if inspect_checkpoint(path)["format"] != "lerobot_pi05":
        raise ValueError("Expected a LeRobot PI05 checkpoint")
    config = PreTrainedConfig.from_pretrained(path)
    config.device = "cpu"
    config.compile_model = False
    config.gradient_checkpointing = False
    # The upstream convenience loader catches errors and can return unloaded
    # weights. Use its mapping but require a complete strict load here.
    with no_init_weights():
        policy = modeling_pi05.PI05Policy(config)
    # HF's no_init_weights also skips the usual embedding/lm_head tying.
    # Restore the declared sharing before resolving safetensors aliases.
    policy.model.paligemma_with_expert.paligemma.tie_weights()
    state = load_file(path / "model.safetensors", device="cpu")
    state = policy._fix_pytorch_state_dict_keys(state, config)
    state = {
        key if key.startswith("model.") else f"model.{key}": value
        for key, value in state.items()
    }
    # Safetensors may store shared embeddings once. Fill only aliases that the
    # instantiated upstream model already declares as the exact same Parameter.
    aliases = {}
    for name, parameter in policy.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    restored_aliases = {}
    for names in aliases.values():
        saved = next((name for name in names if name in state), None)
        if saved is not None:
            for name in names:
                if name not in state:
                    state[name] = state[saved]
                    restored_aliases[name] = saved
    # CPU safetensors are memory mapped; assignment avoids a second weight copy.
    policy.load_state_dict(state, strict=True, assign=True)
    del state
    # assign=True can create distinct Parameter objects sharing CPU storage.
    # Re-tie before a device transfer so the large embedding is copied once.
    policy.model.paligemma_with_expert.paligemma.tie_weights()
    policy.to(device).eval().requires_grad_(False)
    config.device = str(device)
    policy._astra_weight_aliases = restored_aliases
    return policy


def prepare_velocity(policy, batch):
    """Prepare the upstream prefix, then expose its existing denoise_step."""
    import torch
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    model = policy.model
    with torch.no_grad():
        images, masks = policy._preprocess_images(batch)
        embeddings, padding, attention = model.embed_prefix(
            images,
            masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
        )
        mask = model._prepare_attention_masks_4d(make_att_2d_masks(padding, attention))
        positions = torch.cumsum(padding, dim=1) - 1
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, cache = model.paligemma_with_expert.forward(
            attention_mask=mask,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[embeddings, None],
            use_cache=True,
        )
        to_numpy(padding)  # Include device synchronization in preparation time.

    def velocity(x, t):
        with torch.no_grad():
            times = torch.full((x.shape[0],), t, dtype=torch.float32, device=x.device)
            return model.denoise_step(
                prefix_pad_masks=padding,
                past_key_values=cache,
                x_t=x,
                timestep=times,
            )

    return velocity


class FrozenLeRobotPI05:
    backend = "pytorch"
    # Keep the simulator's 256px images for LeRobot's own resize operation.
    observation_image_size = None

    def __init__(self, policy, preprocessor, postprocessor, metadata):
        from lerobot.processor import PolicyProcessorPipeline, TokenizerProcessorStep

        self.policy, self.model, self.config = policy, policy.model, policy.config
        self.preprocessor, self.postprocessor = preprocessor, postprocessor
        self.metadata = metadata
        self.horizon, self.action_dim = (
            self.config.chunk_size,
            self.config.max_action_dim,
        )
        self.device = next(policy.parameters()).device
        if self.config.output_features["action"].shape != (7,):
            raise ValueError("LIBERO requires seven controller action channels")
        if self.config.n_obs_steps != 1 or policy._rtc_enabled():
            raise ValueError(
                "Reversal uses single observations and the unmodified PI05 flow"
            )
        tokenizers = [
            (i, step)
            for i, step in enumerate(preprocessor.steps)
            if isinstance(step, TokenizerProcessorStep)
        ]
        if len(tokenizers) != 1:
            raise ValueError("Expected one checkpoint tokenizer processor")
        index, self.tokenizer_step = tokenizers[0]
        self.max_token_len = min(
            self.config.tokenizer_max_length, self.tokenizer_step.max_length
        )
        self.prompt_preprocessor = PolicyProcessorPipeline(
            steps=preprocessor.steps[:index]
        )
        policy.eval().requires_grad_(False)

    @classmethod
    def load(
        cls,
        checkpoint,
        device="cpu",
        provenance="unknown",
        training_overlap="unknown",
        tokenizer_path=None,
    ):
        from lerobot.policies.pi05 import modeling_pi05
        from lerobot.processor import TokenizerProcessorStep
        from transformers.models.gemma import modeling_gemma
        from transformers.models.paligemma import modeling_paligemma
        from transformers.models.siglip import modeling_siglip

        path = Path(checkpoint).expanduser().resolve()
        artifact = inspect_checkpoint(path)
        if artifact["format"] != "lerobot_pi05":
            raise ValueError("Expected a LeRobot PI05 checkpoint")
        preprocess, postprocess = load_processors(path, device, tokenizer_path)
        policy = load_native_model(path, device)
        config = policy.config
        tokenizer = next(
            step.input_tokenizer
            for step in preprocess.steps
            if isinstance(step, TokenizerProcessorStep)
        )
        files = [path / "config.json", path / "model.safetensors"]
        files += sorted(path.glob("policy_*processor*"))
        metadata = {
            "artifact": str(path),
            "requested_artifact": str(checkpoint),
            "format": "lerobot_pi05",
            "backend": "pytorch",
            "frozen": True,
            "horizon": config.chunk_size,
            "model_action_dim": config.max_action_dim,
            "device": str(device),
            "dtype": config.dtype,
            "shared_weight_aliases": policy._astra_weight_aliases,
            "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
            "tokenizer_sha256": digest(
                {
                    "vocabulary": tokenizer.get_vocab(),
                    "special_tokens": tokenizer.special_tokens_map,
                    "backend": tokenizer.backend_tokenizer.to_str()
                    if hasattr(tokenizer, "backend_tokenizer")
                    else None,
                }
            ),
            "artifact_sha256": {p.name: file_sha256(p) for p in files if p.is_file()},
            "model_source_sha256": file_sha256(inspect.getfile(modeling_pi05)),
            "adapter_source_sha256": file_sha256(__file__),
            "processor_source_sha256": {
                type(step).__name__: file_sha256(inspect.getfile(type(step)))
                for step in [*preprocess.steps, *postprocess.steps]
            },
            "torch_version": version("torch"),
            "transformers_version": version("transformers"),
            "transformers_source_sha256": {
                module.__name__: file_sha256(inspect.getfile(module))
                for module in (modeling_gemma, modeling_paligemma, modeling_siglip)
            },
            "normalization": artifact["normalization"],
            "provenance": provenance,
            "training_overlap": training_overlap,
        }
        return cls(policy, preprocess, postprocess, metadata)

    def tensor(self, array):
        import torch

        return torch.as_tensor(
            np.asarray(array).copy(), dtype=torch.float32, device=self.device
        )

    def noise(self, rng):
        return self.tensor(
            rng.standard_normal((1, self.horizon, self.action_dim)).astype(np.float32)
        )

    def _batch(self, raw, *, images=True):
        import torch

        state = np.asarray(raw["observation/state"], dtype=np.float32)
        if (
            state.shape != tuple(self.config.input_features["observation.state"].shape)
            or not np.isfinite(state).all()
        ):
            raise ValueError(
                "Observation state must match the checkpoint's feature shape"
            )
        batch = {
            "observation.state": torch.from_numpy(state.copy()),
            "task": raw.get("prompt", ""),
        }
        if images:
            for source, target in (
                ("observation/image", "observation.images.image"),
                ("observation/wrist_image", "observation.images.image2"),
            ):
                image = np.asarray(raw[source])
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError("Camera images must be uint8 HWC RGB")
                batch[target] = (
                    torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
                )
        if "actions" in raw:
            actions = np.asarray(raw["actions"], dtype=np.float32)
            if actions.shape != (self.horizon, 7) or not np.isfinite(actions).all():
                raise ValueError("Actions must be a finite full [H, 7] chunk")
            batch["action"] = torch.from_numpy(actions.copy())[None]
        return batch

    def prompt_length(self, prompt, observation):
        # Run the checkpoint's own normalization and state formatting first.
        batch = self.prompt_preprocessor(
            self._batch({**observation, "prompt": prompt}, images=False)
        )
        encoded = self.tokenizer_step.input_tokenizer(
            batch["task"],
            padding=False,
            truncation=False,
        )
        return len(encoded["input_ids"][0])

    def assemble_prompt(self, original, subgoal, constraints, *, observation):
        if self.prompt_length(original, observation) > self.max_token_len:
            raise ValueError(
                "Task plus robot state exceeds the checkpoint token budget"
            )
        accepted, omitted = original, []
        for piece in [
            f"Current subgoal: {subgoal}",
            *[f"Constraint: {x}" for x in constraints],
        ]:
            candidate = accepted + "\n" + piece
            if self.prompt_length(candidate, observation) <= self.max_token_len:
                accepted = candidate
            else:
                omitted.append(piece)
        return accepted, omitted

    def _preprocess(self, raw):
        if self.prompt_length(raw.get("prompt", ""), raw) > self.max_token_len:
            raise ValueError("Task plus robot state would be silently truncated")
        return self.preprocessor(self._batch(raw))

    def input_transform(self, raw):
        batch = self._preprocess(raw)
        result = {"state": to_numpy(batch["observation.state"])[0]}
        if "action" in batch:
            result["actions"] = to_numpy(self.policy.prepare_action(batch))[0]
        return result

    def output_transform(self, data):
        actions = self.tensor(np.asarray(data["actions"])[None, :, :7])
        return {"actions": to_numpy(self.postprocessor(actions))[0]}

    def prepare(self, observation, observation_id, prompt):
        started = time.perf_counter()
        raw = {**copy.deepcopy(observation), "prompt": prompt}
        batch = self._preprocess(raw)
        velocity = prepare_velocity(self.policy, batch)

        return Condition(
            digest(raw),
            observation_id,
            prompt,
            raw,
            batch["observation.state"],
            velocity,
            time.perf_counter() - started,
        )

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
        import torch

        with torch.no_grad():
            batch = self._preprocess(condition.raw)
            actions = self.policy.predict_action_chunk(
                batch, noise=noise, num_steps=steps
            )
            return to_numpy(self.postprocessor(actions))[0]
