"""Audit RLinf's math-only OpenPI core against the selected LeRobot export.

This imports the pinned implementation without its unrelated Ray/robot factory
initializers. It is a checkpoint-compatibility probe, not an RL training result.
"""

import contextlib
import importlib
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REVISION = "c70606f08cdca259b8dec03d4430926b5b8fac9d"


def import_core(root):
    root = Path(root).resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if (
        revision != REVISION
        or subprocess.check_output(
            ["git", "diff", "HEAD", "--", "rlinf"], cwd=root, text=True
        ).strip()
    ):
        raise ValueError("RLinf must match the unmodified pinned source")
    sys.path.insert(0, str(root))
    for name in (
        "rlinf.models",
        "rlinf.models.embodiment",
        "rlinf.models.embodiment.openpi",
    ):
        path = root / Path(*name.split("."))
        if name in sys.modules:
            if list(sys.modules[name].__path__) != [str(path)]:
                raise ValueError("A different RLinf installation is already imported")
            continue
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    base = "rlinf.models.embodiment.openpi."
    return types.SimpleNamespace(
        Pi0=importlib.import_module(base + "pi0").Pi0,
        Config=importlib.import_module(base + "pi0_config").Pi0Config,
        model=importlib.import_module(base + "modules.model"),
        gemma=importlib.import_module(base + "modules.gemma"),
        utils=importlib.import_module(base + "modules.utils"),
        converter=importlib.import_module(
            "rlinf.utils.ckpt_convertor.openpi.openpi_pytorch_to_openpi"
        ).old_to_new_state_dict,
    )


def load_converted(core, checkpoint, *, horizon, device):
    from safetensors.torch import load_file

    core.utils.set_torch_compile(False)
    with torch.device("meta"):
        model = core.Pi0(
            core.Config(
                pi05=True,
                action_horizon=horizon,
                dtype="float32",
                max_token_len=200,
                discrete_state_input=False,
            )
        )
    source = load_file(Path(checkpoint) / "model.safetensors", device="cpu")
    converted = core.converter(source)
    receipt = {"source_tensors": len(source), "converted_tensors": len(converted)}
    model.load_state_dict(converted, strict=True, assign=True)
    del source, converted
    model.to(device).eval().requires_grad_(False)
    return model, receipt


def observation_from_native(core, policy, batch):
    """Reuse exact native pixels/tokens; do not apply a second normalization."""
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, masks = policy.policy._preprocess_images(batch)
    if len(images) != len(core.model.IMAGE_KEYS):
        raise ValueError("Expected native two-camera plus masked-camera input")
    return core.model.Observation(
        images={
            key: img.permute(0, 2, 3, 1)
            for key, img in zip(core.model.IMAGE_KEYS, images)
        },
        image_masks=dict(zip(core.model.IMAGE_KEYS, masks)),
        state=batch["observation.state"],
        tokenized_prompt=batch[OBS_LANGUAGE_TOKENS],
        tokenized_prompt_mask=batch[OBS_LANGUAGE_ATTENTION_MASK],
    )


def difference(actual, reference, *, atol=1e-4, rtol=1e-4):
    actual, reference = actual.detach().float(), reference.detach().float()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    return {
        "max_abs": float((actual - reference).abs().max()),
        "rms": float((actual - reference).square().mean().sqrt()),
        "reference_rms": float(reference.square().mean().sqrt()),
        "finite": finite,
        "atol": atol,
        "rtol": rtol,
        "passed": finite
        and bool(torch.allclose(actual, reference, atol=atol, rtol=rtol)),
    }


@contextlib.contextmanager
def match_native_gelu(core, model, native_policy):
    """Explicit compatibility experiment; restore every changed callable."""
    text_act = native_policy.model.paligemma_with_expert.paligemma.config.text_config.hidden_activation
    vision_act = native_policy.model.paligemma_with_expert.paligemma.config.vision_config.hidden_act
    if text_act != "gelu_pytorch_tanh" or vision_act != "gelu_pytorch_tanh":
        raise ValueError(f"Unexpected native activations: {text_act}, {vision_act}")
    old_gelu = core.gemma.gelu_glu
    saved = [(layer.mlp, layer.mlp.forward) for layer in model.img.encoder.layers]

    def vision_forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x), approximate="tanh")))

    core.gemma.gelu_glu = lambda gate, value: F.gelu(gate, approximate="tanh") * value
    for module, _ in saved:
        module.forward = types.MethodType(vision_forward, module)
    try:
        yield {"text_activation": text_act, "vision_activation": vision_act}
    finally:
        core.gemma.gelu_glu = old_gelu
        for module, forward in saved:
            module.forward = forward


@torch.no_grad()
def measure(core, converted, native, observation, prompt, *, seeds=(173, 179)):
    from astra_reversal.lerobot_policy import prepare_velocity

    raw = {**observation, "prompt": prompt}
    batch = native._preprocess(raw)
    obs = observation_from_native(core, native, batch)
    native_velocity = prepare_velocity(native.policy, batch)
    _, mask, cache = converted.build_prefix_cache(obs)

    def velocity(x, t):
        hidden = converted.run_suffix(
            obs, x, torch.full((x.shape[0],), t, device=x.device), cache, mask
        )
        return converted.velocity_from_suffix(hidden)

    results = []
    for seed in seeds:
        noise = native.noise(np.random.default_rng(seed))
        row = {"seed": seed, "velocities": []}
        for t in (1.0, 0.5, 0.1):
            row["velocities"].append(
                {"t": t, **difference(velocity(noise, t), native_velocity(noise, t))}
            )
        x, ref = noise.clone(), noise.clone()
        for i in range(10):
            t = 1.0 - i / 10
            x = x - 0.1 * velocity(x, t)
            ref = ref - 0.1 * native_velocity(ref, t)
        row["normalized_actions"] = difference(x, ref, atol=5e-4)
        row["controller_actions"] = difference(
            native.postprocessor(x[:, :, :7]),
            native.postprocessor(ref[:, :, :7]),
            atol=5e-4,
        )
        row["passed"] = (
            all(v["passed"] for v in row["velocities"])
            and row["normalized_actions"]["passed"]
            and row["controller_actions"]["passed"]
        )
        results.append(row)
    return {"cases": results, "passed": all(row["passed"] for row in results)}
