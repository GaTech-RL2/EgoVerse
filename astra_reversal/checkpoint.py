"""Identify checkpoint layouts without importing or allocating a policy model."""

import json
from pathlib import Path


def inspect_checkpoint(path):
    root = Path(path).expanduser()
    available = (root / "model.safetensors").is_file() or (root / "params").is_dir()
    config_path = root / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    result = {"path": str(root.resolve()), "weights_available": available}
    if config.get("type") == "pi05" and "chunk_size" in config:
        state_tokenized = False
        normalization = []
        for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            processor_path = root / name
            if not processor_path.is_file():
                continue
            for step in json.loads(processor_path.read_text()).get("steps", []):
                kind = step.get("registry_name", "")
                state_tokenized |= kind == "pi05_prepare_state_tokenizer_processor_step"
                if kind in ("normalizer_processor", "unnormalizer_processor"):
                    normalization.append(
                        {
                            "file": name,
                            "step": kind,
                            "features": step.get("config", {}).get("features", {}),
                            "norm_map": step.get("config", {}).get("norm_map", {}),
                            "state_file": step.get("state_file"),
                        }
                    )
        result.update(
            format="lerobot_pi05",
            horizon=config["chunk_size"],
            default_execute_steps=config.get("n_action_steps"),
            model_action_dim=config["max_action_dim"],
            environment_action_dim=config["output_features"]["action"]["shape"][0],
            state_in_language_tokens=state_tokenized,
            checkpoint_dtype=config.get("dtype"),
            normalization=normalization,
            openpi_loader_compatible=False,
            loader="lerobot",
            loader_compatible=available
            and all(
                (root / name).is_file()
                for name in ("policy_preprocessor.json", "policy_postprocessor.json")
            ),
            compatibility_note=(
                "LeRobot PI05: load_policy selects the existing LeRobot model, "
                "saved processors, and denoise_step velocity. "
                "FrozenOpenPI is only the OpenPI-specific loader."
            ),
        )
    elif available:
        stats = sorted((root / "assets").rglob("norm_stats.json"))
        result.update(
            format="openpi_pytorch"
            if (root / "model.safetensors").is_file()
            else "openpi_jax",
            normalization_files=[str(p.relative_to(root)) for p in stats],
            openpi_loader_compatible=bool(stats),
            loader="openpi",
            loader_compatible=bool(stats),
        )
    else:
        result.update(
            format="unknown", openpi_loader_compatible=False, loader_compatible=False
        )
    return result
