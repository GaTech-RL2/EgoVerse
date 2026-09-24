"""Explicit OpenPI LIBERO input profile for the selected LeRobot weight export.

This is a separate experiment, not the artifact's saved processor behavior.
Weights and native LeRobot flow velocity remain unchanged. The source settings
are OpenPI's pi05_libero config: H=10, plain language, quantile normalization,
and the example client's 224px PIL images.
"""

import copy
import json
from pathlib import Path

import numpy as np

from .lerobot_policy import FrozenLeRobotPI05
from .records import file_sha256, to_numpy


class OpenPILiberoInputs(FrozenLeRobotPI05):
    observation_image_size = 224

    def _tokens(self, prompt):
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        return self.sentencepiece.encode(
            cleaned, add_bos=True
        ) + self.sentencepiece.encode("\n")

    def prompt_length(self, prompt, observation):
        return len(self._tokens(prompt))

    def normalize(self, array, key):
        stats = self.norm_stats[key]
        return (array - stats["q01"]) / (stats["q99"] - stats["q01"] + 1e-6) * 2.0 - 1.0

    def _preprocess(self, raw):
        import torch
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )
        from openpi_client import image_tools

        tokens = self._tokens(raw.get("prompt", ""))
        if len(tokens) > self.max_token_len:
            raise ValueError("Task would be silently truncated")
        images = {
            key: image_tools.resize_with_pad(raw[key], 224, 224)
            for key in ("observation/image", "observation/wrist_image")
        }
        batch = self._batch({**raw, **images})
        batch["observation.state"] = self.tensor(
            np.pad(
                self.normalize(to_numpy(batch["observation.state"]), "state"), (0, 24)
            )[None]
        )
        if "action" in batch:
            batch["action"] = self.tensor(
                self.normalize(to_numpy(batch["action"]), "actions")
            )
        batch[OBS_LANGUAGE_TOKENS] = torch.tensor(
            [tokens + [0] * (self.max_token_len - len(tokens))],
            dtype=torch.long,
            device=self.device,
        )
        batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.tensor(
            [[True] * len(tokens) + [False] * (self.max_token_len - len(tokens))],
            dtype=torch.bool,
            device=self.device,
        )
        for key in self.config.image_features:
            if key in batch:
                batch[key] = batch[key][None].to(self.device)
        return batch

    def output_transform(self, data):
        values = np.asarray(data["actions"])[..., :7]
        stats = self.norm_stats["actions"]
        return {
            "actions": (values + 1.0) / 2.0 * (stats["q99"] - stats["q01"] + 1e-6)
            + stats["q01"]
        }

    def reference_actions(self, condition, noise, *, steps):
        import torch

        with torch.no_grad():
            values = self.policy.predict_action_chunk(
                self._preprocess(condition.raw), noise=noise, num_steps=steps
            )
        return self.output_transform({"actions": to_numpy(values)[0]})["actions"]


def use_openpi_libero_inputs(native, asset_directory):
    import sentencepiece

    if native.horizon != 50 or native.action_dim != 32:
        raise ValueError(
            "This profile is specific to the selected 50 x 32 LeRobot export"
        )
    root = Path(asset_directory).resolve()
    inventory = json.loads(
        (
            Path(__file__).parent / "checkpoints/openpi_libero_input_assets.json"
        ).read_text()
    )
    files = {
        "norm_stats": root / "norm_stats.json",
        "tokenizer": root / "paligemma_tokenizer.model",
    }
    for key, path in files.items():
        if file_sha256(path) != inventory[key]["sha256"]:
            raise ValueError(f"OpenPI reference asset hash mismatch: {key}")
    result = OpenPILiberoInputs(
        native.policy,
        native.preprocessor,
        native.postprocessor,
        copy.deepcopy(native.metadata),
    )
    result.norm_stats = {
        key: {name: np.asarray(value) for name, value in stats.items()}
        for key, stats in json.loads(files["norm_stats"].read_text())[
            "norm_stats"
        ].items()
    }
    result.sentencepiece = sentencepiece.SentencePieceProcessor(
        model_proto=files["tokenizer"].read_bytes()
    )
    result.config.chunk_size = result.horizon = 10
    result.metadata.update(
        horizon=10,
        input_profile="openpi_libero",
        input_profile_source_sha256=file_sha256(__file__),
        input_profile_assets=inventory,
        normalization={"type": "quantile", "sha256": inventory["norm_stats"]["sha256"]},
        runtime_overrides={
            "chunk_size": 10,
            "discrete_state_input": False,
            "image_size": 224,
        },
        interpretation="LeRobot exported weights with OpenPI LIBERO inputs; published checkpoint success is not established",
    )
    return result
