"""Compile a stored input-setting skill through existing frozen pi0.5 hooks."""

from collections import OrderedDict
from pathlib import Path

import numpy as np

from .demo_segments import CAMERAS
from .demo_skill_program import has_effect, substituted_observation
from .interpolation_catalog import DATASET_REPO, DATASET_REVISION
from .records import digest, file_sha256


class InputSkillConditioner:
    def __init__(self, policy, bank, text_banks):
        self.policy, self.bank, self.text_banks = policy, bank, text_banks
        self.vision_cache = OrderedDict()
        self.capture_count = 0
        self.cache_identity = digest(
            {
                "policy": getattr(policy, "metadata", {}),
                "bank_id": bank.bank_id,
                "paired_cameras": CAMERAS,
                "hooks": file_sha256(
                    Path(__file__).with_name("vision_interpolation.py")
                ),
                "compiler": file_sha256(__file__),
            }
        )

    def _donor_pair(self, source_id, frame):
        from .image_donor_bank import DonorImage

        row = self.bank.sources[source_id]
        obs = self.bank.observation(source_id, frame)
        donor_id = f"{self.bank.bank_id}:{source_id}:{frame}"
        common = {
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "library_id": self.bank.bank_id,
            "sample_sha256": digest([donor_id, obs]),
            "source_id": source_id,
            "prompt": row["prompt"],
            "episode_index": row["episode_index"],
            "frame_index": frame,
            "phase": frame / max(1, row["frame_count"] - 1),
            "donor_id": donor_id,
        }
        return {
            camera: DonorImage(
                donor_id,
                camera,
                obs[camera],
                {**common, "camera": camera, "pixels_sha256": digest(obs[camera])},
            )
            for camera in CAMERAS
        }

    def prepare(self, live, prompt, choice):
        policy = self.policy
        if not has_effect(choice, "input_skill_library"):
            return policy.prepare(live, digest(live), prompt), None, live
        effective = {k: np.array(v, copy=True) for k, v in live.items()}
        visual, alpha, language = (
            choice["vision_operator"],
            choice["alpha"],
            choice["language"],
        )
        if (
            language is not None
            and language["operator"] == "tli"
            and language["source_a_id"] == language["source_b_id"]
        ):
            language = None  # Its latent difference is exactly zero.
        if visual == "pixels":
            effective = substituted_observation(self.bank, choice, live)
        elif visual == "occlusion" and alpha:
            box = choice["occlusion_box"]
            for camera in CAMERAS:
                height, width, _ = effective[camera].shape
                # Round inward so rasterization never exceeds the declared area.
                x0, y0, x1, y1 = (
                    int(np.ceil(box[0] * width)),
                    int(np.ceil(box[1] * height)),
                    int(box[2] * width),
                    int(box[3] * height),
                )
                region = effective[camera][y0:y1, x0:x1]
                effective[camera][y0:y1, x0:x1] = np.rint(
                    (1 - alpha) * region.astype(np.float32) + alpha * 127
                ).astype(np.uint8)
        text = {}
        if language is not None:
            a, b = language["source_a_id"], language["source_b_id"]
            text = {
                "source_prompts": (
                    self.bank.sources[a]["prompt"],
                    self.bank.sources[b]["prompt"],
                )
            }
            if language["operator"] == "tli" and language["alpha"] != 0.5 and a != b:
                if a not in self.text_banks or b not in self.text_banks:
                    raise ValueError(
                        "The selected TLI source has no verified text bank"
                    )
                text["text_latents"] = {
                    "a": self.text_banks[a],
                    "b": self.text_banks[b],
                }
        if visual in ("vei", "vli"):
            from .vision_interpolation import (
                capture_vision_bank,
                prepare_vision_interpolated,
            )

            vision_bank = None
            if alpha:
                key = (
                    self.cache_identity,
                    prompt,
                    choice["source_id"],
                    choice["frame"],
                    visual,
                )
                if key not in self.vision_cache:
                    # At most two 18-layer visual tensors stay resident on CPU.
                    while len(self.vision_cache) >= 2:
                        self.vision_cache.popitem(last=False)
                    self.vision_cache[key] = capture_vision_bank(
                        policy,
                        self._donor_pair(choice["source_id"], choice["frame"]),
                        prompt=prompt,
                        include_layers=visual == "vli",
                    )
                    self.capture_count += 1
                self.vision_cache.move_to_end(key)
                vision_bank = self.vision_cache[key]
            if language is not None:
                if language["operator"] != "tli":
                    raise ValueError(
                        "Simultaneous TEI and visual latent hooks are unsupported"
                    )
                text.update(language_alpha=language["alpha"], language_operator="tli")
            condition, receipt = prepare_vision_interpolated(
                policy,
                effective,
                digest(effective),
                prompt,
                bank=vision_bank,
                alpha=alpha,
                operator=visual,
                **text,
            )
        elif language is not None:
            condition, receipt = policy.prepare_interpolated(
                effective,
                digest(effective),
                prompt,
                alpha=language["alpha"],
                operator=language["operator"],
                **text,
            )
        else:
            condition, receipt = (
                policy.prepare(effective, digest(effective), prompt),
                {"operator": visual, "alpha": alpha},
            )
        return (
            condition,
            {
                "setting": choice,
                "operator_receipt": receipt,
                "donor_captures_total": self.capture_count,
                "live_observation_sha256": digest(live),
                "effective_observation_sha256": digest(effective),
            },
            effective,
        )
