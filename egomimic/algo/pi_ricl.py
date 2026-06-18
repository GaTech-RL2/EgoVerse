"""PIRicl: pi0.5 + retrieval-based in-context learning (P2).

A thin subclass of :class:`egomimic.algo.pi.PI`. Per P0, the flow pi0.5
``embed_prefix`` embeds *all* images in the observation plus the full
``tokenized_prompt`` as one bidirectional prefix, so RICL needs **no change to
``PI0Pytorch``**. This subclass only augments how the observation is built:

- ``_build_prompts``: splice each query's k retrieved demos' (state, action),
  discretized with the same binning as the State block, into the prompt text.
- ``_robomimic_to_pi_data``: append the k retrieved ``base_0_rgb`` frames as
  extra entries in the observation image dict (the model embeds them into the
  prefix automatically).
- ``process_batch_for_training``: carry the collate's ``ricl_*`` keys through to
  the per-embodiment processed batch.

If no ``ricl_*`` keys are present (e.g. the k=0 zero-context floor used in eval),
behaviour is identical to the base ``PI``. The actual injection logic lives in
:mod:`egomimic.ricl.conditioning` (import-light, unit-tested without openpi).
"""

from __future__ import annotations

import logging
import random

import torch
from overrides import override

from egomimic.algo.pi import PI
from egomimic.ricl import conditioning as ricl_cond
from egomimic.rldb.embodiment.embodiment import get_embodiment_id

logger = logging.getLogger(__name__)


def _install_retrieved_image_passthrough() -> None:
    """Make openpi keep the retrieved demo images instead of dropping them.

    ``augment_images_with_retrieved`` appends ``retrieved_*`` keys to the
    observation image dict, and pi0.5's ``embed_prefix`` *does* embed every image
    it is handed. But ``preprocess_observation_pytorch`` (called just before
    ``embed_prefix``) iterates a FIXED ``IMAGE_KEYS`` = (base_0_rgb,
    left_wrist_0_rgb, right_wrist_0_rgb) and silently drops every other key — so
    the retrieved demo *images* never reach the model (only the discretized
    text). This wraps that function to also pass through any extra image keys
    present on the observation (e.g. ``retrieved_*``). Backward compatible:
    non-RICL observations carry no extra keys, so behaviour is identical there.

    We patch the module attribute (``pi0_pytorch`` calls
    ``_preprocessing.preprocess_observation_pytorch`` by attribute lookup), so
    the override takes effect without editing vendored openpi.
    """
    import openpi.models_pytorch.preprocessing_pytorch as _pp

    if getattr(_pp, "_ricl_image_passthrough", False):
        return
    _orig = _pp.preprocess_observation_pytorch

    def _preprocess_with_retrieved(
        observation,
        *,
        train: bool = False,
        image_keys=_pp.IMAGE_KEYS,
        image_resolution=_pp.IMAGE_RESOLUTION,
    ):
        extra = tuple(k for k in observation.images if k not in image_keys)
        keys = tuple(image_keys) + extra
        return _orig(
            observation,
            train=train,
            image_keys=keys,
            image_resolution=image_resolution,
        )

    _pp.preprocess_observation_pytorch = _preprocess_with_retrieved
    _pp._ricl_image_passthrough = True
    logger.info(
        "PIRicl: patched preprocess_observation_pytorch to pass through "
        "retrieved_* demo images (were being dropped by fixed IMAGE_KEYS)."
    )


_install_retrieved_image_passthrough()

# Keys the RICL collate attaches per query sample (see egomimic/ricl + P3 collate).
RICL_BATCH_KEYS = (
    "ricl_retrieved_images",  # (B, k, C, H, W) or (B, k, H, W, C)
    "ricl_retrieved_state",  # (B, k, Ds)  normalized to the query convention
    "ricl_retrieved_action",  # (B, k, Ha, Da) or (B, k, Da), normalized 32-D
    "ricl_retrieved_mask",  # (B, k) bool, valid neighbor (handles < k)
    "ricl_retrieved_dist",  # (B, k) float, kNN distances (for future interpolation)
)


class PIRicl(PI):
    """pi0.5 with prefix-concatenated retrieved in-context demonstrations."""

    def __init__(
        self,
        *args,
        num_retrieved_observations: int = 4,
        retrieved_action_steps: int = 1,
        ricl_base_key: str = "base_0_rgb",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_retrieved_observations = int(num_retrieved_observations)
        self.retrieved_action_steps = int(retrieved_action_steps)
        self.ricl_base_key = ricl_base_key

        est = ricl_cond.estimate_prompt_tokens(
            self.num_retrieved_observations, self.retrieved_action_steps
        )
        if self.tokenizer_max_length is not None and est > self.tokenizer_max_length:
            logger.warning(
                "RICL prompt may exceed tokenizer_max_length (~%d est tokens vs %d). "
                "Increase model.max_token_len / tokenizer_max_length, or reduce "
                "num_retrieved_observations / retrieved_action_steps.",
                est,
                self.tokenizer_max_length,
            )
        logger.info(
            "PIRicl: k=%d retrieved obs, action_steps=%d, base_key=%s",
            self.num_retrieved_observations,
            self.retrieved_action_steps,
            self.ricl_base_key,
        )

    # ------------------------------------------------------------------
    # Carry ricl_* keys through process_batch_for_training
    # ------------------------------------------------------------------
    @override
    def process_batch_for_training(self, batch):
        processed = super().process_batch_for_training(batch)
        for embodiment_name, _batch in batch.items():
            emb_id = get_embodiment_id(embodiment_name)
            if emb_id not in processed:
                continue
            for key in RICL_BATCH_KEYS:
                if key in _batch:
                    val = _batch[key]
                    if isinstance(val, torch.Tensor):
                        val = val.to(self.device)
                        if val.is_floating_point():
                            val = val.float()
                    processed[emb_id][key] = val
        return processed

    # ------------------------------------------------------------------
    # Splice retrieved (state, action) text into the prompt
    # ------------------------------------------------------------------
    def _raw_prompts(self, _batch, batch_size: int) -> list[str]:
        """Sample one *raw* task string per item (mirrors ``PI._build_prompts``'s
        own sampling), used to label each retrieved demo exemplar. Within-group
        retrieval means the demos share the query's task, so reusing the query's
        task text is faithful and needs no per-demo prompt plumbing in the collate.
        """
        if self.annotation_key is None or self.annotation_key not in _batch:
            return [self.default_prompt] * batch_size
        out = []
        for sample in _batch[self.annotation_key]:
            if not sample:
                out.append(self.default_prompt)
            elif self.sampling_mode == "random":
                out.append(sample[random.randint(0, len(sample) - 1)])
            else:  # "first"
                out.append(sample[0])
        return out

    @override
    def _build_prompts(
        self, _batch, embodiment_name: str, batch_size: int
    ) -> list[str]:
        prompts = super()._build_prompts(_batch, embodiment_name, batch_size)
        if "ricl_retrieved_state" not in _batch:
            return prompts  # zero-context (k=0) -> identical to base PI
        states = _batch["ricl_retrieved_state"]
        actions = _batch["ricl_retrieved_action"]
        valid = _batch.get("ricl_retrieved_mask")
        raw = self._raw_prompts(_batch, batch_size)
        spliced = []
        for i in range(batch_size):
            block = ricl_cond.build_retrieved_prompt_block(
                states[i],
                actions[i],
                valid[i] if valid is not None else None,
                prompt=raw[i],
                num_bins=self.state_num_bins,
                action_steps=self.retrieved_action_steps,
            )
            spliced.append(ricl_cond.splice_retrieved_into_prompt(prompts[i], block))
        return spliced

    # ------------------------------------------------------------------
    # Append retrieved images to the observation image dict
    # ------------------------------------------------------------------
    @override
    def _robomimic_to_pi_data(
        self, batch, cam_keys, proprio_keys, lang_keys, ac_key, embodiment
    ):
        obs, action32 = super()._robomimic_to_pi_data(
            batch, cam_keys, proprio_keys, lang_keys, ac_key, embodiment
        )
        if "ricl_retrieved_images" in batch:
            image_resolution = getattr(self, "image_resolution", (224, 224))
            ricl_cond.augment_images_with_retrieved(
                obs.images,
                obs.image_masks,
                batch["ricl_retrieved_images"],
                batch.get("ricl_retrieved_mask"),
                image_resolution=image_resolution,
                base_key=self.ricl_base_key,
            )
        return obs, action32
