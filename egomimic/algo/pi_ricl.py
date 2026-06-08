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

import torch
from overrides import override

from egomimic.algo.pi import PI
from egomimic.ricl import conditioning as ricl_cond
from egomimic.rldb.embodiment.embodiment import get_embodiment_id

logger = logging.getLogger(__name__)

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
        spliced = []
        for i in range(batch_size):
            block = ricl_cond.build_retrieved_prompt_block(
                states[i],
                actions[i],
                valid[i] if valid is not None else None,
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
