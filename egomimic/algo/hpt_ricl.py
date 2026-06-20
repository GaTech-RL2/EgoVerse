"""HptRicl: HPT + retrieval-based in-context learning (RICL).

Ports the RICL idea from :class:`egomimic.algo.pi_ricl.PIRicl` (pi0.5) onto the
stem/trunk/head HPT model. Unlike pi0.5 (a unified VLM where retrieved demos are
just extra images + discretized prompt text), HPT encodes every modality with a
*separate stem*, so each retrieved demonstration's ``(image, state, action-chunk)``
is turned into tokens by **dedicated retrieved stems**:

- ``ricl_image``  : shares the query ResNet backbone (generic vision; retrieval is
  DINOv2-similarity based) but has its own cross-attention pooling head.
- ``ricl_state``  : a separate ``MLPPolicyStem`` (32-D shared-space input).
- ``ricl_action`` : a new :class:`~egomimic.models.hpt_nets.ActionChunkStem`
  (HPT otherwise never encodes an action chunk as input).

Fusion is **flat concatenation** (no trunk architecture surgery): the retrieved
token blocks are appended after the query stem tokens, tagged with learned
demo-index + modality embeddings, and invalid demos (``ricl_retrieved_mask``) are
both zeroed at the stem output and hidden from attention via a per-sample
``key_padding_mask`` threaded through the trunk. With no ``ricl_*`` keys present
(the k=0 zero-context floor), behaviour is identical to plain HPT.

The retrieval data pipeline (DINOv2 kNN, bank normalization + 32-D conversion,
the 5 ``ricl_*`` collate keys) is model-agnostic and reused as-is from
``egomimic/ricl`` via ``RiclDataModuleWrapper``.
"""

from __future__ import annotations

import logging

import torch
from overrides import override

from egomimic.algo.hpt import HPT, HPTModel
from egomimic.rldb.embodiment.embodiment import get_embodiment_id

logger = logging.getLogger(__name__)

# Keys the RICL collate attaches per query sample (see egomimic/ricl + the P3
# collate). Identical contract to egomimic.algo.pi_ricl.RICL_BATCH_KEYS.
RICL_BATCH_KEYS = (
    "ricl_retrieved_images",  # (B, k, C, H, W)
    "ricl_retrieved_state",  # (B, k, Ds)  normalized to the bank convention
    "ricl_retrieved_action",  # (B, k, Ha, Da)  normalized, converted to shared space
    "ricl_retrieved_mask",  # (B, k) bool, valid neighbor (handles < k)
    "ricl_retrieved_dist",  # (B, k) float, kNN distances (unused in v1)
)

# Order in which retrieved modalities are encoded / concatenated. The index into
# this tuple selects the per-modality learned embedding row.
_RICL_MODALITIES = ("ricl_image", "ricl_state", "ricl_action")


class HptRiclModel(HPTModel):
    """HPTModel that also encodes retrieved in-context demonstrations.

    Constructed via ``HPT.model_cls`` with the same ``trunk`` kwargs as HPTModel
    plus ``num_retrieved_observations`` (k) and ``ricl_image_encoder_key``.
    """

    def __init__(
        self,
        *args,
        num_retrieved_observations: int = 4,
        ricl_image_encoder_key: str = "front_img_1",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_retrieved = int(num_retrieved_observations)
        self.ricl_image_encoder_key = ricl_image_encoder_key
        D = self.embed_dim
        # Learned tags so the trunk can tell which demo / modality a token came
        # from. Zero-init => early training stays close to plain-HPT behaviour.
        self.ricl_demo_embed = torch.nn.Parameter(torch.zeros(self.num_retrieved, D))
        self.ricl_modality_embed = torch.nn.Parameter(
            torch.zeros(len(_RICL_MODALITIES), D)
        )
        # Stashed per-forward; read by HPTModel.forward_features (defaults to None
        # there, so plain HPT is unaffected).
        self._ricl_key_padding_mask = None

    # ------------------------------------------------------------------
    # Don't add the singleton instance axis to retrieved tensors: they carry a
    # leading k dim and are handled wholesale in stem_process. (Base only special-
    # cases keys containing "state"; "ricl_state" would match.)
    # ------------------------------------------------------------------
    @override
    def preprocess_states(self, domain, data):
        for key in data:
            if "state" in key and not key.startswith("ricl_"):
                data[key] = data[key][:, :, None]
        return data

    # ------------------------------------------------------------------
    # Encode query modalities (super), then the retrieved demos.
    # ------------------------------------------------------------------
    @override
    def stem_process(self, domain, data):
        ricl = {}
        for key in (*_RICL_MODALITIES, "ricl_retrieved_mask"):
            if key in data:
                ricl[key] = data.pop(key)

        # Query stems exactly as plain HPT (ricl keys removed, so the base loop's
        # `if modality not in data: continue` skips the retrieved stems).
        feats, feat_dict = super().stem_process(domain, data)

        self._ricl_key_padding_mask = None
        if "ricl_image" in ricl:
            ricl_tokens, kpm = self._process_retrieved(domain, ricl, feats)
            feats.append(ricl_tokens)
            feat_dict["ricl"] = ricl_tokens
            self._ricl_key_padding_mask = kpm

        return feats, feat_dict

    # ------------------------------------------------------------------
    def _encode_ricl_image(self, domain, images):
        """images: (B, k, C, H, W) -> (B*k, t_img, D) via shared ResNet + own head."""
        B, k = images.shape[0], images.shape[1]
        x = images.reshape(B * k, *images.shape[2:])  # (B*k, C, H, W)
        x = x.unsqueeze(1).unsqueeze(1)  # (B*k, 1, 1, C, H, W) — ResNet input shape
        feat = self.encoders[self.ricl_image_encoder_key](x)  # (B*k, M, D)
        stem = self.stems[f"{domain}_ricl_image"]
        return stem.compute_latent(feat)  # (B*k, t_img, D)

    def _process_retrieved(self, domain, ricl, query_feats):
        """Encode the 3 retrieved modalities, tag + mask them, and build the
        trunk key_padding_mask. Returns (ricl_tokens (B, sum_k_t, D), kpm (B, L))."""
        images = ricl["ricl_image"]  # (B, k, C, H, W)
        states = ricl["ricl_state"]  # (B, k, Ds)
        actions = ricl["ricl_action"]  # (B, k, Ha, Da)
        mask = ricl.get("ricl_retrieved_mask")  # (B, k) bool, True = valid

        B, k = images.shape[0], images.shape[1]
        assert k <= self.ricl_demo_embed.shape[0], (
            f"got k={k} retrieved demos but ricl_demo_embed sized for "
            f"{self.ricl_demo_embed.shape[0]} (set trunk.num_retrieved_observations)"
        )

        per_modality_tokens = [
            self._encode_ricl_image(domain, images),  # (B*k, t, D)
            self.stems[f"{domain}_ricl_state"].compute_latent(
                states.reshape(B * k, 1, states.shape[-1])
            ),
            self.stems[f"{domain}_ricl_action"].compute_latent(
                actions.reshape(B * k, actions.shape[2], actions.shape[3])
            ),
        ]

        if mask is not None:
            valid = mask.bool().view(B, k, 1, 1)  # True = valid
        blocks, mask_blocks = [], []
        for m_idx, tok in enumerate(per_modality_tokens):
            t, D = tok.shape[1], tok.shape[-1]
            tok = tok.view(B, k, t, D)
            tok = tok + self.ricl_demo_embed[:k].view(1, k, 1, D)
            tok = tok + self.ricl_modality_embed[m_idx].view(1, 1, 1, D)
            if mask is not None:
                tok = tok * valid.to(tok.dtype)  # zero invalid demo tokens
                # key_padding_mask convention: True = ignore.
                mask_blocks.append(
                    (~mask.bool()).view(B, k, 1).expand(B, k, t).reshape(B, k * t)
                )
            else:
                mask_blocks.append(
                    torch.zeros(B, k * t, dtype=torch.bool, device=tok.device)
                )
            blocks.append(tok.reshape(B, k * t, D))

        ricl_tokens = torch.cat(blocks, dim=1)  # (B, sum_k_t, D)

        # Prefix = learnable action queries (if any) + all query stem tokens, all
        # valid (never ignored). Order matches preprocess_tokens:
        # [action_tokens, *query_feats, ricl_tokens].
        n_query = sum(int(f.shape[1]) for f in query_feats)
        n_prefix = n_query + (
            self.action_horizon if self.token_postprocessing == "action_token" else 0
        )
        prefix = torch.zeros(B, n_prefix, dtype=torch.bool, device=ricl_tokens.device)
        kpm = torch.cat([prefix, *mask_blocks], dim=1)  # (B, L)
        return ricl_tokens, kpm


class HptRicl(HPT):
    """HPT with flat-concatenated retrieved in-context demonstrations."""

    model_cls = HptRiclModel

    def __init__(self, *args, num_retrieved_observations: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_retrieved_observations = int(num_retrieved_observations)
        logger.info("HptRicl: k=%d retrieved demos", self.num_retrieved_observations)

    # ------------------------------------------------------------------
    # Carry ricl_* keys through process_batch_for_training (mirror PIRicl).
    # ------------------------------------------------------------------
    @override
    def process_batch_for_training(self, batch):
        processed = super().process_batch_for_training(batch)
        for embodiment_name, _batch in batch.items():
            emb_id = get_embodiment_id(embodiment_name)
            if emb_id not in processed:
                continue
            # The base loop maps unknown keys (the ricl_* tensors) to keyname None
            # via zarr_key_to_keyname; drop that garbage slot, then re-add cleanly.
            processed[emb_id].pop(None, None)
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
    # Route retrieved tensors into the data dict under the new stem modality keys.
    # ------------------------------------------------------------------
    @override
    def _robomimic_to_hpt_data(
        self, batch, cam_keys, proprio_keys, lang_keys, ac_key, aux_ac_keys=[]
    ):
        data = super()._robomimic_to_hpt_data(
            batch, cam_keys, proprio_keys, lang_keys, ac_key, aux_ac_keys
        )
        if "ricl_retrieved_images" not in batch:
            return data  # zero-context (k=0) -> identical to base HPT

        imgs = batch["ricl_retrieved_images"]  # (B, k, C, H, W)
        B, k = imgs.shape[0], imgs.shape[1]
        flat = imgs.reshape(B * k, *imgs.shape[2:])
        # Normalize retrieved frames the same way as query frames (ImageNet stats
        # for the pretrained ResNet); invalid (zero) demos are masked downstream.
        if self.nets.training and self.train_image_augs is not None:
            flat = self.train_image_augs(flat)
        elif self.eval_image_augs is not None:
            flat = self.eval_image_augs(flat)
        data["ricl_image"] = flat.reshape(B, k, *flat.shape[1:])
        data["ricl_state"] = batch["ricl_retrieved_state"]
        data["ricl_action"] = batch["ricl_retrieved_action"]
        data["ricl_retrieved_mask"] = batch["ricl_retrieved_mask"]
        return data
