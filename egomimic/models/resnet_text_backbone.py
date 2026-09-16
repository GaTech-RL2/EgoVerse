"""ResNet + frozen Qwen3-Embedding backbone for FlowVLA.

Produces the ``(contexts, context_mask)`` pair ``LayerwiseFMHead`` consumes: one
ResNet per canonical camera role and one frozen text encoder, joined into a
single context sequence that every DiT block cross-attends to. Design:
docs/hpt-experiments/2026-09-16_flowvla_resnet_text_design.md.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from egomimic.models.hpt_nets import ResNet


class ResNetTextBackbone(nn.Module):
    """One ResNet per camera role + a text encoder -> one shared context.

    Args:
        camera_roles: canonical camera keys, in the order
            ``FlowVLA._to_model_data`` stacks them (the embodiment's
            ``camera_keys``). Keying the encoders by role rather than by slot
            index is what lets two embodiments share the module for a role they
            both have (``Embodiment.VIZ_IMAGE_KEY`` is the same string for
            every vendor).
        text_encoder: a ``QwenPerTokenEncoder``-like module exposing
            ``forward_with_mask(prompts) -> ((B, L, output_dim), (B, L) bool)``
            with ``True`` marking real tokens.
        hidden_size: context width. Every modality is projected to it.
        num_layers: how many DiT blocks the head builds. The same context is
            returned once per block; the head's per-layer projectors are what
            give each block its own view of it, so they must be real Linears
            (i.e. the head's ``dit_hidden`` must differ from ``hidden_size``,
            or they collapse to ``nn.Identity``).
    """

    def __init__(
        self,
        camera_roles: List[str],
        text_encoder: nn.Module,
        hidden_size: int = 1024,
        num_layers: int = 12,
        resnet_model: str = "resnet18",
        weights: Optional[str] = "DEFAULT",
        freeze_backbone: bool = False,
        modality_embed: bool = True,
    ) -> None:
        super().__init__()
        camera_roles = list(camera_roles)
        if not camera_roles:
            raise ValueError("camera_roles must name at least one camera")
        if len(set(camera_roles)) != len(camera_roles):
            raise ValueError(f"camera_roles has duplicate entries: {camera_roles}")
        self.camera_roles = camera_roles
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")

        self.image_encoders = nn.ModuleDict(
            {
                self._role_key(role): ResNet(
                    output_dim=self.hidden_size,
                    resnet_model=resnet_model,
                    weights=weights,
                    num_of_copy=1,
                    freeze_backbone=freeze_backbone,
                )
                for role in self.camera_roles
            }
        )
        self.text_encoder = text_encoder
        text_dim = int(getattr(text_encoder, "output_dim", self.hidden_size))
        if text_dim != self.hidden_size:
            raise ValueError(
                f"text_encoder.output_dim is {text_dim} but the backbone's "
                f"hidden_size is {self.hidden_size}; set them equal"
            )
        # One row per camera role plus one for text, so a patch token is
        # distinguishable from a word token. Zero-init: inert at step 0.
        self.modality_embed = (
            nn.Parameter(torch.zeros(len(self.camera_roles) + 1, self.hidden_size))
            if modality_embed
            else None
        )

    @staticmethod
    def _role_key(role: str) -> str:
        """``nn.ModuleDict`` keys cannot contain '.', and camera roles are dotted."""
        return role.replace(".", "_")

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """(B, F, 3, H, W) -> (B, F * tokens_per_camera, hidden_size)."""
        if images.ndim != 5:
            raise ValueError(
                f"images must be (B, F, 3, H, W), got {tuple(images.shape)}"
            )
        if images.shape[1] != len(self.camera_roles):
            raise ValueError(
                f"images carry {images.shape[1]} cameras but the backbone was "
                f"built for {len(self.camera_roles)} cameras: {self.camera_roles}"
            )
        tokens = []
        for index, role in enumerate(self.camera_roles):
            # One frame per call: ResNet.forward's reshape is only
            # channel-correct at F == 1 (hpt_nets.py:940).
            feature = self.image_encoders[self._role_key(role)](
                images[:, index : index + 1]
            )
            if self.modality_embed is not None:
                feature = feature + self.modality_embed[index]
            tokens.append(feature)
        return torch.cat(tokens, dim=1)

    def forward(
        self, images: torch.Tensor, prompts: List[str]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """-> (``num_layers`` references to one context, its attend mask)."""
        image_tokens = self._encode_images(images)
        text_tokens, text_mask = self.text_encoder.forward_with_mask(prompts)
        text_tokens = text_tokens.to(
            dtype=image_tokens.dtype, device=image_tokens.device
        )
        if self.modality_embed is not None:
            text_tokens = text_tokens + self.modality_embed[-1]
        context = torch.cat([image_tokens, text_tokens], dim=1)
        image_mask = torch.ones(
            image_tokens.shape[:2], dtype=torch.bool, device=context.device
        )
        mask = torch.cat([image_mask, text_mask.bool().to(context.device)], dim=1)
        # One reference per DiT block: the head's per-layer projectors give each
        # block its own view.
        return [context] * self.num_layers, mask

    def backbone_parameters(self) -> List[nn.Parameter]:
        """The ResNet trunks, for ``ModelWrapper._backbone_param_groups``.

        The projections stay in the main LR group, as they do for HPT, and the
        text encoder is frozen so it never reaches an optimizer group.
        """
        params: List[nn.Parameter] = []
        for encoder in self.image_encoders.values():
            params.extend(encoder.backbone_parameters())
        return params
