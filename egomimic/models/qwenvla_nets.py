"""QwenVLA: Qwen 3.5 backbone + layer-wise cross-DiT head (starVLA QwenPI port).

``QwenVLBackbone`` reuses ``Qwen35VLMEncoder``'s loading, freezing, dtype and
pretrained-weight bookkeeping and changes two things: the chat turn holds
every camera frame of the sample (``[image] * F + [text]``), and the forward
returns the last ``num_layers`` text-layer hidden states (one context per
DiT block) plus the padding mask, instead of the visual tokens of one layer.
State never enters the VLM (spec section 4): it goes to the head.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.hpt_nets import Qwen35VLMEncoder
from egomimic.models.layerwise_dit import LayerwiseFMHead


class QwenVLBackbone(Qwen35VLMEncoder):
    """Qwen 3.5 as the QwenVLA backbone.

    Args (beyond ``Qwen35VLMEncoder``): ``gradient_checkpointing`` is applied
    to the VLM whenever any of it trains; ``num_layers`` (default: the text
    stack's depth) is how many of the LAST hidden states are returned, i.e.
    the DiT depth. ``dtype="float32"`` for a fine-tune (Lightning's bf16
    autocast does the mixed precision; AdamW must see fp32 master weights);
    ``"bfloat16"`` is fine for a frozen backbone.
    """

    def __init__(
        self,
        model_name: str = Qwen35VLMEncoder.DEFAULT_MODEL,
        dtype: str = "float32",
        freeze: bool = False,
        trainable_layers: int = 0,
        gradient_checkpointing: bool = True,
        image_size: tuple = (352, 640),
        max_text_tokens: int = 128,
        num_layers: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            dtype=dtype,
            freeze=freeze,
            trainable_layers=trainable_layers,
            feature_layer=-1,
            image_size=image_size,
            max_text_tokens=max_text_tokens,
            **kwargs,
        )
        depth = int(self.model.config.text_config.num_hidden_layers)
        self.num_layers = depth if num_layers is None else int(num_layers)
        if not 1 <= self.num_layers <= depth:
            raise ValueError(
                f"num_layers={self.num_layers} but {self.model_name} has {depth} text layers"
            )
        self.gradient_checkpointing = bool(gradient_checkpointing) and not self._no_grad
        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def backbone_parameters(self) -> List[nn.Parameter]:
        """The VLM's parameters, for ``ModelWrapper._backbone_param_groups``."""
        return list(self.model.parameters())

    def _processor_inputs(self, images: torch.Tensor, prompts: List[str]) -> dict:
        """Chat layout ``[image] * F + [prompt]`` per sample; ``images`` is
        ``(B, F, 3, H, W)`` or ``(B, 3, H, W)`` (F = 1)."""
        if images.ndim == 4:
            images = images[:, None]
        batch, frames = images.shape[:2]
        if len(prompts) != batch:
            raise RuntimeError(
                f"QwenVLBackbone got {len(prompts)} prompts for {batch} samples"
            )
        tokenizer = self.processor.tokenizer
        texts, flat = [], []
        for i, prompt in enumerate(prompts):
            ids = tokenizer(str(prompt), add_special_tokens=False)["input_ids"]
            text = tokenizer.decode(ids[: self.max_text_tokens])
            content = [{"type": "image"}] * frames + [{"type": "text", "text": text}]
            texts.append(
                self.processor.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            flat.extend(list(images[i]))
        return self.processor(
            text=texts,
            images=flat,
            padding=True,
            return_tensors="pt",
            do_rescale=False,
            do_resize=False,
        )

    def _prepare_frames(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 4:
            images = images[:, None]
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(
                f"images must be (B, F, 3, H, W), got {tuple(images.shape)}"
            )
        batch, frames, _, height, width = images.shape
        images = images.float()
        if (height, width) != self.image_size:
            images = F.interpolate(
                images.flatten(0, 1),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).view(batch, frames, 3, *self.image_size)
        return images.clamp(0.0, 1.0)

    def forward(
        self, images: torch.Tensor, prompts: List[str]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """One VLM forward -> (last ``num_layers`` hidden states, padding mask)."""
        images = self._prepare_frames(images)
        frames = images.shape[1]
        inputs = self._processor_inputs(images, prompts)
        device = next(self.model.parameters()).device
        inputs = {
            key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in inputs.items()
        }
        if torch.is_tensor(inputs.get("pixel_values")):
            inputs["pixel_values"] = inputs["pixel_values"].to(self._load_dtype)

        def run():
            return self.model.model(
                **inputs, output_hidden_states=True, use_cache=False
            )

        if self._no_grad:
            with torch.no_grad():
                out = run()
        else:
            out = run()

        counts = (inputs["input_ids"] == self.image_token_id).sum(dim=1)
        expected = frames * self.num_visual_tokens
        if not bool((counts == expected).all()):
            raise RuntimeError(
                f"expected {expected} visual tokens per sample, got {counts.tolist()}"
            )
        hidden = list(out.hidden_states[-self.num_layers :])
        mask = inputs["attention_mask"].bool()
        return [h.float() for h in hidden], mask


class QwenVLAModel(nn.Module):
    """``backbone`` + ``head``; ``encoders`` is the plain dict
    ``ModelWrapper._backbone_param_groups`` reads for the VLM LR group."""

    def __init__(self, backbone: QwenVLBackbone, head: LayerwiseFMHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.encoders = {"vlm": backbone}

    def encode(self, data: dict):
        return self.backbone(data["images"], data["prompts"])

    def compute_loss(self, data: dict) -> torch.Tensor:
        contexts, mask = self.encode(data)
        return self.head.compute_loss(
            contexts,
            mask,
            data["action"],
            data["loss_mask"],
            data.get("state"),
            data["embodiment_name"],
        )

    def sample(
        self, data: dict, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        contexts, mask = self.encode(data)
        return self.head.sample(
            contexts,
            mask,
            data.get("state"),
            data["embodiment_name"],
            generator=generator,
        )
