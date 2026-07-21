"""Template for plugging YOUR vision encoder into the RBY1 whole-body policy.

THE CONTRACT (what the policy expects from any image encoder)
=============================================================
Input  : torch.Tensor of shape [B, T, N, 3, H, W]
         - B batch, T horizon (always 1 here), N views (always 1 here), H=W=224
         - values are ALREADY ImageNet-normalized (mean/std applied upstream in
           `hpt.py:_robomimic_to_hpt_data` via train/eval_image_augs) -> roughly
           in [-2.1, +2.6]. Do NOT re-normalize inside the encoder.
Output : torch.Tensor of shape [B, M, output_dim]
         - M = any number of "spatial tokens" (ResNet-18 emits 49 = 7x7;
           DINOv2 ViT-S/14 emits 256 = 16x16). More tokens = finer spatial detail
           for the policy's cross-attention to read.
         - output_dim MUST equal the image stem's input_dim in the model YAML
           (256 in all our configs). A sinusoid positional embedding is ADDED to
           your tokens downstream, sized dynamically from your output shape.

Wiring: reference implementations live in `egomimic/models/hpt_nets.py` —
`ResNet` (fully fine-tuned CNN) and `DINOv2` (frozen ViT + optional LoRA /
transformer-neck / residual-MLP-head capacity knobs). Copy whichever pattern is
closer to your idea.

Freezing: if you freeze part of your backbone, wrap its forward in
`torch.set_grad_enabled(...)` correctly (see DINOv2.forward — adapters INSIDE a
frozen backbone still need gradients flowing through activations) and re-assert
`.eval()` on frozen submodules in `train()` (the Lightning loop calls `.train()`
every epoch). The optimizer receives all params; frozen ones simply get no grads.

Register your encoder in the model YAML:
    encoder_specs:
      front_img_1:
        _target_: egomimic.models.custom_encoders.MyEncoder
        output_dim: 256
        # ... your kwargs
"""

import importlib.util
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.hpt_nets import PolicyStem


class MyEncoder(PolicyStem):
    """Skeleton encoder — replace the backbone with your model.

    This minimal example: a small conv stack -> 14x14 = 196 tokens of dim
    `output_dim`. It trains end-to-end and satisfies the contract; it will NOT
    reach the reference numbers — it exists so the pipeline runs before your
    real encoder is ready.
    """

    def __init__(self, output_dim: int = 256, **kwargs) -> None:
        super().__init__(**kwargs)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2), nn.GELU(),    # 224 -> 112
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),  # 112 -> 56
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GELU(), # 56 -> 28
            nn.Conv2d(256, 256, 3, stride=2, padding=1),            # 28 -> 14
        )
        self.proj = nn.Linear(256, output_dim)
        self.out_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, N, 3, H, W] (ImageNet-normalized) -> [B, M, output_dim]."""
        B, *_, H, W = x.shape
        x = x.reshape(-1, 3, H, W)                    # fold T,N into batch
        feat = self.backbone(x)                       # (B*, C, h, w)
        feat = feat.flatten(2).transpose(1, 2)        # (B*, h*w, C) = tokens
        feat = feat.reshape(B, -1, feat.shape[-1])    # unfold T,N into tokens
        return self.proj(feat)                        # (B, M, output_dim)


# Default NVS-3D asset directory (overridable via the NVS3D_DIR env var). The
# directory must contain BOTH `model.py` (architecture) and the `.pt` weights.
_NVS3D_DIR = os.environ.get(
    "NVS3D_DIR", "/storage/project/r-agarg35-0/shared/3dlfv/checkpoints"
)


def _load_nvs3d_module(asset_dir: str):
    """Import the NVS-3D `model.py` living next to the weights file.

    We deliberately avoid `sys.path.insert + import model`: a bare module named
    `model` is too collision-prone inside a large codebase. The module is
    registered in sys.modules so repeated instantiation reuses it.
    """
    import sys

    if "nvs3d_model" in sys.modules:
        return sys.modules["nvs3d_model"]
    spec = importlib.util.spec_from_file_location(
        "nvs3d_model", os.path.join(asset_dir, "model.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["nvs3d_model"] = module
    spec.loader.exec_module(module)
    return module


# Cold-load support: Lightning checkpoints pickle the encoder object graph, and
# the frozen net's classes live in the dynamically-loaded module 'nvs3d_model'.
# Unpickling imports THIS module first (for NVS3DEncoder), so registering
# nvs3d_model here lets ModelWrapper.load_from_checkpoint work without any
# pre-import. NVS3D_DIR env wins; the flash7 staging dir is the cluster default.
for _cand in (_NVS3D_DIR, "/coc/flash7/czhang883/pretrained/nvs3d"):
    if os.path.isfile(os.path.join(_cand, "model.py")):
        try:
            _load_nvs3d_module(_cand)
        except Exception:
            pass
        break


class NVS3DEncoder(PolicyStem):
    """Frozen NVS-3D (multiview 3D-aware ViT) + trainable linear projection.

    Satisfies the contract at the top of this file: consumes ImageNet-normalized
    [B, T, N, 3, H, W] and returns [B, M, output_dim] with M = T*N*px*py patch
    tokens (256/image at the default 16x16 grid).

    Implementation notes (each one is load-bearing):
      * The NVS-3D `Model._extract` applies ImageNet normalization INTERNALLY and
        expects [0, 1] input, but this pipeline normalizes upstream (train/
        eval_image_augs). We therefore INVERT the upstream normalization before
        calling the model to avoid double normalization.
      * The model requires H = W = px*16 (256 by default); RBY1 aria frames are
        224x224, so we bilinearly resize.
      * The backbone's rotary position embeddings misbehave in bf16 (NaNs; same
        finding as the imitation-repo integration of this model), and the trainer
        default is `precision: bf16`. We disable autocast and run the backbone in
        float32; only the tiny projection sees autocast.
      * All NVS-3D weights (including the DINOv3 featurizer) come from the .pt
        checkpoint — no HuggingFace download at init. timm just needs to be new
        enough to know the `vit_base_patch16_dinov3` architecture.
    """

    def __init__(
        self,
        output_dim: int = 256,
        checkpoint_path: str = os.path.join(_NVS3D_DIR, "0981at.pt"),
        px: int = 16,
        py: int = 16,
        freeze_backbone: bool = True,
        conv_neck_blocks: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # If the configured checkpoint path doesn't exist (e.g. running on a
        # different machine), fall back to NVS3D_DIR/<filename>. This lets a
        # collaborator serve a trained ckpt by setting one env var, without
        # editing the hparams pickled inside the Lightning checkpoint.
        if not os.path.isfile(checkpoint_path):
            fallback = os.path.join(_NVS3D_DIR, os.path.basename(checkpoint_path))
            if os.path.isfile(fallback):
                print(
                    f"[NVS3DEncoder] checkpoint_path {checkpoint_path} not found; "
                    f"using {fallback} (from NVS3D_DIR)"
                )
                checkpoint_path = fallback
        asset_dir = os.path.dirname(checkpoint_path)
        nvs3d = _load_nvs3d_module(asset_dir)
        self.net = nvs3d.load_full(checkpoint_path, px=px, py=py, device="cpu")
        self.img_size = px * 16

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.net.parameters():
                p.requires_grad = False
            self.net.eval()

        model_dim = self.net.backbone.norm.weight.shape[0]
        if conv_neck_blocks > 0:
            # capacity knob: ConvNeck consumes (B, N, C) square-grid tokens, same
            # contract as the Linear it replaces (9 blocks ~= ResNet-18 budget)
            from egomimic.models.hpt_nets import ConvNeck

            self.proj = ConvNeck(model_dim, output_dim, conv_neck_blocks)
        else:
            self.proj = nn.Linear(model_dim, output_dim)
        self.out_dim = output_dim

        # ImageNet stats used by train/eval_image_augs upstream; needed to map
        # the input back to [0, 1] for the model's own internal normalization.
        self.register_buffer(
            "_inv_mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_inv_sg", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        n_total = sum(p.numel() for p in self.net.parameters())
        n_train = sum(p.numel() for p in self.proj.parameters())
        print(
            f"[NVS3DEncoder] loaded {checkpoint_path}: model_dim={model_dim}, "
            f"{n_total:,} frozen backbone params, {n_train:,} trainable proj params, "
            f"{px * py} tokens/image at {self.img_size}x{self.img_size}"
        )

    def train(self, mode: bool = True):
        # Lightning calls .train() every epoch; keep the frozen backbone in eval
        # so e.g. attention dropout stays off (same pattern as hpt_nets.DINOv2).
        super().train(mode)
        if self.freeze_backbone:
            self.net.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, N, 3, H, W] (ImageNet-normalized) -> [B, T*N*px*py, output_dim]."""
        B, *_, H, W = x.shape
        x = x.reshape(B, -1, 3, H, W)  # (B, K, 3, H, W): T*N views for NVS-3D
        K = x.shape[1]

        x = x.reshape(B * K, 3, H, W).float()
        x = x * self._inv_sg + self._inv_mu       # undo upstream ImageNet norm
        x = x.clamp_(0.0, 1.0)
        if H != self.img_size or W != self.img_size:
            x = F.interpolate(
                x, size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            )
        x = x.reshape(B, K, 3, self.img_size, self.img_size)

        backbone_grad = torch.is_grad_enabled() and not self.freeze_backbone
        with torch.set_grad_enabled(backbone_grad):
            # Force fp32: rotary embeddings NaN under bf16 autocast.
            with torch.autocast(device_type=x.device.type, enabled=False):
                feat = self.net(x)                # (B, K*px*py, model_dim)

        return self.proj(feat)                    # (B, M, output_dim)
