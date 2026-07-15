"""Image VisualCore for BC-RNN (robomimic faithful port), kept inside the
BC-RNN package so this codebase needs no edits to its own image encoders.

PROVENANCE / PORT NOTE
----------------------
``SpatialSoftmax`` + ``VisualCore`` were copied VERBATIM from EgoVerse2's
``egomimic/models/hnet_nets/image_encoders.py`` (branch hpt-hnet-pusher-nc3)
during the BC-RNN port into EgoVerse-pact-2. The EgoVerse2 obs encoder reaches
this class via the config ``_target_:
egomimic.models.hnet_nets.image_encoders.VisualCore``. EgoVerse-pact diverged
from EgoVerse2 around 2026-05-18 and its own
``hnet_nets/image_encoders.py`` carries ONLY ``SimpleConv`` — it has no
``VisualCore`` / ``SpatialSoftmax`` / ``ResNetEncoder``. Rather than splice the
class (and its closure) into a diverged foreign file, VisualCore lives here and
the BC-RNN model configs point their ``front_img_1._target_`` at
``egomimic.models.stems.visual_core.VisualCore`` (its role home). This keeps
VisualCore self-contained and touches ZERO of pact-2's existing classes.

Shape contract (matches SimpleConv): ``(..., C, H, W) -> (..., embed_dim)`` and
exposes ``.embed_dim`` (== feature_dimension), so ``ObsEncoder`` can size the
fusion / concat without inspecting weights.
"""

import torch
import torch.nn as nn


class SpatialSoftmax(nn.Module):
    """Spatial-softmax keypoint pooling — faithful port of robomimic's
    ``robomimic.models.base_nets.SpatialSoftmax`` (Finn et al. DSAE).

    Takes a conv feature map ``(N, C, H, W)`` and, per (optionally 1x1-conv
    re-projected) channel, builds a 2D spatial softmax over the H*W pixel grid,
    then returns the expected (x, y) pixel coordinate of each of ``num_kp``
    keypoints -> ``(N, num_kp, 2)``.

    robomimic source (base_nets.py), quoted semantics matched line-for-line:
        self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
        pos_x, pos_y = np.meshgrid(np.linspace(-1.,1.,W), np.linspace(-1.,1.,H))
        feature = feature.reshape(-1, H * W)
        attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)
    """

    def __init__(
        self,
        in_channels: int,
        in_h: int,
        in_w: int,
        num_kp: int = 32,
        temperature: float = 1.0,
    ):
        super().__init__()
        self._in_c = int(in_channels)
        self._in_h = int(in_h)
        self._in_w = int(in_w)
        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, int(num_kp), kernel_size=1)
            self._num_kp = int(num_kp)
        else:
            self.nets = None
            self._num_kp = self._in_c
        # constant temperature (robomimic learnable_temperature default False)
        self.register_buffer("temperature", torch.ones(1) * float(temperature))
        import numpy as _np

        pos_x, pos_y = _np.meshgrid(
            _np.linspace(-1.0, 1.0, self._in_w),
            _np.linspace(-1.0, 1.0, self._in_h),
        )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    @property
    def num_kp(self) -> int:
        return self._num_kp

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        # feature: (N, C, H, W) -> (N, num_kp, 2)
        if self.nets is not None:
            feature = self.nets(feature)
        feature = feature.reshape(-1, self._in_h * self._in_w)
        attention = torch.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)
        return feature_keypoints


class VisualCore(nn.Module):
    """robomimic ``VisualCore`` for per-frame image obs — faithful port.

    Structure (robomimic obs_core.VisualCore defaults, bc.json empty
    core_kwargs => these defaults apply):
        ResNet18Conv backbone  -> (N, 512, H/32, W/32)
        SpatialSoftmax(num_kp) -> (N, num_kp, 2)
        Flatten                -> (N, num_kp*2)
        Linear(feature_dimension=64)

    The ResNet18 conv backbone is ``nn.Sequential(*list(resnet18.children())
    [:-2])`` — byte-identical to robomimic's ``ResNet18Conv`` and to the
    existing ``hpt_nets.ResNet.net``. ``pretrained`` defaults to False to match
    robomimic's ResNet18Conv default (bc.json passes no backbone_kwargs).

    Shape contract (matches SimpleConv): ``(..., C, H, W) ->
    (..., embed_dim)`` and exposes ``.embed_dim`` (== feature_dimension).
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 96,
        num_kp: int = 32,
        feature_dimension: int = 64,
        pretrained: bool = False,
        resnet_model: str = "resnet18",
        crop_aug: bool = False,
        crop_height: int = None,
        crop_width: int = None,
        crop_eval_mode: str = "center",
        crop_sample_mode: str = "inclusive",
        norm_layer: str = "batch",
    ):
        super().__init__()
        import torchvision

        # norm_layer: "batch" (default, byte-identical to before this knob) or
        # "group" -> every BatchNorm2d in the backbone is replaced with
        # GroupNorm(C//16, C) (diffusion-policy's obs_encoder_group_norm swap).
        # Required when training with EMACallback: EMA-averaged conv weights +
        # live BN running stats mismatch; GroupNorm has no running stats.
        if norm_layer not in ("batch", "group"):
            raise ValueError(f"norm_layer must be batch|group, got {norm_layer!r}")
        self.norm_layer = str(norm_layer)

        # crop_eval_mode (robomimic CropRandomizer.forward_in, v0.2
        # base_nets.py:1351): v0.2 crops RANDOMLY and UNCONDITIONALLY -- there is
        # NO train/eval branch, so paper rollouts saw stochastic crops at eval.
        #   "center" (default) -> deterministic center crop at eval (our current
        #                         behavior; byte-identical to before this knob).
        #   "random"           -> random crop at eval too (paper-exact).
        # crop_sample_mode (off-by-one): the valid top-left corner range.
        #   "inclusive" (default) -> corner in {0..(H-crop_h)} via randint
        #                            (our current behavior, INCLUSIVE upper).
        #   "v02"                 -> corner in {0..(H-crop_h)-1} via
        #                            floor(rand*(H-crop_h)), matching v0.2
        #                            obs_utils.py:677,686-687 EXACTLY.
        if crop_eval_mode not in ("center", "random"):
            raise ValueError(f"crop_eval_mode must be center|random, got {crop_eval_mode!r}")
        if crop_sample_mode not in ("inclusive", "v02"):
            raise ValueError(f"crop_sample_mode must be inclusive|v02, got {crop_sample_mode!r}")
        self.crop_eval_mode = str(crop_eval_mode)
        self.crop_sample_mode = str(crop_sample_mode)

        # robomimic CropRandomizer (generate_paper_configs.py + obs_core.py):
        # for EVERY image experiment robomimic random-crops crop_h x crop_w from
        # the input (76x76 from 84x84, ~90.5% of side, num_crops=1) -- random
        # position at TRAIN, deterministic center crop at EVAL -- BEFORE the
        # ResNet. Gated + default OFF => every existing config is byte-identical:
        # crop_aug=False takes the exact pre-existing code path (no crop, the
        # SpatialSoftmax grid is sized to the full `image_size`).
        self.crop_aug = bool(crop_aug)
        if self.crop_aug:
            if crop_height is None or crop_width is None:
                raise ValueError(
                    "VisualCore(crop_aug=True) requires crop_height and crop_width."
                )
            self.crop_height = int(crop_height)
            self.crop_width = int(crop_width)
            if self.crop_height > image_size or self.crop_width > image_size:
                raise ValueError(
                    f"crop ({self.crop_height}x{self.crop_width}) must fit in "
                    f"image_size {image_size}."
                )
            # SpatialSoftmax must be sized to the CROPPED feature map, so the
            # backbone-dummy below is run at the crop size, not image_size.
            backbone_size_h, backbone_size_w = self.crop_height, self.crop_width
        else:
            self.crop_height = None
            self.crop_width = None
            backbone_size_h = backbone_size_w = image_size

        weights = "DEFAULT" if pretrained else None
        pretrained_model = getattr(torchvision.models, resnet_model)(weights=weights)
        if in_channels != 3:
            pretrained_model.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        # ResNet18Conv backbone: cut avgpool + fc (last two children).
        self.backbone = nn.Sequential(*list(pretrained_model.children())[:-2])
        if self.norm_layer == "group":
            def _swap_bn(mod: nn.Module) -> None:
                for name, child in mod.named_children():
                    if isinstance(child, nn.BatchNorm2d):
                        c = child.num_features
                        setattr(mod, name, nn.GroupNorm(max(1, c // 16), c))
                    else:
                        _swap_bn(child)
            _swap_bn(self.backbone)
        with torch.no_grad():
            _f = self.backbone(
                torch.zeros(1, in_channels, backbone_size_h, backbone_size_w)
            )
            _, c, h, w = _f.shape
        self.pool = SpatialSoftmax(
            in_channels=c, in_h=h, in_w=w, num_kp=num_kp
        )
        self.feature_dimension = int(feature_dimension)
        self.head = nn.Linear(self.pool.num_kp * 2, self.feature_dimension)
        self.embed_dim = int(feature_dimension)

    def _crop(self, x: torch.Tensor) -> torch.Tensor:
        """robomimic CropRandomizer crop. num_crops=1.

        x: (N, C, H, W) -> (N, C, crop_h, crop_w). Random crop at TRAIN always;
        at EVAL random iff crop_eval_mode="random" else a centered crop
        ((H-crop_h)//2, (W-crop_w)//2). The random corner range is
        {0..(H-crop_h)} inclusive (crop_sample_mode="inclusive", default) or
        {0..(H-crop_h)-1} via floor(rand*(H-crop_h)) (crop_sample_mode="v02",
        matching v0.2 obs_utils.py:686-687 exactly).
        """
        N, C, H, W = x.shape
        ch, cw = self.crop_height, self.crop_width
        max_h, max_w = H - ch, W - cw
        # Random crop at TRAIN always; at EVAL only when crop_eval_mode="random".
        do_random = self.training or (self.crop_eval_mode == "random")
        if do_random:
            # robomimic samples a random crop position PER image in the batch.
            if self.crop_sample_mode == "v02":
                # v0.2 obs_utils.py:686-687: corner = floor(rand * max_sample),
                # i.e. in {0..max-1} (EXCLUSIVE upper) -- matches the paper.
                if max_h > 0:
                    h0 = (max_h * torch.rand(N, device=x.device)).long()
                else:
                    h0 = torch.zeros(N, dtype=torch.long, device=x.device)
                if max_w > 0:
                    w0 = (max_w * torch.rand(N, device=x.device)).long()
                else:
                    w0 = torch.zeros(N, dtype=torch.long, device=x.device)
            else:
                # inclusive (pre-existing): corner in {0..max} via randint.
                if max_h > 0:
                    h0 = torch.randint(0, max_h + 1, (N,), device=x.device)
                else:
                    h0 = torch.zeros(N, dtype=torch.long, device=x.device)
                if max_w > 0:
                    w0 = torch.randint(0, max_w + 1, (N,), device=x.device)
                else:
                    w0 = torch.zeros(N, dtype=torch.long, device=x.device)
            # gather per-sample crops (vectorized via advanced indexing on rows).
            rows = h0[:, None] + torch.arange(ch, device=x.device)[None, :]  # (N, ch)
            cols = w0[:, None] + torch.arange(cw, device=x.device)[None, :]  # (N, cw)
            bidx = torch.arange(N, device=x.device)[:, None, None]
            x = x[bidx, :, rows[:, :, None], cols[:, None, :]]  # (N, ch, cw, C)
            x = x.permute(0, 3, 1, 2).contiguous()              # (N, C, ch, cw)
        else:
            h0 = max_h // 2
            w0 = max_w // 2
            x = x[:, :, h0:h0 + ch, w0:w0 + cw]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C, H, W) -> (..., embed_dim)
        leading = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        if self.crop_aug:
            x = self._crop(x)              # (N, C, crop_h, crop_w)
        fmap = self.backbone(x)            # (N, 512, h, w)
        kp = self.pool(fmap)               # (N, num_kp, 2)
        feat = self.head(kp.flatten(1))    # (N, embed_dim)
        return feat.reshape(*leading, self.embed_dim)
