"""``VAE`` algo — image autoencoder pre-training stage.

Wraps ``ImageVAE`` so it plugs into the existing trainHydra / Lightning
pipeline. No norm_stats interactions, no embodiment-specific routing,
no obs/action concept — purely "give me images, I'll reconstruct them".

Training step:
  * Pull the image stream from the packed batch
    (``observations.images.<image_key>`` -> ``self.image_key``).
  * For each frame: forward through VAE -> (x_rec, mu, logvar).
  * Loss = MSE(x, x_rec) + beta * KL(N(mu, sigma) || N(0, I)). Reported
    as ``recon_loss``, ``kl_loss``, ``vae_loss``.

Eval step: same forward, returns reconstructions for the recon
evaluator (``egomimic/eval/eval_vae_recon.py``) to write a comparison
mp4.

After training, the saved Lightning checkpoint contains the VAE's
weights under ``nets["vae"]``. ``load_pretrained_vae(path)`` is the
helper used by ObsActionImageDFoTOuterStage to load the frozen
encoder/decoder at policy-training time.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from overrides import override

from egomimic.algo.algo import Algo
from egomimic.models.diffusion.image_vae import ImageVAE
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


class VAE(Algo):
    """Image VAE training algo.

    Args:
        vae: ``ImageVAE`` instance (built by Hydra).
        image_key: which obs key carries the image stream (default
            ``"front_img_1"``).
        beta: KL weight in the ELBO. Default 1.0 (standard VAE). Drop
            to ~1e-3 for lower-KL "near-AE" behavior if the latent
            collapses.
        domains: embodiment-name list (for batch routing parity with
            the other algos).
        device: optional device override.
    """

    def __init__(
        self,
        vae: ImageVAE,
        norm_stats=None,
        image_key: str = "front_img_1",
        beta: float = 1.0,
        lpips_weight: float = 0.0,
        lpips_net: str = "alex",
        domains: Optional[list] = None,
        device=None,
        **kwargs,
    ):
        super().__init__()
        self.image_key = str(image_key)
        self.beta = float(beta)
        self.lpips_weight = float(lpips_weight)
        self.lpips_net = str(lpips_net)
        self.norm_stats = norm_stats
        self.domains = list(domains or [])
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # LPIPS perceptual loss. Built lazily and frozen — its job is to
        # measure feature-space distance via a small pretrained net
        # (alex / vgg / squeeze), which is much more edge-sensitive than
        # per-pixel MSE. Inputs MUST be in [-1, 1] before being passed in.
        self._lpips = None
        if self.lpips_weight > 0:
            try:
                import lpips

                self._lpips = lpips.LPIPS(net=self.lpips_net).eval()
                for p in self._lpips.parameters():
                    p.requires_grad = False
            except ImportError as e:
                raise ImportError(
                    "LPIPS requested (lpips_weight > 0) but the `lpips` "
                    "package isn't installed. uv pip install lpips."
                ) from e
        modules = {"vae": vae}
        if self._lpips is not None:
            modules["lpips"] = self._lpips
        self.nets = nn.ModuleDict(modules).float().to(self.device)
        # Mirror the other algos' embodiment routing so PL batches
        # arriving as {emb_name: {...}} work without surprise. We only
        # care about per-image data here, so the rest is trivial.
        self.embodiment_ids = {emb: get_embodiment_id(emb) for emb in self.domains}

    # ---- Algo API ---------------------------------------------------- #

    @override
    def process_batch_for_training(self, batch):
        """Convert the loader's {emb_name: ...} batch into
        {emb_id: {'images': (B, ...)}}.

        Accepts both packed (image: (T_total, C, H, W)) and padded
        (image: (B, T, C, H, W)) layouts. Output is always a flat
        (N, C, H, W) tensor on device so the VAE forward is one big
        batched call.
        """
        out = {}
        for emb_name, _batch in batch.items():
            emb_id = get_embodiment_id(emb_name)
            # Look up the image tensor by zarr key first (raw batch),
            # then by friendly key (already remapped batches).
            image = None
            for cand in (self.image_key, f"observations.images.{self.image_key}"):
                if cand in _batch:
                    image = _batch[cand]
                    break
            if image is None:
                continue
            if image.dim() == 4:
                # Packed: (T_total, C, H, W) -> already flat per-frame.
                imgs = image
            elif image.dim() == 5:
                # Padded: (B, T, C, H, W) -> collapse the (B, T) prefix.
                B, T = image.shape[:2]
                imgs = image.reshape(B * T, *image.shape[2:])
            else:
                raise ValueError(
                    f"image '{self.image_key}' has unexpected dim {image.dim()}; "
                    f"expected 4 (packed) or 5 (padded)."
                )
            imgs = imgs.to(self.device).float()
            # Robomimic image floats are in [0,1] already; if a future
            # data path delivers uint8 we'd /255 here. Defensive guard:
            if imgs.max() > 1.5:
                imgs = imgs / 255.0
            out[emb_id] = {"images": imgs}
        return out

    @override
    def forward_training(self, batch):
        """One forward pass per embodiment -> per-key loss tensors."""
        predictions = OrderedDict()
        for emb_id, _batch in batch.items():
            x = _batch["images"]
            x_rec, mu, logvar = self.nets["vae"](x)
            loss, recon, kl = ImageVAE.vae_loss(
                x,
                x_rec,
                mu,
                logvar,
                beta=self.beta,
            )
            # Add LPIPS perceptual loss if configured. LPIPS expects
            # inputs in [-1, 1]; our images are in [0, 1], so rescale.
            lpips_term = torch.zeros((), device=x.device, dtype=x.dtype)
            if self._lpips is not None:
                x_lp = x * 2.0 - 1.0
                xr_lp = x_rec * 2.0 - 1.0
                lpips_term = self.nets["lpips"](x_lp, xr_lp).mean()
                loss = loss + self.lpips_weight * lpips_term
            predictions[f"{emb_id}_recon_loss"] = recon
            predictions[f"{emb_id}_kl_loss"] = kl
            predictions[f"{emb_id}_lpips_loss"] = lpips_term.detach()
            predictions[f"{emb_id}_vae_loss"] = loss
            predictions[f"{emb_id}_x_rec"] = x_rec.detach()
            predictions[f"{emb_id}_x_gt"] = x.detach()
        return predictions

    @override
    def forward_eval(self, batch):
        """Same forward as training, but no grads. Returns (x_gt, x_rec)
        per embodiment under keys ``emb{id}_x_gt`` / ``emb{id}_x_rec``.
        """
        out = {}
        with torch.no_grad():
            for emb_id, _batch in batch.items():
                x = _batch["images"]
                mu, logvar = self.nets["vae"].encode(x)
                # Eval-time: use the posterior mean (no noise) for cleaner
                # reconstructions in the recon video.
                x_rec = self.nets["vae"].decode(mu)
                out[f"emb{emb_id}_x_gt"] = x.detach()
                out[f"emb{emb_id}_x_rec"] = x_rec.detach()
        return out

    @override
    def compute_losses(self, predictions, batch):
        total = torch.tensor(0.0, device=self.device)
        loss_dict = OrderedDict()
        for emb_id in batch.keys():
            r = predictions[f"{emb_id}_recon_loss"]
            k = predictions[f"{emb_id}_kl_loss"]
            total_emb = predictions[f"{emb_id}_vae_loss"]
            loss_dict[f"emb{emb_id}_recon_loss"] = r
            loss_dict[f"emb{emb_id}_kl_loss"] = k
            loss_dict[f"emb{emb_id}_vae_loss"] = total_emb
            if f"{emb_id}_lpips_loss" in predictions:
                loss_dict[f"emb{emb_id}_lpips_loss"] = predictions[
                    f"{emb_id}_lpips_loss"
                ]
            total = total + total_emb
        # Use "action_loss" key name for back-compat with the trainer's
        # progress-bar / logging hook that reads losses["action_loss"]
        # as the primary scalar.
        loss_dict["action_loss"] = total / max(len(batch), 1)
        return loss_dict

    @override
    def log_info(self, info):
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for k, v in info["losses"].items():
            log[k] = v.item()
        return log


def load_pretrained_vae(
    checkpoint_path: str,
    map_location: str = "cpu",
    **vae_kwargs,
) -> ImageVAE:
    """Load a frozen VAE from a Lightning checkpoint produced by training
    this algo. Returns the inner ``ImageVAE`` with all params
    ``requires_grad=False`` and the module in eval mode.

    Path convention: pass the Lightning .ckpt file. The state dict prefix
    is ``robomimic_model.nets.vae.*``.

    Architecture detection: shapes are sniffed from the state dict for
    common combos (flat vs spatial latent; residual vs plain; channel
    widths). You can pass explicit ``vae_kwargs`` to override the
    inference if the model was built with a non-default config.
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    # Different Lightning wrappers stick the VAE under different prefixes
    # ("robomimic_model.nets.vae." for the trainHydra+ModelWrapper path,
    # or just "nets.vae." for older PL wrappers). Try both in priority
    # order; first non-empty match wins.
    vae_state = {}
    for prefix in ("robomimic_model.nets.vae.", "nets.vae.", "vae."):
        candidate = {
            k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)
        }
        if candidate:
            vae_state = candidate
            break
    if not vae_state:
        raise RuntimeError(
            f"No keys with any known VAE prefix in {checkpoint_path}; "
            f"sample keys: {list(state.keys())[:5]}"
        )

    # Sniff architecture (overridden by anything passed in vae_kwargs).
    spatial_latent = "to_mu.weight" in vae_state
    residual = (
        "encoder.0.conv1.weight" in vae_state or "encoder.0.skip.weight" in vae_state
    )
    if spatial_latent:
        to_mu_w = vae_state["to_mu.weight"]  # (latent_c, bottleneck_c, 1, 1)
        latent_channels = int(to_mu_w.shape[0])
        bottleneck_c = int(to_mu_w.shape[1])
        latent_dim = 64  # unused in spatial mode
    else:
        fc_mu_w = vae_state["fc_mu.weight"]
        latent_dim = int(fc_mu_w.shape[0])
        flat_dim = int(fc_mu_w.shape[1])
        bottleneck_size_inferred = 6
        bottleneck_c = flat_dim // (bottleneck_size_inferred**2)
        latent_channels = 4  # unused in flat mode

    # Encoder channel widths from the state dict.
    if residual:
        # _ResBlock has conv1 + conv2; encoder.{i}.conv1.weight has shape
        # (c_out, c_in, 3, 3).
        c_keys = sorted(
            [k for k in vae_state if k.startswith("encoder.") and "conv1.weight" in k]
        )
        channels = tuple(int(vae_state[k].shape[0]) for k in c_keys)
    else:
        c_keys = sorted(
            [k for k in vae_state if k.startswith("encoder.") and ".conv.weight" in k]
        )
        channels = tuple(int(vae_state[k].shape[0]) for k in c_keys)
    if not channels:
        channels = (32, 64, 128, int(bottleneck_c))

    # decoder_upsample: only matters for non-residual builds. Detect from
    # the presence of a stride-1 stand-alone Conv2d vs ConvTranspose in
    # the last decoder block by looking for ``decoder.<last>.weight``.
    decoder_upsample = "nearest" if residual else "transpose"

    defaults = dict(
        in_channels=3,
        image_size=96,
        latent_dim=latent_dim,
        channels=channels,
        decoder_upsample=decoder_upsample,
        residual=residual,
        spatial_latent=spatial_latent,
        latent_channels=latent_channels,
    )
    defaults.update(vae_kwargs)
    vae = ImageVAE(**defaults)
    vae.load_state_dict(vae_state, strict=True)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae
