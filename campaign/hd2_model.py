# Exact hd2 HPTModel class extracted to avoid unrelated dataset/AWS imports.
import os, math
from collections import OrderedDict
from typing import List, Optional
from functools import partial
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import hydra, einops
from egomimic.models.hpt_nets import *
from egomimic.utils.egomimicUtils import get_sinusoid_encoding_table, STD_SCALE, EinOpsRearrange, download_from_huggingface
_SINUSOID_TABLES={}
def _sinusoid_table(position_end,d_hid,device):
 key=(position_end,d_hid,str(device))
 if key not in _SINUSOID_TABLES:_SINUSOID_TABLES[key]=get_sinusoid_encoding_table(0,position_end,d_hid).to(device)
 return _SINUSOID_TABLES[key]
class HPTModel(nn.Module):
    """
    Heterogenous Pretrained Transformer (HPT) implementation based on the HPT paper, with additional modifications.
    This model integrates modality-specific stems, a transformer trunk, and domain-specific heads to process
    multi-modal data.
    """

    def __init__(
        self,
        embed_dim=1024,
        num_blocks=24,
        num_heads=16,
        token_postprocessing="action_token",
        observation_horizon=4,
        action_horizon=1,
        no_trunk=False,
        shared_modality_trunk=None,
        use_domain_embedding=False,
        drop_path=0.0,
        weight_init_style="pytorch",
        proprio_dropout_p=0.0,
        **kwargs,
    ):
        """
        Initialize the HPTModel.

        Parameters
        ----------
        embed_dim : int, optional
            Dimension of the token embeddings (default is 1024).
        num_blocks : int, optional
            Number of transformer blocks (default is 24).
        num_heads : int, optional
            Number of attention heads in each transformer block (default is 16).
        token_postprocessing : str, optional
            Strategy for postprocessing tokens. Options include "action_token", "mean", "max", "last", and "no-op"
            (default is "action_token").
        observation_horizon : int, optional
            Number of past observations to consider (default is 4).
        action_horizon : int, optional
            Number of action tokens to predict (default is 1).
        no_trunk : bool, optional
            If True, the transformer trunk is skipped (default is False).
        shared_modality_trunk : optional
            Shared trunk module for modality-specific processing if provided.
        use_domain_embedding : bool, optional
            Whether to use domain-specific embeddings (default is False).
        drop_path : float, optional
            Drop path rate for regularization (default is 0.0).
        weight_init_style : str, optional
            Weight initialization style (default is "pytorch").
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.shared_modality_trunk = shared_modality_trunk
        self.no_trunk = no_trunk

        self.encoders = nn.ModuleDict()

        self.trunk = self._create_policy_trunk(
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            drop_path=drop_path,
            weight_init_style=weight_init_style,
        )

        # --- Proprio augmentation ---
        # Both features are configured per-key via each stem's `specs.proprio_dropout_p` /
        # `specs.noise_std` in the model YAML. The trunk-level `proprio_dropout_p` acts as a
        # global fallback when a key has no per-key override.
        #
        # Augmentations are applied in preprocess_states() in this order:
        #   1. Gaussian noise  (always on kept samples)
        #   2. Dropout         (replaces the entire tensor with a learned null token)
        # Noise before dropout ensures dropped samples receive the null token, not noisy values.
        #
        # ASSUMPTION: the batch is already normalized (quantile/zscore) when preprocess_states
        # is called, so noise_std is in normalized space (e.g. 0.05 ≈ 5% of the [-1,1] range).
        self.proprio_dropout_p = proprio_dropout_p
        self._proprio_null_tokens_raw = {}
        self._proprio_dropout_p_per_key = {}  # null_key -> dropout probability
        self._proprio_noise_std_per_key = {}   # null_key -> Gaussian noise std
        self._proprio_noise_raw_per_key = {}   # null_key -> raw-space (rad) noise std
        self._proprio_noise_vec_per_key = {}   # null_key -> per-dim normalized noise tensor
        self._proprio_clamp_per_key = {}       # null_key -> clamp bound on normalized proprio

        self.stems = {}
        self.heads = {}
        # self.normalizer = {}
        self.domains = []
        self.use_modality_embedding = use_domain_embedding
        self.observation_horizon = observation_horizon
        self.action_horizon = action_horizon
        self.token_postprocessing = token_postprocessing
        # self.modalities_tokens = {}
        self.action_tokens = None
        self.stem_spec = {}
        self.head_spec = {}

        self.modalities = {}

        self.shared_keys = []

        self.auxiliary_ac_keys = None
        self.shared_action = False
        self.device = None

        self.ot_6dof = False
        self.use_dtw = False
        self.depth = None
        self.lambd = None

        self.diffusion = None

    def init_encoder(self, modality, encoder_spec):
        """
        Initialize an encoder for the specified modality.

        Parameters
        ----------
        modality : str
            The name of the modality.
        encoder_spec : dict or object
            The specification or configuration for the encoder.
        """
        self.encoders[modality] = encoder_spec

    def init_domain_stem(self, domain_name, stem_spec):
        """
        Initialize the stem (feature extractor) for a given domain along with its modalities.

        Parameters
        ----------
        domain_name : str
            The name of the domain.
        stem_spec : dict-like
            A specification containing configurations for each modality's stem.
        """

        self.stem_spec[domain_name] = stem_spec
        self.modalities[domain_name] = list(stem_spec.keys())

        for modality in self.modalities[domain_name]:
            stem_name = f"{domain_name}_{modality}"
            self.stems[stem_name] = stem_spec[modality]
            if hasattr(self.stems[stem_name], "init_cross_attn"):
                self.stems[stem_name].init_cross_attn(
                    stem_spec[modality].specs.cross_attn
                )
            if modality.startswith("state_") and hasattr(stem_spec[modality], "input_dim"):
                null_key = f"{domain_name}_{modality}"
                # Learnable null token: replaces the proprio tensor for dropped samples.
                # Initialized to zero; the model learns the optimal "absent proprio" embedding.
                # Shape matches input_dim so it passes through the same MLP as real values.
                self._proprio_null_tokens_raw[null_key] = nn.Parameter(
                    torch.zeros(stem_spec[modality].input_dim)
                )
                specs = getattr(stem_spec[modality], "specs", None)
                # Per-key dropout: falls back to the trunk-level global if not set on the stem.
                per_key_p = getattr(specs, "proprio_dropout_p", None) if specs is not None else None
                self._proprio_dropout_p_per_key[null_key] = (
                    per_key_p if per_key_p is not None else self.proprio_dropout_p
                )
                # Per-key noise std in normalized space (0.0 = disabled).
                self._proprio_noise_std_per_key[null_key] = (
                    getattr(specs, "noise_std", 0.0) if specs is not None else 0.0
                )
                # Per-key noise std in RAW units (rad); converted to a per-dim normalized
                # vector by the algo once norm stats are available (see HPT init).
                self._proprio_noise_raw_per_key[null_key] = (
                    getattr(specs, "noise_std_raw", 0.0) if specs is not None else 0.0
                )
                # Per-key clamp on normalized proprio, applied at train AND eval. Guards
                # near-static joints whose quantile range ~0 maps small raw offsets to
                # huge normalized values (the live-proprio cliff). None = disabled.
                self._proprio_clamp_per_key[null_key] = (
                    getattr(specs, "proprio_clamp", None) if specs is not None else None
                )

    def init_domain_head(self, domain_name, head_spec):
        """
        Initialize the head (prediction module) for a given domain.

        Parameters
        ----------
        domain_name : str
            The name of the domain.
        head_spec : dict or object
            The specification or configuration for the head, used with hydra.utils.instantiate.
        """
        self.head_spec[domain_name] = head_spec
        self.domains.append(domain_name)
        self.heads[domain_name] = head_spec

    def finalize_modules(self):
        """
        Finalize the module initialization by converting stems, heads, and modality tokens into
        nn.ModuleDict/nn.ParameterDict objects, applying weight initialization, and creating shared
        action tokens if required.
        """
        self.stems = nn.ModuleDict(self.stems)
        self.heads = nn.ModuleDict(self.heads)
        # Guard: finalize_modules may be called more than once (e.g. shared/pretrained init paths).
        if hasattr(self, "_proprio_null_tokens_raw"):
            self.proprio_null_tokens = nn.ParameterDict(self._proprio_null_tokens_raw)
            del self._proprio_null_tokens_raw
        self.apply(self._init_weights)

        # Shared action tokens
        if self.token_postprocessing == "action_token":
            self.action_tokens = nn.Parameter(
                torch.randn(1, self.action_horizon, self.embed_dim) * STD_SCALE
            )

    def _create_policy_trunk(
        self, embed_dim, num_blocks, num_heads, drop_path, weight_init_style
    ):
        """
        Create the transformer trunk module for policy processing.

        Parameters
        ----------
        embed_dim : int
            Dimension of token embeddings.
        num_blocks : int
            Number of transformer blocks.
        num_heads : int
            Number of attention heads in each block.
        drop_path : float
            Drop path rate for regularization.
        weight_init_style : str
            Weight initialization style.

        Returns
        -------
        nn.ModuleDict
            A module dictionary containing the main trunk transformer and, if provided, shared modality trunks.
        """
        trunk = {}

        trunk["trunk"] = SimpleTransformer(
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            ffn_dropout_rate=0.0,
            drop_path_rate=drop_path,
            attn_target=partial(
                MultiheadAttention,
                embed_dim=embed_dim,
                num_heads=num_heads,
                bias=True,
                add_bias_kv=True,
            ),
            pre_transformer_layer=nn.Sequential(
                nn.Identity(),
                EinOpsRearrange("b l d -> l b d"),
            ),
            post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
            weight_init_style=weight_init_style,
        )
        if (
            hasattr(self, "shared_modality_trunk")
            and self.shared_modality_trunk is not None
        ):
            for modality in self.shared_modality_trunk.modalities:
                trunk[modality] = self.shared_modality_trunk[modality]

        return nn.ModuleDict(trunk)

    def get_position_embedding(self, feature, embed_dim):
        """
        Generate sinusoidal positional embeddings for a given feature tensor.

        Parameters
        ----------
        feature : torch.Tensor
            The input tensor for which positional embeddings are computed.
        embed_dim : int
            The embedding dimension.

        Returns
        -------
        torch.Tensor
            The positional embedding tensor with the same device as the input.
        """
        tokensize = int(feature.shape[1])
        tokens = _sinusoid_table(tokensize, self.embed_dim, feature.device)
        return tokens.repeat((1, 1, 1)).to(feature.device)

    def preprocess_tokens(self, domain, features):
        """
        Preprocess and combine stem tokens with optional action tokens and add positional embeddings.

        Parameters
        ----------
        domain : str
            The domain for which tokens are being processed.
        features : list of torch.Tensor
            List of feature tokens from different modalities.

        Returns
        -------
        torch.Tensor
            The combined token tensor after adding positional embeddings.
        """
        tokens = torch.cat(features, dim=-2)

        if self.token_postprocessing == "action_token":
            action_tokens = self.action_tokens.repeat(len(tokens), 1, 1)
            tokens = torch.cat([action_tokens, tokens], dim=-2)

        position_tokens = self.get_position_embedding(tokens, self.embed_dim)
        return tokens + position_tokens

    def postprocess_tokens(self, trunk_tokens):
        """
        Postprocess the tokens output from the transformer trunk based on the token_postprocessing strategy.

        Parameters
        ----------
        trunk_tokens : torch.Tensor
            The token tensor output from the transformer trunk.

        Returns
        -------
        torch.Tensor
            The processed token tensor (e.g., averaged, max pooled, or selected action tokens).
        """
        if self.token_postprocessing == "mean":
            return trunk_tokens.mean(dim=1)
        elif self.token_postprocessing == "action_token":
            return trunk_tokens[:, : self.action_horizon]
        elif self.token_postprocessing == "max":
            return trunk_tokens.max(dim=1)[0]
        elif self.token_postprocessing == "last":
            return trunk_tokens[:, -1]
        elif self.token_postprocessing == "no-op":
            return trunk_tokens
        else:
            raise ValueError(
                f"Invalid token_postprocessing: {self.token_postprocessing}"
            )

    def preprocess_states(self, domain, data):
        """
        Preprocess state information in the input data by adding a new dimension if necessary.

        Parameters
        ----------
        domain : str
            The domain name.
        data : dict
            Dictionary containing input data with potential "state" keys.

        Returns
        -------
        dict
            Updated data dictionary with preprocessed state information.
        """
        for key in data:
            if "state" in key:
                data[key] = data[key][:, :, None]  # [B, S, D] -> [B, 1, S, D] for stem
                null_key = f"{domain}_{key}"

                # Step 1: Gaussian noise in normalized space. Applied first so dropped samples
                # receive the clean null token (not noisy values). Only active during training.
                noise_std = self._proprio_noise_std_per_key.get(null_key, 0.0)
                if self.training and noise_std > 0.0:
                    data[key] = data[key] + torch.randn_like(data[key]) * noise_std
                    # DEBUG: print("[NOISE] {} range after noise: [{:.3f}, {:.3f}]".format(null_key, data[key].min().item(), data[key].max().item()))

                # Step 1b: per-dim noise vector (raw-space sigma converted via norm stats),
                # trains tolerance to a uniform rad-level error on every joint.
                # getattr guards: old ckpts unpickle whole HPTModel objects (no __init__),
                # so these dicts may not exist on them.
                noise_vec = getattr(self, "_proprio_noise_vec_per_key", {}).get(null_key)
                if self.training and noise_vec is not None:
                    nv = noise_vec.to(device=data[key].device, dtype=data[key].dtype)
                    data[key] = data[key] + torch.randn_like(data[key]) * nv

                # Step 1c: clamp normalized proprio (train AND eval). Near-static joints
                # (quantile range ~0) map tiny raw offsets to huge normalized values at
                # deploy time; the clamp bounds them so they read as "saturated", which
                # training (with capped noise) teaches the model to ignore.
                clamp_val = getattr(self, "_proprio_clamp_per_key", {}).get(null_key)
                if clamp_val is not None:
                    data[key] = data[key].clamp(-clamp_val, clamp_val)

                # Step 2: Proprio dropout. Replaces the entire proprio tensor for a random
                # fraction of samples with a learned null token, forcing the model to use
                # other modalities (images, april tag) for those samples.
                p = self._proprio_dropout_p_per_key.get(null_key, self.proprio_dropout_p)
                if self.training and p > 0.0 and null_key in self.proprio_null_tokens:
                    B = data[key].shape[0]
                    drop_mask = torch.rand(B, device=data[key].device) < p
                    # branch-free (no drop_mask.any() host sync, 2026-10-16): dropped rows get the null token
                    mask = drop_mask.view(B, *([1] * (data[key].dim() - 1)))
                    data[key] = torch.where(mask, self.proprio_null_tokens[null_key].to(data[key].dtype), data[key])
        return data

    def stem_process(self, domain, data):
        """
        Process input data through modality-specific stems to compute latent feature tokens.

        Parameters
        ----------
        domain : str
            The domain corresponding to the input data.
        data : dict
            Dictionary containing input data for various modalities.

        Returns
        -------
        tuple
            A tuple containing:
                - A list of tokens from each modality.
                - A dictionary mapping each modality to its computed token.
        """
        feats = []
        feat_dict = {}
        for modality in self.modalities.get(domain, []) + self.shared_keys:
            if modality not in data:
                raise ValueError(f"Modality {modality} not in data for domain {domain}. Make sure the key and modality name match.")
            if modality in self.shared_keys:
                domain = "shared"

            stem = self.stems[f"{domain}_{modality}"]
            if modality in self.encoders:
                data[modality] = self.encoders[modality](data[modality])

            data_shape = data[modality].shape
            data_horizon = data_shape[1]
            horizon = data_horizon

            if (
                getattr(self, "train_mode", False)
                and self.stem_spec[domain][modality].specs.random_horizon_masking
                and data_horizon > 1
            ):
                horizon = np.random.randint(1, data_horizon + 1)
                data[modality] = data[modality][:, data_horizon - horizon :]

            positional_embedding = _sinusoid_table(
                horizon * int(np.prod(data_shape[2:-1])), data_shape[-1], data[modality].device
            ).to(data[modality])
            positional_embedding = einops.repeat(
                positional_embedding, "b h w -> (repeat b) h w", repeat=data_shape[0]
            )

            data[modality] = data[modality] + positional_embedding.view(
                data[modality].shape
            )
            stem_token = stem.compute_latent(data[modality])
            feats.append(stem_token)
            feat_dict[modality] = stem_token

        return feats, feat_dict

    def resume_from_depth(self, block_outputs, depth):
        """
        Detach at trunk depth and resume trunk forward pass.
        Gradients will only flow from depth upward.
        """
        cut_tokens = block_outputs[depth - 1].detach()

        blocks = self.trunk["trunk"].blocks
        for blk in list(blocks)[depth:]:
            cut_tokens = blk(cut_tokens, attn_mask=None)

        if self.trunk["trunk"].post_transformer_layer is not None:
            cut_tokens = self.trunk["trunk"].post_transformer_layer(cut_tokens)

        return self.postprocess_tokens(cut_tokens)

    def get_visual_embeds(self, domain, data, modality):
        """
        Compute visual embeddings for a given modality from the input data.

        Parameters
        ----------
        domain : str
            The domain corresponding to the input data.
        data : dict
            Dictionary containing input data.
        modality : str
            The modality for which visual embeddings are to be computed.

        Returns
        -------
        list
            A list containing:
                - The encoded features from the encoder.
                - The latent tokens computed by the modality stem.
        """
        if modality in self.shared_keys:
            domain = "shared"

        stem = self.stems[f"{domain}_{modality}"]

        encoder_feats = None

        if modality in self.encoders:
            encoder_feats = self.encoders[modality](data[modality])
        data_shape = encoder_feats.shape
        data_horizon = data_shape[1]
        horizon = data_horizon

        positional_embedding = get_sinusoid_encoding_table(
            0, horizon * int(np.prod(data_shape[2:-1])), data_shape[-1]
        ).to(encoder_feats)
        positional_embedding = einops.repeat(
            positional_embedding, "b h w -> (repeat b) h w", repeat=data_shape[0]
        )
        stem_feats = encoder_feats + positional_embedding.view(encoder_feats.shape)
        stem_token = stem.compute_latent(stem_feats)
        return [encoder_feats, stem_token]

    def forward_features(self, domain, data):
        """
        Compute feature tokens by processing the input data through stems and the transformer trunk.

        Parameters
        ----------
        domain : str
            The domain name for which features are computed.
        data : dict
            Dictionary containing input data for various modalities.

        Returns
        -------
        torch.Tensor
            The processed feature tokens after trunk and postprocessing.
        """
        data = self.preprocess_states(domain, data)
        stem_tokens, token_dict = self.stem_process(domain, data)

        trunk_tokens = self.preprocess_tokens(domain, stem_tokens)

        if not self.no_trunk:
            trunk_tokens, block_outputs = self.trunk["trunk"](trunk_tokens)

        proc_tokens = self.postprocess_tokens(trunk_tokens)
        return proc_tokens, block_outputs

    def init_dtw(self):
        self.dtw = SoftDTWLossPyTorch(gamma=0.1)
        self.use_dtw = True

    def compute_ot_loss(self, batch1, batch2, supervised=False):
        # with amp.autocast(enabled=False, device_type=self.device.type):
        depth = self.depth
        embodiment1 = batch1["domain"]
        embodiment2 = batch2["domain"]

        features1, block_outputs1 = self.forward_features(embodiment1, batch1["data"])
        features2, block_outputs2 = self.forward_features(embodiment2, batch2["data"])

        tokens1 = block_outputs1[depth].permute(1, 0, 2)  # B, S1, D
        tokens2 = block_outputs2[depth].permute(1, 0, 2)  # B, S2, D

        tokens1 = tokens1[:, : self.action_horizon]
        tokens2 = tokens2[:, : self.action_horizon]

        assert tokens1.shape[1] == tokens2.shape[1], (
            "input tokens must be of the same sequence length"
        )

        emb1_actions = batch1["data"]["action"]
        emb2_actions = batch2["data"]["action"]

        min_dim = min(emb1_actions.shape[-1], emb2_actions.shape[-1])

        emb1_actions = emb1_actions[..., :min_dim]
        emb2_actions = emb2_actions[..., :min_dim]

        ot_loss, avg_feature_dist = self.compute_ot(
            tokens1,
            tokens2,
            emb1_actions,
            emb2_actions,
            supervised=supervised,
            lambd=self.lambd,
        )
        return ot_loss, avg_feature_dist

    def make_custom_cost(self, scaling_mask):
        def custom_cost(x, y):
            cost = 0.5 * (((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(dim=-1))
            return cost * scaling_mask

        return custom_cost

    def compute_ot(
        self, tokens1, tokens2, emb1_actions, emb2_actions, supervised, lambd
    ):
        tokens1 = tokens1.reshape(tokens1.shape[0], -1)
        tokens2 = tokens2.reshape(tokens1.shape[0], -1)

        if not supervised:
            ot_loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05, truncate=18)
            ot_loss = ot_loss_fn(tokens2, tokens1)
            avg_feature_dist = torch.norm(tokens2 - tokens1, dim=-1).mean()
            return ot_loss, avg_feature_dist
        else:
            B = tokens1.shape[0]
            if not self.ot_6dof:
                emb1_actions = emb1_actions[..., :3]
                emb2_actions = emb2_actions[..., :3]
            if self.use_dtw:
                emb2_delta = emb2_actions
                emb1_delta = emb1_actions
                emb2_expand = emb2_delta.unsqueeze(1).expand(B, B, -1, -1)
                emb1_expand = emb1_delta.unsqueeze(0).expand(B, B, -1, -1)
                pairwise_dist = self.dtw(
                    emb2_expand.reshape(B * B, *emb2_actions.shape[1:]),
                    emb1_expand.reshape(B * B, *emb1_actions.shape[1:]),
                ).view(B, B)
            else:
                emb2_expand = emb2_actions.unsqueeze(1)  # (B, 1, T, D)
                emb1_expand = emb1_actions.unsqueeze(0)  # (1, B, T, D)
                pairwise_dist = ((emb2_expand - emb1_expand) ** 2).mean(
                    dim=(2, 3)
                )  # (B, B) #changed

            labels = torch.argmin(pairwise_dist, dim=1)
            W = torch.ones(B, B).to(self.device)
            W[torch.arange(B), labels] = lambd

            custom_cost_fn = self.make_custom_cost(W)

            ot_loss_fn = SamplesLoss(
                loss="sinkhorn", p=2, blur=0.05, cost=custom_cost_fn, truncate=18
            )

            ot_loss = ot_loss_fn(tokens2, tokens1)
            avg_feature_dist = torch.norm(tokens2 - tokens1, dim=-1).mean()
            return ot_loss, avg_feature_dist

    def compute_loss_depth(self, batch, depth):
        """
        Compute BC loss but restrict gradient flow to trunk blocks from `depth` upward.
        """
        self.train_mode = True
        domain, data = batch["domain"], batch["data"]

        # with amp.autocast(device_type=self.device.type):
        _, block_outputs = self.forward_features(domain, data)
        features = self.resume_from_depth(block_outputs, depth)
        action_loss = torch.tensor(0.0, device=self.device)
        shared_action_loss = torch.tensor(0.0, device=self.device)
        auxiliary_action_loss = torch.tensor(0.0, device=self.device)

        if domain in self.heads:
            action_loss += self.heads[domain].compute_loss(features, data)

        if self.shared_action:
            shared_action_loss += self.heads["shared"].compute_loss(features, data)

        if domain in self.auxiliary_ac_keys:
            for key in self.auxiliary_ac_keys[domain]:
                if f"{domain}_{key}" in self.heads:
                    data["action"] = data[key]
                    auxiliary_action_loss += self.heads[f"{domain}_{key}"].compute_loss(
                        features, data
                    )

        total_loss = action_loss + shared_action_loss + auxiliary_action_loss

        return total_loss

    def compute_loss(self, batch):
        """
        Compute the loss for a given batch of training data.

        Parameters
        ----------
        batch : dict
            Dictionary containing the keys "domain" and "data" for the input batch.

        Returns
        -------
        torch.Tensor
            The computed loss value.
        """
        self.train_mode = True
        domain, data = batch["domain"], batch["data"]
        self._stash_head_obs(domain, data)

        # scaler = amp.GradScaler()
        # with amp.autocast(device_type=self.device.type):
        features, block_outputs = self.forward_features(domain, data)
        action_loss = torch.tensor(0.0, device=self.device)
        shared_action_loss = torch.tensor(0.0, device=self.device)
        auxiliary_action_loss = torch.tensor(0.0, device=self.device)
        self.last_block_losses = None
        if domain in self.heads:
            action_loss += self.heads[domain].compute_loss(features, data)
            self.last_block_losses = getattr(
                self.heads[domain], "last_block_losses", None
            )

        if self.shared_action:
            shared_action_loss = self.heads["shared"].compute_loss(features, data)

        if domain in self.auxiliary_ac_keys:
            for key in self.auxiliary_ac_keys[domain]:
                if f"{domain}_{key}" in self.heads:
                    data["action"] = data[key]
                    auxiliary_action_loss += self.heads[f"{domain}_{key}"].compute_loss(
                        features, data
                    )

        total_loss = action_loss + shared_action_loss + auxiliary_action_loss
        if auxiliary_action_loss.requires_grad:   # 11-02 ptfwam: log the action and world-head parts (Train/<domain>_loss/act, /aux)
            self.last_block_losses = dict(self.last_block_losses or {}, act=action_loss.detach(), aux=auxiliary_action_loss.detach())
        return total_loss

    def forward(self, domain, data):
        """
        Forward pass of the HPTModel to compute actions.

        Parameters
        ----------
        domain : str
            The domain corresponding to the input data.
        data : dict
            Dictionary containing input data for various modalities.

        Returns
        -------
        torch.Tensor
            The predicted action output.
        """
        self._stash_head_obs(domain, data)
        features, block_outputs = self.forward_features(domain, data)
        action = {}

        if self.diffusion:
            features = (features, domain)

        if domain in self.heads:
            head = self.heads[domain]
            action[domain] = head(features, data) if getattr(head, "takes_data", False) else head(features)

        if self.shared_action:
            action["shared"] = self.heads["shared"](features)

        # Auxiliary heads (e.g. world-model targets) are TRAIN-time representation
        # shaping; skip them at inference unless explicitly enabled (EgoWAM-style:
        # the world head is dropped at deployment).
        if domain in self.auxiliary_ac_keys and getattr(self, "aux_heads_at_inference", False):
            for key in self.auxiliary_ac_keys[domain]:
                if f"{domain}_{key}" in self.heads:
                    action[key] = self.heads[f"{domain}_{key}"](features)

        return action

    def _stash_head_obs(self, domain, data):
        """heads that read observations (mixture of frames, 10-17: the proprio defines the frames) get them clean, before
        the stems add noise / dropout to data in place"""
        head = self.heads[domain] if domain in self.heads else None
        if getattr(head, "takes_data", False):
            data["head_obs"] = {k: data[k] for k in head.obs_keys}

    def save(self, checkpoint_path="./.checkpoints/hpt/full/"):
        """
        Save the state of the HPTModel to a specified checkpoint path.

        Parameters
        ----------
        checkpoint_path : str, optional
            The path to save the checkpoint (default is "./.checkpoints/hpt/full/").
        """
        try:
            torch.save(self.state_dict(), checkpoint_path)
        except FileNotFoundError:
            print(f"Could not save module parameters for trunk to {checkpoint_path}.")

    def _init_weights(self, m):
        """
        Initialize weights of a module using Xavier uniform initialization for Linear layers and constant
        initialization for LayerNorm layers.

        Parameters
        ----------
        m : nn.Module
            The module to initialize.
        """
        if getattr(m, "_hpt_keep_init", False):  # pretrained sub-modules, e.g. UtoniaPointEncoder.backbone
            return
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def freeze_trunk(self, num_layers=0):
        """
        Freeze a specified number of layers in the transformer trunk to prevent them from updating during training.

        Parameters
        ----------
        num_layers : int, optional
            The number of layers to freeze from the end of the trunk (default is 0).
        """
        layers = list(self.trunk["trunk"].children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_trunk(self, num_layers=0):
        """
        Unfreeze a specified number of layers in the transformer trunk to allow them to update during training.

        Parameters
        ----------
        num_layers : int, optional
            The number of layers to unfreeze from the end of the trunk (default is 0).
        """
        layers = list(self.trunk["trunk"].children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def load_trunk(self, path):
        """
        Load the transformer trunk state from a given file path or a HuggingFace URL.

        Parameters
        ----------
        path : str
            The file path or HuggingFace identifier (prefixed with "hf://") from which to load the trunk state.
        """
        if "hf://" in path:
            if "output" in path:
                path = path.replace("output/", "")
            path = download_from_huggingface(path[len("hf://") :])
        self.trunk.load_state_dict(torch.load(path), strict=True)

    def load_pretrained(self, checkpoint_path):
        """
        Load pretrained trunk weights from a specified checkpoint directory or HuggingFace URL.

        Parameters
        ----------
        checkpoint_path : str
            The path or HuggingFace identifier (prefixed with "hf://") for the pretrained checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            checkpoint_path = download_from_huggingface(checkpoint_path[len("hf://") :])

        self.load_trunk(os.path.join(checkpoint_path, "trunk.pth"))
