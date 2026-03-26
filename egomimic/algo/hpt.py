import os
from collections import OrderedDict
from functools import partial

import einops
import numpy as np
import torch
import torch.nn as nn
from geomloss import SamplesLoss
from overrides import override
from termcolor import cprint
from torchmetrics import MeanSquaredError
from tslearn.metrics import SoftDTWLossPyTorch

from egomimic.algo.algo import Algo
from egomimic.models.hpt_nets import MultiheadAttention, SimpleTransformer
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id
from egomimic.utils.egomimicUtils import (
    STD_SCALE,
    EinOpsRearrange,
    download_from_huggingface,
    frechet_gaussian_over_time,
    get_sinusoid_encoding_table,
    reverse_kl_from_samples,
)


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
        tokens = get_sinusoid_encoding_table(0, tokensize, self.embed_dim)
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
                data[key] = data[key][:, :, None]
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
                continue
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

            positional_embedding = get_sinusoid_encoding_table(
                0, horizon * int(np.prod(data_shape[2:-1])), data_shape[-1]
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

        assert (
            tokens1.shape[1] == tokens2.shape[1]
        ), "input tokens must be of the same sequence length"

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

        # scaler = amp.GradScaler()
        # with amp.autocast(device_type=self.device.type):
        features, block_outputs = self.forward_features(domain, data)
        action_loss = torch.tensor(0.0, device=self.device)
        shared_action_loss = torch.tensor(0.0, device=self.device)
        auxiliary_action_loss = torch.tensor(0.0, device=self.device)
        if domain in self.heads:
            action_loss += self.heads[domain].compute_loss(features, data)

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
        features, block_outputs = self.forward_features(domain, data)
        action = {}

        if self.diffusion:
            features = (features, domain)

        if domain in self.heads:
            action[domain] = self.heads[domain](features)

        if self.shared_action:
            action["shared"] = self.heads["shared"](features)

        if domain in self.auxiliary_ac_keys:
            for key in self.auxiliary_ac_keys[domain]:
                if f"{domain}_{key}" in self.heads:
                    action[key] = self.heads[f"{domain}_{key}"](features)

        return action

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


class HPT(Algo):
    """ """

    def __init__(
        self,
        data_schematic,
        camera_transforms,
        # ---------------------------
        # Image augmentations
        # ---------------------------
        train_image_augs,
        eval_image_augs,
        # ---------------------------
        # Trunk params
        # ---------------------------
        trunk: dict = None,
        # ---------------------------
        # Other model params
        # ---------------------------
        stem_specs: dict = None,
        head_specs: dict = None,
        shared_stem_specs: dict = None,
        shared_obs_keys: list = None,
        encoder_specs: dict = None,
        domains: list = None,
        auxiliary_ac_keys: dict = {},
        viz_func: dict = None,
        # ---------------------------
        # Pretrained
        # ---------------------------
        pretrained: bool = False,
        pretrained_checkpoint: str = "",
        # ---------------------------
        # Catch-all kwargs
        # ---------------------------
        **kwargs,
    ):
        self.nets = nn.ModuleDict()
        self.data_schematic = data_schematic
        self.viz_func = viz_func

        self.camera_transforms = camera_transforms
        self.train_image_augs = train_image_augs
        self.eval_image_augs = eval_image_augs
        self.stem_specs = stem_specs
        self.head_specs = head_specs
        self.encoders = encoder_specs

        self.shared_stem_specs = shared_stem_specs
        self.shared_obs_keys = shared_obs_keys

        self.pretrained = pretrained
        self.pretrained_checkpoint = pretrained_checkpoint

        self.domains = domains.copy()
        self.auxiliary_ac_keys = auxiliary_ac_keys.copy()
        self.shared_ac_key = kwargs.get("shared_ac_key", None)
        self.is_6dof = kwargs.get("6dof", False)
        self.kinematics_solver = kwargs.get("kinematics_solver", None)

        model = HPTModel(**trunk)
        model.auxiliary_ac_keys = self.auxiliary_ac_keys

        self.multitask = kwargs.get("multitask", False)
        self.device = kwargs.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        model.device = self.device

        self.diffusion = kwargs.get("diffusion", False)
        model.diffusion = self.diffusion

        if self.diffusion:
            if self.data_schematic.norm_mode == "zscore":
                cprint(
                    "WARNING: HPTModel with diffusion / flow matching is using 'zscore' normalization. "
                    "Consider switching to 'minmax' or 'quantile' norm_mode in train.yaml for better stability",
                    color="yellow",
                    attrs=["bold"],
                )

        if self.pretrained:
            model.load_pretrained(self.pretrained_checkpoint)

        if self.shared_obs_keys is not None:
            model.init_domain_stem("shared", self.shared_stem_specs)
            model.shared_keys = self.shared_obs_keys

        for domain in self.domains:
            if self.stem_specs[domain]:
                model.init_domain_stem(domain, self.stem_specs[domain])
            if self.head_specs[domain]:
                model.init_domain_head(domain, self.head_specs[domain])

        if self.shared_ac_key is not None:
            domain = "shared"
            model.shared_action = True
            model.init_domain_head(domain, self.head_specs[domain])

        for domain, key_list in self.auxiliary_ac_keys.items():
            for key in key_list:
                domain_key = f"{domain}_{key}"
                model.init_domain_head(domain_key, self.head_specs[domain_key])

        for modality, encoder_cfg in self.encoders.items():
            model.init_encoder(modality, encoder_cfg)

        model.finalize_modules()

        self.ac_keys = {}
        self.camera_keys = {}
        self.proprio_keys = {}
        self.lang_keys = {}

        self.ot = kwargs.get("ot", False)
        self.freeze_repr = kwargs.get("freeze_repr", False)
        self.depth = kwargs.get("depth", 8)
        self.freeze_depth = kwargs.get("freeze_depth", 8)
        model.depth = self.depth

        self.rkl_samples = kwargs.get("reverse_kl_samples", 4)

        if self.ot:
            self.ot_warm_start_steps = kwargs.get("ot_warm_start_steps", 0)
            self.ot_6dof = kwargs.get("ot_6dof", False)
            model.ot_6dof = self.ot_6dof
            self.warm_start_steps = kwargs.get("warm_start_steps", 30000)
            self.supervised = kwargs.get("supervised", False)
            if self.supervised:
                self.lambd = kwargs.get("lambda", 0.5)
                model.lambd = self.lambd
                self.dtw = kwargs.get("dtw", False)
                if self.dtw:
                    model.init_dtw()
            self.temperature = kwargs.get("temperature", 1.0)

        self.ac_keys = kwargs.get("ac_keys", {})

        for embodiment in self.domains:
            embodiment_id = get_embodiment_id(embodiment)
            self.camera_keys[embodiment_id] = []
            self.proprio_keys[embodiment_id] = []
            self.lang_keys[embodiment_id] = []
            for key in data_schematic.keys_of_type("action_keys", embodiment_id):
                if (
                    data_schematic.is_key_with_embodiment(key, embodiment_id)
                    and key == self.ac_keys[embodiment]
                ):
                    self.ac_keys[embodiment_id] = key
            for key in data_schematic.keys_of_type("camera_keys", embodiment_id):
                if data_schematic.is_key_with_embodiment(key, embodiment_id):
                    self.camera_keys[embodiment_id].append(key)
            for key in data_schematic.keys_of_type("proprio_keys", embodiment_id):
                if data_schematic.is_key_with_embodiment(key, embodiment_id):
                    self.proprio_keys[embodiment_id].append(key)
            for key in data_schematic.keys_of_type("lang_keys", embodiment_id):
                if data_schematic.is_key_with_embodiment(key, embodiment_id):
                    self.lang_keys[embodiment_id].append(key)

        model.finalize_modules()

        self.nets["policy"] = model
        self.nets = self.nets.float().to(self.device)

        self.training_step = 0

    @override
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            batch (dict): processed dict of batchs of form
                front_img_1 torch.Size([32, 3, 480, 640])
                right_wrist_img: torch.Size([32, 3, 480, 640])
                joint_positions: torch.Size([32, 1, 7])
                actions_joints_act: torch.Size([32, 100, 7])
                demo_number: torch.Size([32])
                _index: torch.Size([32])
                pad_mask: torch.Size([32, 100, 1])
                embodiment: torch.Size([])
        """
        processed_batch = {}
        for embodiment_name, _batch in batch.items():
            embodiment_id = get_embodiment_id(embodiment_name)
            processed_batch[embodiment_id] = {}
            for key, value in _batch.items():
                key_name = self.data_schematic.zarr_key_to_keyname(key, embodiment_id)
                if key is not None:
                    processed_batch[embodiment_id][key_name] = value

            ac_key = self.ac_keys[embodiment_id]
            if len(processed_batch[embodiment_id][ac_key].shape) != 3:
                raise ValueError("Action shape in batch is not 2")

            B, S, _ = processed_batch[embodiment_id][ac_key].shape
            device = processed_batch[embodiment_id][ac_key].device
            processed_batch[embodiment_id]["pad_mask"] = torch.ones(
                B, S, 1, device=device
            )

            processed_batch[embodiment_id] = self.data_schematic.normalize_data(
                processed_batch[embodiment_id], embodiment_id
            )
            processed_batch[embodiment_id]["embodiment"] = torch.tensor(
                [embodiment_id], device=self.device, dtype=torch.int64
            )
            # TODO make this work with any fp type
            for key, value in processed_batch[embodiment_id].items():
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    if value.is_floating_point():
                        value = value.float()
                    processed_batch[embodiment_id][key] = value

        return processed_batch

    @override
    def forward_training(self, batch):
        """
        One iteration of training. Sequentially, forward pass loss, Compute forward pass and compute losses.  Return predictions dictionary.  HPT also calculates loss here.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            predictions (dict): {ac_key: torch.Tensor (B, Seq, D), loss_key_name: torch.Tensor (1)}
        """

        predictions = OrderedDict()
        hpt_batches = {}
        self.training_step += 1
        for (
            embodiment_id,
            _batch,
        ) in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            cam_keys = self.camera_keys[embodiment_id]
            proprio_keys = self.proprio_keys[embodiment_id]
            lang_keys = self.lang_keys[embodiment_id]
            ac_key = self.ac_keys[embodiment_id]
            aux_ac_keys = self.auxiliary_ac_keys.get(embodiment_name, [])
            data = self._robomimic_to_hpt_data(
                _batch, cam_keys, proprio_keys, lang_keys, ac_key, aux_ac_keys
            )
            hpt_batch = {
                "domain": embodiment_name,  # readability on config side
                "data": data,
            }
            hpt_batches[embodiment_id] = self._clone_batch(hpt_batch)

            if self.freeze_repr:
                loss = self.nets["policy"].compute_loss_depth(
                    hpt_batch, depth=self.freeze_depth
                )
            else:
                loss = self.nets["policy"].compute_loss(hpt_batch)

            predictions[f"{embodiment_name}_{ac_key}"] = _batch[ac_key]
            predictions[f"{embodiment_name}_loss"] = loss

        if self.ot:
            ot_loss, avg_feat_distance = self._forward_ot(
                hpt_batches,
                get_embodiment_id(self.domains[0]),
                get_embodiment_id(self.domains[1]),
            )
            predictions["ot_loss"] = ot_loss
            predictions["avg_feature_distance"] = avg_feat_distance

        return predictions

    @override
    def forward_eval(self, batch):
        """
        Compute forward pass and return network outputs in @predictions dict.
        Unnormalize data here.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            unnorm_preds (dict): {<embodiment_name>_<ac_key>: torch.Tensor (B, Seq, D)}
        """
        unnorm_preds = {}
        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            cam_keys = self.camera_keys[embodiment_id]
            proprio_keys = self.proprio_keys[embodiment_id]
            lang_keys = self.lang_keys[embodiment_id]
            ac_key = self.ac_keys[embodiment_id]
            aux_ac_keys = self.auxiliary_ac_keys.get(embodiment_name, [])
            data = self._robomimic_to_hpt_data(
                _batch, cam_keys, proprio_keys, lang_keys, ac_key, aux_ac_keys
            )
            hpt_batch = {
                "domain": embodiment_name,  # readability on config side
                "data": data,
            }

            actions = self.nets["policy"].forward(
                hpt_batch["domain"], hpt_batch["data"]
            )
            predictions = OrderedDict()

            for key in actions:
                if key == embodiment_name:
                    pred = actions[embodiment_name]
                    ref = _batch[ac_key]
                    name = ac_key
                elif key == "shared":
                    pred = actions[key]
                    ref = _batch[self.shared_ac_key]
                    name = self.shared_ac_key
                else:
                    pred = actions[key]
                    ref = _batch[key]
                    name = key

                B, T, D = ref.shape
                pred = pred[:, :T, :D]
                predictions[name] = pred

            unnorm_actions = self.data_schematic.unnormalize_data(
                predictions, embodiment_id
            )
            for key in unnorm_actions:
                unnorm_preds[f"{embodiment_name}_{key}"] = unnorm_actions[key]

        return unnorm_preds

    @override
    def forward_eval_logging(self, batch):
        """
        Called by pl_model to generate a dictionary of metrics and an image visualization
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            metrics (dict):
                metricname: value (float)
            image: (B, 3, H, W)
        """
        preds = self.forward_eval(batch)
        metrics = {}
        images_dict = {}
        mse = MeanSquaredError()
        for embodiment_id, _batch in batch.items():
            _batch = self.data_schematic.unnormalize_data(_batch, embodiment_id)
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = self.ac_keys[embodiment_id]
            if f"{embodiment_name}_{ac_key}" in preds and ac_key != self.shared_ac_key:
                metrics[f"Valid/{embodiment_name}_{ac_key}_paired_mse_avg"] = mse(
                    (preds[f"{embodiment_name}_{ac_key}"]).cpu(), _batch[ac_key].cpu()
                )
                metrics[f"Valid/{embodiment_name}_{ac_key}_final_mse_avg"] = mse(
                    (preds[f"{embodiment_name}_{ac_key}"][:, -1]).cpu(),
                    _batch[ac_key][:, -1].cpu(),
                )
                fd = frechet_gaussian_over_time(
                    preds[f"{embodiment_name}_{ac_key}"], _batch[ac_key]
                )
                metrics[f"Valid/{embodiment_name}_{ac_key}_frechet_gauss_avg"] = (
                    fd.mean().item()
                )
                metrics[f"Valid/{embodiment_name}_{ac_key}_frechet_gauss_min"] = (
                    fd.min().item()
                )
                metrics[f"Valid/{embodiment_name}_{ac_key}_frechet_gauss_max"] = (
                    fd.max().item()
                )

            if embodiment_name in self.auxiliary_ac_keys:
                for aux_key in self.auxiliary_ac_keys[embodiment_name]:
                    pred_key = f"{embodiment_name}_{aux_key}"
                    if pred_key in preds:
                        metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                            preds[pred_key].cpu(), _batch[aux_key].cpu()
                        )
                        metrics[f"Valid/{pred_key}_final_mse_avg"] = mse(
                            preds[pred_key][:, -1].cpu(), _batch[aux_key][:, -1].cpu()
                        )
                        fd = frechet_gaussian_over_time(
                            preds[pred_key], _batch[aux_key]
                        )
                        metrics[f"Valid/{pred_key}_frechet_gauss_avg"] = (
                            fd.mean().item()
                        )
                        metrics[f"Valid/{pred_key}_frechet_gauss_min"] = fd.min().item()
                        metrics[f"Valid/{pred_key}_frechet_gauss_max"] = fd.max().item()

            if (
                self.shared_ac_key
                and f"{embodiment_name}_{self.shared_ac_key}" in preds
            ):
                pred_key = f"{embodiment_name}_{self.shared_ac_key}"
                metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                    preds[pred_key].cpu(), _batch[self.shared_ac_key].cpu()
                )
                metrics[f"Valid/{pred_key}_final_mse_avg"] = mse(
                    preds[pred_key][:, -1].cpu(),
                    _batch[self.shared_ac_key][:, -1].cpu(),
                )
                fd = frechet_gaussian_over_time(
                    preds[pred_key], _batch[self.shared_ac_key]
                )
                metrics[f"Valid/{pred_key}_frechet_gauss_avg"] = fd.mean().item()
                metrics[f"Valid/{pred_key}_frechet_gauss_min"] = fd.min().item()
                metrics[f"Valid/{pred_key}_frechet_gauss_max"] = fd.max().item()

            if self.rkl_samples and self.rkl_samples > 1:
                hpt_batch = {
                    "domain": embodiment_name,
                    "data": self._robomimic_to_hpt_data(
                        batch[embodiment_id],
                        self.camera_keys[embodiment_id],
                        self.proprio_keys[embodiment_id],
                        self.lang_keys[embodiment_id],
                        ac_key,
                        self.auxiliary_ac_keys.get(embodiment_name, []),
                    ),
                }
                rkl_targets = []

                if (
                    f"{embodiment_name}_{ac_key}" in preds
                    and ac_key != self.shared_ac_key
                ):
                    rkl_targets.append(
                        (
                            f"{embodiment_name}_{ac_key}",
                            _batch[ac_key].to(self.device),
                            embodiment_name,
                        )
                    )

                if embodiment_name in self.auxiliary_ac_keys:
                    for aux_key in self.auxiliary_ac_keys[embodiment_name]:
                        aux_pred_key = f"{embodiment_name}_{aux_key}"
                        if aux_pred_key in preds:
                            rkl_targets.append(
                                (aux_pred_key, _batch[aux_key].to(self.device), aux_key)
                            )

                if self.shared_ac_key:
                    shared_pred_key = f"{embodiment_name}_{self.shared_ac_key}"
                    if shared_pred_key in preds:
                        rkl_targets.append(
                            (
                                shared_pred_key,
                                _batch[self.shared_ac_key].to(self.device),
                                "shared",
                            )
                        )

                M = int(self.rkl_samples)
                for pred_key_name, gt_tensor, head_key in rkl_targets:
                    samples = self._collect_policy_samples(
                        hpt_batch, ref=gt_tensor, key_name=head_key, M=M
                    )
                    rkl = reverse_kl_from_samples(samples, gt_tensor)
                    metrics[f"Valid/{pred_key_name}_reverse_kl_M{M}"] = rkl.item()

            ims = self.visualize_preds(preds, _batch)
            images_dict[embodiment_id] = ims
        return metrics, images_dict

    @override
    def visualize_preds(self, predictions, batch):
        """
        Helper function to visualize predictions on top of images
        Args:
            predictions (dict): {ac_key: torch.Tensor (B, Seq, D)}
            batch (dict): {ac_key: torch.Tensor (B, Seq, D), front_img_1: torch.Tensor (B, 3, H, W), embodiment: torch.Tensor (1)}
        Returns:
            ims (np.ndarray): (B, H, W, 3) - images with actions drawn on top
        """
        if self.viz_func is None:
            raise ValueError("viz_func is not set")
        embodiment_id = batch["embodiment"][0].item()
        embodiment_name = get_embodiment(embodiment_id).lower()

        return self.viz_func[embodiment_name](predictions, batch)

    @override
    def compute_losses(self, predictions, batch):
        """
        Compute losses based on network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            losses (dict): dictionary of losses computed over the batch
                loss_key_name: torch.Tensor (1)
        """
        total_action_loss = torch.tensor(0.0, device=self.device)
        loss_dict = OrderedDict()

        if self.ot:
            bc_weight = 1.0 if self.training_step >= self.warm_start_steps else 0.0
            ot_weight = 1.0 if self.training_step >= self.ot_warm_start_steps else 0.0
        else:
            bc_weight = 1.0

        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            bc_loss = predictions[f"{embodiment_name}_loss"]
            scaled_bc_loss = bc_weight * bc_loss
            total_action_loss += scaled_bc_loss
            loss_dict[f"{embodiment_name}_loss"] = bc_loss  # for logging

        if self.ot:
            loss_dict["ot_loss"] = predictions["ot_loss"]
            loss_dict["avg_feature_distance"] = predictions["avg_feature_distance"]
            total_action_loss += ot_weight * self.temperature * predictions["ot_loss"]

        loss_dict["action_loss"] = total_action_loss / len(self.domains)
        return loss_dict

    @override
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of losses returned by compute_losses
                losses:
                    loss_key_name: torch.Tensor (1)
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for loss_key, loss in info["losses"].items():
            log[loss_key] = loss.item()
        return log

    @torch.no_grad()
    def _collect_policy_samples(self, hpt_batch, ref, key_name, M):
        """
        Collect policy samples for Reverse KL loss
        """
        B, T, D = ref.shape
        samples = []
        was_training = self.nets.training
        self.nets.eval()
        for _ in range(M):
            out = self.nets["policy"].forward(
                hpt_batch["domain"], self._clone_batch(hpt_batch["data"])
            )
            if key_name in out:
                pred = out[key_name]
            else:
                pred = out[hpt_batch["domain"]]

            pred = pred[:, :T, :D]
            samples.append(pred.unsqueeze(0))
        if was_training:
            self.nets.train()
        return torch.cat(samples, dim=0)

    def _forward_ot(self, batch, embodiment1_id, embodiment2_id):
        hpt_batch_1 = batch[embodiment1_id]
        hpt_batch_2 = batch[embodiment2_id]

        return self.nets["policy"].compute_ot_loss(
            hpt_batch_1,
            hpt_batch_2,
            supervised=self.supervised,
        )

    def _robomimic_to_hpt_data(
        self, batch, cam_keys, proprio_keys, lang_keys, ac_key, aux_ac_keys=[]
    ):
        """
        helper method that returns data in the format required for the HPT model
        """
        data = {}

        for key in proprio_keys:
            if key in batch:
                data[f"state_{key}"] = batch[key].unsqueeze(1)

        for key in cam_keys:
            if key in batch:
                _data = batch[key]
                if not torch.all(_data == 0):
                    if self.nets.training and key in self.encoders:
                        _data = self.train_image_augs(_data)
                    elif self.eval_image_augs and key in self.encoders:
                        _data = self.eval_image_augs(_data)

                data[key] = _data.unsqueeze(1).unsqueeze(1)

        for key in lang_keys:
            if key in batch:
                data[key] = batch[key]

        data["is_6dof"] = self.is_6dof
        data["pad_mask"] = batch["pad_mask"]
        data["embodiment"] = batch["embodiment"]

        for aux_ac_key in aux_ac_keys:
            data[aux_ac_key] = batch[aux_ac_key]

        if self.shared_ac_key:
            data["action"] = batch[self.shared_ac_key]
        else:
            data["action"] = batch[ac_key]
        return data

    def _clone_batch(self, batch):
        """Recursively clones all tensors inside a nested dictionary."""
        if isinstance(batch, dict):
            return {key: self._clone_batch(val) for key, val in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.clone()
        else:
            return batch  # Return as is for non-tensor types

    @staticmethod
    def _extract_xyz(x):
        """
        Extract xyz (3D position) and rotation from 6DoF or 6DoF+gripper actions.

        Supports:
        - 6: 6DoF (single arm)
        - 7: 6DoF + gripper (single arm)
        - 12: 2 arms × 6DoF
        - 14: 2 arms × (6DoF + gripper)

        Returns:
            xyz: Tensor with only xyz per arm (shape: ..., 3) or (..., 6) for dual-arm.
            rot: Tensor with only rotation per arm (shape: ..., 3) or (..., 6) for dual-arm.
        """
        if x.shape[-1] == 6:
            return x[..., :3], x[..., 3:6]
        elif x.shape[-1] == 7:
            return x[..., :3], x[..., 3:6]
        elif x.shape[-1] == 12:
            xyz_right = x[..., :3]
            rot_right = x[..., 3:6]
            xyz_left = x[..., 6:9]
            rot_left = x[..., 9:12]
            return torch.cat([xyz_right, xyz_left], dim=-1), torch.cat(
                [rot_right, rot_left], dim=-1
            )
        elif x.shape[-1] == 14:
            xyz_right = x[..., :3]
            rot_right = x[..., 3:6]
            xyz_left = x[..., 7:10]
            rot_left = x[..., 10:13]
            return torch.cat([xyz_right, xyz_left], dim=-1), torch.cat(
                [rot_right, rot_left], dim=-1
            )
        else:
            raise ValueError(f"Unexpected shape for 6DoF input: {x.shape}")
