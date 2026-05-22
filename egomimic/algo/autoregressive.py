from collections import OrderedDict
from functools import partial

import einops
import hydra.utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from oat.model.common.normalizer import SingleFieldLinearNormalizer
from omegaconf import OmegaConf
from overrides import override

from egomimic.algo.algo import Algo
from egomimic.models.hpt_nets import MultiheadAttention, SimpleTransformer
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id
from egomimic.utils.egomimicUtils import (
    EinOpsRearrange,
    get_sinusoid_encoding_table,
)


def oattok_from_egomimic_lightning_ckpt(
    checkpoint,
    encoder,
    decoder,
    quantizer,
    state_dict_prefix="nets.tokenizer.",
):
    """Instantiate ``OATTok`` and load weights saved by ``model=oat_tokenizer`` runs."""
    tok = hydra.utils.instantiate(
        OmegaConf.create(
            {
                "_target_": "oat.tokenizer.oat.tokenizer.OATTok",
                "encoder": encoder,
                "decoder": decoder,
                "quantizer": quantizer,
            }
        )
    )
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    p = len(state_dict_prefix)
    sub = {k[p:]: v for k, v in sd.items() if k.startswith(state_dict_prefix)}
    tok.load_state_dict(sub, strict=True)
    return tok


class ARmodel(nn.Module):
    """HPT-style encoder + trunk feeding the AR decoder (see encode_obs / generate)."""

    def __init__(
        self,
        vocab_size: int,
        bos_id: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        embed_dim: int = 256,
        num_blocks: int = 16,
        num_heads: int = 8,
        drop_path: float = 0.1,
        weight_init_style: str = "pytorch",
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.bos_id = bos_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        # --- observation encoding (HPT-style) ---
        self.encoders = nn.ModuleDict()
        self.stems = {}
        self.stem_spec = {}
        self.modalities = {}
        self.shared_keys = []
        if not torch.cuda.is_available():
            raise RuntimeError("ARmodel requires CUDA; no CUDA device is available.")
        self.device = torch.device("cuda")

        self.trunk = SimpleTransformer(
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

        # vocab_size action tokens + 1 BOS token
        self.tok_emb = nn.Embedding(vocab_size + 1, embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    # ------------------------------------------------------------------
    # Stem / encoder registration (called by AutoregressivePolicy init)
    # ------------------------------------------------------------------

    def init_encoder(self, modality, encoder_spec):
        self.encoders[modality] = encoder_spec

    def init_domain_stem(self, domain_name, stem_spec):
        self.stem_spec[domain_name] = stem_spec
        self.modalities[domain_name] = list(stem_spec.keys())
        for modality in self.modalities[domain_name]:
            stem_name = f"{domain_name}_{modality}"
            self.stems[stem_name] = stem_spec[modality]
            if hasattr(self.stems[stem_name], "init_cross_attn"):
                self.stems[stem_name].init_cross_attn(
                    stem_spec[modality].specs.cross_attn
                )

    def finalize_modules(self):
        self.stems = nn.ModuleDict(self.stems)
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def encode_obs(self, domain: str, data: dict) -> torch.Tensor:
        """
        Encode raw observations into context tokens for the AR decoder.
        Returns (B, T_obs, embed_dim).
        """
        data = self.preprocess_states(domain, data)
        stem_tokens, _ = self.stem_process(domain, data)

        tokens = torch.cat(stem_tokens, dim=-2).to(self.device)  # (B, T_obs, embed_dim)
        pos = self.get_position_embedding(tokens)
        tokens = tokens + pos
        out, _ = self.trunk(tokens)
        return out  # (B, T_obs, embed_dim)

    # ------------------------------------------------------------------
    # Loss and inference
    # ------------------------------------------------------------------

    def generate(self, batch) -> torch.Tensor:
        domain, data = batch["domain"], batch["data"]

        # 1. encode observations into conditioning tokens
        conditioning_features = self.encode_obs(domain, data)  # (B, T_cond, embed_dim)
        B, T_cond, _ = conditioning_features.shape

        # 2. build teacher-forced prefix: [BOS, gt_tokens[0], ..., gt_tokens[T-2]]
        gt_tokens = data["action_tokens"].to(self.device)  # (B, T_act)
        T_act = gt_tokens.shape[1]
        prefix = torch.cat(
            [
                torch.full((B, 1), self.bos_id, dtype=torch.long, device=self.device),
                gt_tokens[:, :-1],
            ],
            dim=1,
        )  # (B, T_act)

        # 3. embed prefix tokens and add positional encoding
        tok_emb = self.tok_emb(prefix)  # (B, T_act, embed_dim)
        tok_emb = tok_emb + self.get_position_embedding(tok_emb)

        # 4. concatenate [cond tokens | action tokens]
        # self-attention then computes K/V from cond tokens once and action tokens
        # attend to them — equivalent to precomputing cond K/V and reusing per step
        x = torch.cat([conditioning_features, tok_emb], dim=1)  # (B, L, embed_dim)
        L = T_cond + T_act

        # 5. attention mask:
        #    - cond rows: attend to cond only (blocked from action tokens)
        #    - action rows: attend to all cond + causally to prior action tokens
        mask = torch.zeros(L, L, device=self.device)
        mask[:T_cond, T_cond:] = float("-inf")
        causal = torch.triu(
            torch.full((T_act, T_act), float("-inf"), device=self.device), diagonal=1
        )
        mask[T_cond:, T_cond:] = causal

        # 6. single forward pass through trunk
        out, _ = self.trunk(x, attn_mask=mask)  # (B, L, embed_dim)

        # 7. project action token outputs to logits
        logits = self.head(out[:, T_cond:, :])  # (B, T_act, vocab_size)
        return logits

    def compute_loss(self, batch) -> torch.Tensor:
        logits = self.generate(batch)
        gt_tokens = batch["data"]["action_tokens"].to(self.device)
        B, T_act = gt_tokens.shape
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            gt_tokens.reshape(-1),
        )
        return loss

    @torch.no_grad()
    def forward(self, domain: str, data: dict) -> torch.Tensor:
        """Autoregressive inference: generate action tokens one step at a time."""
        conditioning_features = self.encode_obs(domain, data)  # (B, T_cond, embed_dim)
        B, T_cond, _ = conditioning_features.shape

        generated = torch.full(
            (B, 1), self.bos_id, dtype=torch.long, device=self.device
        )

        for _ in range(self.max_new_tokens):
            T_gen = generated.shape[1]
            tok_emb = self.tok_emb(generated)
            tok_emb = tok_emb + self.get_position_embedding(tok_emb)

            L = T_cond + T_gen
            x = torch.cat([conditioning_features, tok_emb], dim=1)

            mask = torch.zeros(L, L, device=self.device)
            mask[:T_cond, T_cond:] = float("-inf")
            causal = torch.triu(
                torch.full((T_gen, T_gen), float("-inf"), device=self.device),
                diagonal=1,
            )
            mask[T_cond:, T_cond:] = causal

            out, _ = self.trunk(x, attn_mask=mask)  # (B, L, embed_dim)
            logits = self.head(out[:, -1, :]) / self.temperature  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated[:, 1:]  # strip BOS → (B, max_new_tokens)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # TODO: verify pos emb
    def get_position_embedding(self, feature):
        tokensize = int(feature.shape[1])
        return torch.zeros(
            1, tokensize, self.embed_dim, device=self.device, dtype=feature.dtype
        )

    def preprocess_states(self, domain, data):
        for key in data:
            if "state" in key:
                data[key] = data[key][:, :, None]
        return data

    def stem_process(self, domain, data):
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

    def get_visual_embeds(self, domain, data, modality):
        if modality in self.shared_keys:
            domain = "shared"
        stem = self.stems[f"{domain}_{modality}"]
        encoder_feats = None
        if modality in self.encoders:
            encoder_feats = self.encoders[modality](data[modality])
        data_shape = encoder_feats.shape
        horizon = data_shape[1]
        positional_embedding = get_sinusoid_encoding_table(
            0, horizon * int(np.prod(data_shape[2:-1])), data_shape[-1]
        ).to(encoder_feats)
        positional_embedding = einops.repeat(
            positional_embedding, "b h w -> (repeat b) h w", repeat=data_shape[0]
        )
        stem_feats = encoder_feats + positional_embedding.view(encoder_feats.shape)
        stem_token = stem.compute_latent(stem_feats)
        return [encoder_feats, stem_token]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class AutoregressivePolicy(Algo):
    def __init__(
        self,
        data_schematic,
        camera_transforms,
        train_image_augs,
        eval_image_augs,
        trunk: dict = None,
        stem_specs: dict = None,
        shared_stem_specs: dict = None,
        shared_obs_keys: list = None,
        encoder_specs: dict = None,
        domains: list = None,
        auxiliary_ac_keys: dict = {},
        viz_func: dict = None,
        action_tokenizer=None,
        freeze_tokenizer: bool = True,
        **kwargs,
    ):
        self.nets = nn.ModuleDict()
        self.data_schematic = data_schematic
        self.viz_func = viz_func

        self.train_image_augs = train_image_augs
        self.eval_image_augs = eval_image_augs
        self.stem_specs = stem_specs
        self.encoders = encoder_specs

        self.shared_stem_specs = shared_stem_specs
        self.shared_obs_keys = shared_obs_keys

        self.domains = domains.copy()
        self.auxiliary_ac_keys = auxiliary_ac_keys.copy()
        self.is_6dof = kwargs.get("6dof", False)

        self.action_tokenizer = action_tokenizer
        self.freeze_tokenizer = freeze_tokenizer
        if freeze_tokenizer:
            for param in self.action_tokenizer.parameters():
                param.requires_grad_(False)
            self.action_tokenizer.eval()
        else:
            # Joint training: data is pre-normalized by data_schematic to [-1, 1].
            # Seed the tokenizer normalizer as identity so it doesn't double-normalize.
            action_dim = self.action_tokenizer.decoder.sample_dim
            scale = torch.ones(action_dim, dtype=torch.float32)
            offset = torch.zeros(action_dim, dtype=torch.float32)
            stats = {
                "min": -torch.ones(action_dim, dtype=torch.float32),
                "max": torch.ones(action_dim, dtype=torch.float32),
                "mean": torch.zeros(action_dim, dtype=torch.float32),
                "std": torch.ones(action_dim, dtype=torch.float32),
            }
            self.action_tokenizer.normalizer["action"] = (
                SingleFieldLinearNormalizer.create_manual(scale, offset, stats)
            )

        # create AR model
        codebook_size = self.action_tokenizer.quantizer.codebook_size
        latent_horizon = self.action_tokenizer.latent_horizon
        model = ARmodel(
            vocab_size=codebook_size,
            bos_id=codebook_size,  # one past the last valid token index
            max_new_tokens=latent_horizon,
            temperature=kwargs.get("temperature", 1.0),
            embed_dim=trunk["embed_dim"],
            num_blocks=trunk["num_blocks"],
            num_heads=trunk["num_heads"],
            drop_path=trunk.get("drop_path", 0.1),
            weight_init_style=trunk.get("weight_init_style", "pytorch"),
            dropout=trunk["dropout"],
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "AutoregressivePolicy requires CUDA; no CUDA device is available."
            )
        if kwargs.get("device") is not None:
            self.device = torch.device(kwargs["device"])
            if self.device.type != "cuda":
                raise ValueError(
                    f"device must be CUDA (e.g. cuda or cuda:0), got {self.device!r}"
                )
        else:
            self.device = torch.device("cuda")
        model.device = self.device

        if self.shared_obs_keys is not None:
            model.init_domain_stem("shared", self.shared_stem_specs)
            model.shared_keys = self.shared_obs_keys

        for domain in self.domains:
            if self.stem_specs[domain]:
                model.init_domain_stem(domain, self.stem_specs[domain])

        for modality, encoder_cfg in self.encoders.items():
            model.init_encoder(modality, encoder_cfg)

        model.finalize_modules()

        self.ac_keys = {}
        self.camera_keys = {}
        self.proprio_keys = {}

        self.freeze_repr = kwargs.get("freeze_repr", False)
        self.depth = kwargs.get("depth", 8)
        self.freeze_depth = kwargs.get("freeze_depth", 8)
        model.depth = self.depth

        self.rkl_samples = kwargs.get("reverse_kl_samples", 4)
        self.ac_keys = kwargs.get("ac_keys", {})

        for embodiment in self.domains:
            embodiment_id = get_embodiment_id(embodiment)
            self.camera_keys[embodiment_id] = []
            self.proprio_keys[embodiment_id] = []
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

        self.nets["policy"] = model
        self.nets = self.nets.float().to(self.device)
        self.action_tokenizer = self.action_tokenizer.to(self.device)

        self.training_step = 0

    @override
    def process_batch_for_training(self, batch):
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
            processed_batch[embodiment_id]["pad_mask"] = torch.ones(
                B, S, 1, device=self.device
            )

            processed_batch[embodiment_id] = self.data_schematic.normalize_data(
                processed_batch[embodiment_id], embodiment_id
            )
            processed_batch[embodiment_id]["embodiment"] = torch.tensor(
                [embodiment_id], device=self.device, dtype=torch.int64
            )
            for key, value in processed_batch[embodiment_id].items():
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    if value.is_floating_point():
                        value = value.float()
                    processed_batch[embodiment_id][key] = value

        return processed_batch

    @override
    def forward_training(self, batch):
        predictions = OrderedDict()
        self.training_step += 1
        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            cam_keys = self.camera_keys[embodiment_id]
            proprio_keys = self.proprio_keys[embodiment_id]
            ac_key = self.ac_keys[embodiment_id]
            aux_ac_keys = self.auxiliary_ac_keys.get(embodiment_name, [])
            data = self._robomimic_to_hpt_data(
                _batch, cam_keys, proprio_keys, ac_key, aux_ac_keys
            )
            actions = data["action"]
            if self.freeze_tokenizer:
                with torch.no_grad():
                    data["action_tokens"] = self.action_tokenizer.tokenize(actions)
            else:
                # Reconstruction objective — same pipeline as OATTok.forward().
                # Policy CE alone does not backprop into discrete token indices.
                nsamples = self.action_tokenizer.normalizer["action"].normalize(actions)
                z = self.action_tokenizer.encoder(nsamples)
                z_q, action_tokens = self.action_tokenizer.quantizer(z)
                recons = self.action_tokenizer.decoder(z_q)
                predictions[f"{embodiment_name}_tokenizer_loss"] = F.mse_loss(
                    recons, nsamples
                )
                data["action_tokens"] = action_tokens

            hpt_batch = {
                "domain": embodiment_name,
                "data": data,
            }
            loss = self.nets["policy"].compute_loss(hpt_batch)

            predictions[f"{embodiment_name}_{ac_key}"] = _batch[ac_key]
            predictions[f"{embodiment_name}_loss"] = loss

        return predictions

    @override
    def forward_eval(self, batch):
        unnorm_preds = {}
        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            cam_keys = self.camera_keys[embodiment_id]
            proprio_keys = self.proprio_keys[embodiment_id]
            ac_key = self.ac_keys[embodiment_id]
            aux_ac_keys = self.auxiliary_ac_keys.get(embodiment_name, [])
            data = self._robomimic_to_hpt_data(
                _batch, cam_keys, proprio_keys, ac_key, aux_ac_keys
            )
            hpt_batch = {
                "domain": embodiment_name,
                "data": data,
            }

            action_tokens = self.nets["policy"].forward(
                hpt_batch["domain"], hpt_batch["data"]
            )  # (B, max_new_tokens)
            with torch.inference_mode():
                actions = self.action_tokenizer.detokenize(
                    tokens=action_tokens.to(self.device)
                )
            predictions = OrderedDict()
            ref = _batch[ac_key]
            B, T, D = ref.shape
            pred = actions[:, :T, :D]
            predictions[ac_key] = pred
            unnorm_actions = self.data_schematic.unnormalize_data(
                predictions, embodiment_id
            )
            for key in unnorm_actions:
                unnorm_preds[f"{embodiment_name}_{key}"] = unnorm_actions[key]

        return unnorm_preds

    @override
    def forward_eval_logging(self, batch):
        preds = self.forward_eval(batch)
        metrics = {}
        images_dict = {}
        for embodiment_id, _batch in batch.items():
            _batch = self.data_schematic.unnormalize_data(_batch, embodiment_id)
            ims = self.visualize_preds(preds, _batch)
            images_dict[embodiment_id] = ims
        return metrics, images_dict

    @override
    def visualize_preds(self, predictions, batch):
        if self.viz_func is None:
            raise ValueError("viz_func is not set")
        embodiment_id = batch["embodiment"][0].item()
        embodiment_name = get_embodiment(embodiment_id).lower()

        return self.viz_func[embodiment_name](predictions, batch)

    @override
    def compute_losses(self, predictions, batch):
        total_action_loss = torch.tensor(0.0, device=self.device)
        loss_dict = OrderedDict()
        bc_weight = 1.0

        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            bc_loss = predictions[f"{embodiment_name}_loss"]
            scaled_bc_loss = bc_weight * bc_loss
            total_action_loss += scaled_bc_loss
            loss_dict[f"{embodiment_name}_loss"] = bc_loss

            if not self.freeze_tokenizer:
                tkey = f"{embodiment_name}_tokenizer_loss"
                if tkey in predictions:
                    tloss = predictions[tkey]
                    total_action_loss += tloss
                    loss_dict[tkey] = tloss

        loss_dict["action_loss"] = total_action_loss / len(self.domains)
        return loss_dict

    @override
    def log_info(self, info):
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for loss_key, loss in info["losses"].items():
            log[loss_key] = loss.item()
        return log

    def _robomimic_to_hpt_data(
        self, batch, cam_keys, proprio_keys, ac_key, aux_ac_keys=[]
    ):
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

        data["is_6dof"] = self.is_6dof
        data["pad_mask"] = batch["pad_mask"]
        data["embodiment"] = batch["embodiment"]

        for aux_ac_key in aux_ac_keys:
            data[aux_ac_key] = batch[aux_ac_key]

        data["action"] = batch[ac_key]
        return data

    def _clone_batch(self, batch):
        if isinstance(batch, dict):
            return {key: self._clone_batch(val) for key, val in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.clone()
        else:
            return batch

    @staticmethod
    def _extract_xyz(x):
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
