"""
Implementation of Action Chunking with Transformers (ACT).
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from egomimic.utils.egomimicUtils import nds
import matplotlib.pyplot as plt

from egomimic.algo.algo import Algo


from egomimic.utils.egomimicUtils import draw_actions

from torchmetrics import MeanSquaredError

import numpy as np

from overrides import override


class ACTModel(nn.Module):
    """
    ACT Model closely following DETRVAE from ACT but using standard torch.nn components

    backbones : visual backbone per cam input
    transformer : encoder-decoder transformer
    encoder : style encoder
    latent_dim : style var dim
    a_dim : action dim
    state_dim : proprio dim
    num_queries : predicted action dim
    camera_names : list of camera inputs
    """

    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        latent_dim,
        a_dim,
        state_dim,
        num_queries,
        camera_names,
        num_channels,
    ):
        super(ACTModel, self).__init__()

        self.action_dim = a_dim
        self.latent_dim = latent_dim
        self.state_dim = state_dim

        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d

        # self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.num_channels = num_channels

        if backbones is not None:
            self.input_proj = nn.Conv2d(self.num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
        else:
            self.backbones = None

        ## ACTSP
        # self.cls_embed = nn.Embedding(1, hidden_dim)

        self.latent_out_proj = nn.Linear(
            self.latent_dim, hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

        self.encoder_action_proj = nn.Linear(
            self.action_dim, hidden_dim
        )  # project robot action to embedding

        self.encoder_joint_proj = nn.Linear(
            self.state_dim, hidden_dim
        )  # project robot qpos to embedding

        self.transformer_input_proj = nn.Linear(self.state_dim, hidden_dim)

        self.action_head = nn.Linear(hidden_dim, self.action_dim)

    def forward(self, qpos, actions, image, is_pad=None, env_state=None):
        return self._forward(
            qpos=qpos,
            actions=actions,
            image=image,
            encoder_action_proj=self.encoder_action_proj,
            encoder_joint_proj=self.encoder_joint_proj,
            transformer_input_proj=self.transformer_input_proj,
            action_head=self.action_head,
            camera_names=self.camera_names,
            is_pad=is_pad,
        )

    def _forward(
        self,
        qpos,
        actions,
        image,
        encoder_action_proj=None,
        encoder_joint_proj=None,
        transformer_input_proj=None,
        action_head=None,
        camera_names=None,
        env_state=None,
        is_pad=None,
        aux_action_head=None,
    ):
        """
        args:
            qpos: (batch, qpos_dim)
            actions: (batch, seq, action_dim)
            image: (batch, camera_number, channel, height, width)
            env_state: None
        """

        is_training = actions is not None
        batch_size = qpos.size(0)

        if is_training:
            actions_encod = encoder_action_proj(actions)
            qpos_encod = encoder_joint_proj(qpos)
            # Use StyleEncoder to get latent distribution and sample
            dist = self.encoder(qpos_encod, actions_encod)
            mu = dist.mean
            logvar = dist.scale.log() * 2
            latent_sample = dist.rsample()
        else:
            # Inference mode, use zeros for latent vector
            mu = logvar = None
            latent_sample = torch.zeros(batch_size, self.latent_dim, device=qpos.device)

        latent_input = self.latent_out_proj(latent_sample)  # [batch_size, hidden_dim]

        all_cam_features = []
        for cam_id in range(len(camera_names)):
            features = self.backbones[cam_id](image[:, cam_id])
            features = self.input_proj(features)
            all_cam_features.append(features)

        src = torch.cat(all_cam_features, dim=-1)  # [B, hidden_dim, H, W * num_cameras]

        batch_size, hidden_dim, height, width = src.shape
        src = src.flatten(2).permute(
            0, 2, 1
        )  # [B, S, hidden_dim], S = H * W * num_cameras]

        proprio_input = transformer_input_proj(qpos).unsqueeze(1)  # [B, 1, hidden_dim]
        latent_input = latent_input.unsqueeze(1)  # [B, 1, hidden_dim]

        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # [B, num_queries, hidden_dim]

        tgt = query_embed  # tgt = torch.zeros_like(query_embed) + query_embed. ACT passes zeros to decoder

        src = torch.cat(
            [latent_input, proprio_input, src], axis=1
        )  # [B, S + 2, hidden_dim]

        # Learnable additional pos embed for latent input, proprio input
        additional_pos_embed = self.additional_pos_embed.weight.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # [B, 2, hidden_dim]
        src[:, :2, :] += additional_pos_embed

        hs_queries = self.transformer(src, tgt)  # [B, tgt, hidden_dim]

        action_pred = action_head(hs_queries)  # [B, num_queries, action_dim]
        # is_pad_pred = self.is_pad_head(hs_queries)  # [B, num_queries, 1]
        is_pad_pred = None

        # aux action head for 2 head output
        if aux_action_head:
            action_pred_aux = aux_action_head(hs_queries)
            return (action_pred, action_pred_aux), is_pad_pred, [mu, logvar]

        return action_pred, is_pad_pred, [mu, logvar]


class ACT(Algo):
    """
    BC training with a VAE policy.
    # TODO (Simar): Add type checking on these params
    """

    def __init__(
        self,
        data_schematic,
        camera_transforms,
        chunk_size,
        backbones,
        kl_weight,
        train_image_augs,
        eval_image_augs,
        transformer,
        style_encoder,
        latent_dim,
    ):
        super().__init__()

        if len(data_schematic.embodiments) > 1:
            raise ValueError("ACT should only have 1 embodiment")

        self.embodiment_id = list(data_schematic.embodiments)[0]
        self.data_schematic = data_schematic
        self.camera_transforms = camera_transforms
        self.camera_keys = self.data_schematic.keys_of_type("camera_keys")
        self.proprio_keys = self.data_schematic.keys_of_type("proprio_keys")
        if len(self.data_schematic.keys_of_type("action_keys")) > 1:
            raise ValueError("ACT should only have one action key")
        self.ac_key = self.data_schematic.keys_of_type("action_keys")[0]
        self.obs_keys = self.proprio_keys + self.camera_keys

        self.chunk_size = chunk_size

        self.kl_weight = kl_weight
        self.train_image_augs = train_image_augs
        self.eval_image_augs = eval_image_augs

        self.backbones = backbones
        if len(backbones) != len(self.camera_keys):
            raise ValueError(
                f"Number of backbones ({len(backbones)}) doesn't match number of camera_keys ({len(self.camera_keys)}) "
            )

        num_channels = backbones[0].output_shape(
            self.data_schematic.key_shape(self.camera_keys[0], self.embodiment_id)
        )[0]
        a_dim = self.data_schematic.key_shape(self.ac_key, self.embodiment_id)[-1]

        if len(self.proprio_keys) > 1:
            raise ValueError(
                f"Current implementation only supports one proprio_key but got proprio_keys={self.proprio_keys}"
            )
        state_dim = self.data_schematic.key_shape(
            self.proprio_keys[0], self.embodiment_id
        )[-1]

        if len(self.data_schematic.norm_stats.keys()) != 1:
            raise ValueError(
                "ACT expects only single embodiment to be in dataset, instead found embodiment keys: ",
                self.data_schematic.norm_stats.keys(),
            )

        model = ACTModel(
            backbones=backbones,
            transformer=transformer,
            encoder=style_encoder,
            latent_dim=latent_dim,
            a_dim=a_dim,
            state_dim=state_dim,
            num_queries=chunk_size,
            camera_names=self.camera_keys,
            num_channels=num_channels,
        )
        self.nets = nn.ModuleDict()
        self.nets["policy"] = model

    @override
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            batch (dict): processed batch of form
                front_img_1 torch.Size([32, 3, 480, 640])
                right_wrist_img: torch.Size([32, 3, 480, 640])
                joint_positions: torch.Size([32, 1, 7])
                actions_joints_act: torch.Size([32, 100, 7])
                demo_number: torch.Size([32])
                _index: torch.Size([32])
                pad_mask: torch.Size([32, 100, 1])
        """
        processed_batch = {}

        batch = batch[next(iter(batch))]
        for key, value in batch.items():
            key_name = self.data_schematic.lerobot_key_to_keyname(
                key, self.embodiment_id
            )
            if key_name is not None:
                processed_batch[key_name] = value

        if len(processed_batch[self.ac_key][0].shape) != 2:
            raise ValueError("Action shape is not 2")

        B, S, _ = processed_batch[self.ac_key].shape
        device = processed_batch[self.ac_key].device
        processed_batch["pad_mask"] = torch.ones(B, S, 1, device=device)

        processed_batch = self.data_schematic.normalize_data(
            processed_batch, self.embodiment_id
        )

        processed_batch["joint_positions"] = processed_batch["joint_positions"][
            :, None, :
        ]

        return processed_batch

    @override
    def forward_training(self, batch):
        """
        One iteration of training.  Compute forward pass and compute losses.  Return predictions dictionary.  ACT also calculates loss here.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            predictions (dict): {ac_key: torch.Tensor (B, Seq, D), loss_key_name: torch.Tensor (1)}
        """

        qpos, images, env_state, actions, is_pad = self._robomimic_to_act_data(
            batch, self.camera_keys, self.proprio_keys
        )

        a_hat, is_pad_hat, (mu, logvar) = self.nets["policy"](
            qpos=qpos, image=images, env_state=env_state, actions=actions, is_pad=is_pad
        )
        total_kld, dim_wise_kld, mean_kld = self._kl_divergence(mu, logvar)
        loss_dict = dict()
        all_l1 = F.l1_loss(actions, a_hat, reduction="none")
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        loss_dict["l1"] = l1
        loss_dict["kl"] = total_kld[0]

        predictions = OrderedDict(
            actions=actions,
            kl_loss=loss_dict["kl"],
            reconstruction_loss=loss_dict["l1"],
        )

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
            predictions (dict): {ac_key: torch.Tensor (B, Seq, D)}
        """

        qpos, images, env_state, _, is_pad = self._robomimic_to_act_data(
            batch, self.camera_keys, self.proprio_keys
        )
        a_hat, is_pad_hat, (mu, logvar) = self.nets["policy"](
            qpos=qpos, image=images, env_state=env_state, actions=None, is_pad=is_pad
        )

        predictions = OrderedDict()
        predictions[self.ac_key] = a_hat

        unnorm_preds = self.data_schematic.unnormalize_data(
            predictions, self.embodiment_id
        )

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
        # forward_eval will unnormalize predictions
        preds = self.forward_eval(batch)
        # Must unnormalize ground truth as well bc this data came from @process_batch_for_training
        batch = self.data_schematic.unnormalize_data(batch, self.embodiment_id)

        metrics = {}
        mse = MeanSquaredError()
        for ac_key in self.data_schematic.keys_of_type("action_keys"):
            if len(preds[ac_key].shape) != 3:
                raise ValueError("predictions should be (B, Seq, D)")
            metrics[f"Valid/{ac_key}_paired_mse_avg"] = mse(
                preds[ac_key].cpu(), batch[ac_key].cpu()
            )
            metrics[f"Valid/{ac_key}_final_mse_avg"] = mse(
                preds[ac_key][:, -1].cpu(), batch[ac_key][:, -1].cpu()
            )

        ims = {self.embodiment_id: self.visualize_preds(preds, batch)}

        return metrics, ims

    @override
    def visualize_preds(self, predictions, batch):
        """
        Helper function to visualize predictions on top of images
        Args:
            preds (dict): {ac_key: torch.Tensor (B, Seq, D)}
            batch (dict): {ac_key: torch.Tensor (B, Seq, D), front_img_1: torch.Tensor (B, 3, H, W)}
        Returns:
            ims (np.ndarray): (B, H, W, 3) - images with actions drawn on top
        """
        ims = (
            batch[self.data_schematic.viz_img_key()[self.embodiment_id]]
            .cpu()
            .numpy()
            .transpose((0, 2, 3, 1))
            * 255
        ).astype(np.uint8)
        preds = preds[self.data_schematic.action_keys()[0]]
        gt = batch[self.data_schematic.action_keys()[0]]

        for b in range(ims.shape[0]):
            if preds.shape[-1] == 7 or preds.shape[-1] == 14:
                ac_type = "joints"
            elif preds.shape[-1] == 3 or preds.shape[-1] == 6:
                ac_type = "xyz"
            else:
                raise ValueError(f"Unknown action type with shape {preds.shape}")

            arm = "right" if preds.shape[-1] == 7 or preds.shape[-1] == 3 else "both"
            ims[b] = draw_actions(
                ims[b],
                ac_type,
                "Purples",
                preds[b].cpu().numpy(),
                self.camera_transforms.extrinsics,
                self.camera_transforms.intrinsics,
                arm=arm,
            )

            ims[b] = draw_actions(
                ims[b],
                ac_type,
                "Greens",
                gt[b].cpu().numpy(),
                self.camera_transforms.extrinsics,
                self.camera_transforms.intrinsics,
                arm=arm,
            )

        return ims

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

        # total loss is sum of reconstruction and KL, weighted by beta
        kl_loss = predictions["kl_loss"]
        recons_loss = predictions["reconstruction_loss"]
        action_loss = recons_loss + self.kl_weight * kl_loss
        return OrderedDict(
            recons_loss=recons_loss,
            kl_loss=kl_loss,
            action_loss=action_loss,
        )

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
        log["KL_Loss"] = info["losses"]["kl_loss"].item()
        log["Reconstruction_Loss"] = info["losses"]["recons_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def _kl_divergence(self, mu, logvar):
        batch_size = mu.size(0)
        assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        dimension_wise_kld = klds.mean(0)
        mean_kld = klds.mean(1).mean(0, True)

        return total_kld, dimension_wise_kld, mean_kld

    def _robomimic_to_act_data(self, batch, cam_keys, proprio_keys):
        qpos = [batch[k] for k in proprio_keys]
        qpos = torch.cat(qpos, axis=1)
        qpos = qpos[:, 0, :]

        images = []

        if len(cam_keys) > 0:
            for cam_name in cam_keys:
                image = batch[cam_name]
                if self.nets.training:
                    image = self.train_image_augs(image)
                else:
                    image = self.eval_image_augs(image)
                image = image.unsqueeze(axis=1)
                images.append(image)
            images = torch.cat(images, axis=1)
        else:
            images = None

        env_state = torch.zeros([qpos.shape[0], 10]).cuda()  # this is not used

        actions = batch[self.ac_key] if self.ac_key in batch else None
        is_pad = batch["pad_mask"] == 0  # from 1.0 or 0 to False and True
        is_pad = is_pad.squeeze(dim=-1)
        B, T = is_pad.shape
        assert T == self.chunk_size

        return qpos, images, env_state, actions, is_pad
