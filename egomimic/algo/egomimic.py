"""
Implementation of EgoMimic.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import robomimic.utils.tensor_utils as TensorUtils
from egomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.algo.bc import BC_VAE

from egomimic.utils.egomimicUtils import nds
import matplotlib.pyplot as plt
import robomimic.utils.obs_utils as ObsUtils

from egomimic.configs import config_factory

from egomimic.models.act_nets import Transformer, StyleEncoder

from robomimic.models.transformers import PositionalEncoding

import robomimic.models.base_nets as BaseNets
import egomimic.models.policy_nets as PolicyNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

import json

from egomimic.algo.act import ACT, ACTModel


@register_algo_factory_func("egomimic")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    algo_class, algo_kwargs = EgoMimic, {}
    return algo_class, algo_kwargs


class EgoMimicModel(ACTModel):
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
        super().__init__(
            backbones,
            transformer,
            encoder,
            latent_dim,
            a_dim,
            state_dim,
            num_queries,
            camera_names,
            num_channels
        )

        hidden_dim = transformer.d
        if a_dim == 7:
            hand_state_dim = 3
            hand_action_dim = 3
        elif a_dim == 14:
            hand_state_dim = 6
            hand_action_dim = 6
        
        
        self.robot_transformer_input_proj = nn.Linear(state_dim, hidden_dim)
        self.robot_action_head = nn.Linear(hidden_dim, a_dim)

        self.encoder_action_proj = nn.Linear(
            a_dim, hidden_dim
        )  # project robot action to embedding
        self.encoder_joint_proj = nn.Linear(
            state_dim, hidden_dim
        )  # project robot qpos to embedding

        self.hand_transformer_input_proj = nn.Linear(hand_state_dim, hidden_dim)
        self.hand_encoder_action_proj = nn.Linear(
            hand_action_dim, hidden_dim
        )  # project action to embedding
        self.hand_encoder_joint_proj = nn.Linear(
            hand_state_dim, hidden_dim
        )  # project qpos to embedding
        self.hand_action_head = nn.Linear(hidden_dim, hand_action_dim)

    def forward(self, qpos, image, env_state, modality, actions, is_pad=None):
        if modality == "robot":
            return self._forward(
                qpos,
                actions,
                image,
                self.encoder_action_proj,
                self.encoder_joint_proj,
                self.robot_transformer_input_proj,
                self.robot_action_head,
                camera_names=self.camera_names,
                is_pad=is_pad,
                aux_action_head=self.hand_action_head,
            )
        elif modality == "hand":
            assert "front_img" in self.camera_names[0], "hand modality assumes first camera is front_img"
            return self._forward(
                qpos,
                actions,
                image,
                self.hand_encoder_action_proj,
                self.hand_encoder_joint_proj,
                self.hand_transformer_input_proj,
                self.hand_action_head,
                camera_names=self.camera_names[:1],
                is_pad=is_pad,
            )

class EgoMimic(ACT):
    def build_model_opt(self, policy_config):
        backbones = []
        if len(policy_config["camera_names"]) > 0:
            for cam_name in policy_config["camera_names"]:
                backbone_class_name = policy_config["backbone_class_name"]
                backbone_kwargs = policy_config["backbone_kwargs"]

                try:
                    backbone_class = getattr(BaseNets, backbone_class_name)
                except AttributeError:
                    raise ValueError(f"Unsupported backbone class: {backbone_class_name}")
                
                backbone = backbone_class(**backbone_kwargs)
                backbones.append(backbone)
        else:
            backbones = None

        if backbones is not None:
            # assume camera input shape is same for all TODO dynamic size
            cam_name = policy_config["camera_names"][0]  
            input_shape = self.obs_key_shapes[cam_name]  # (C, H, W)
            num_channels = backbones[0].output_shape(input_shape)[0]
        else:
            num_channels = None

        transformer = Transformer(
            d=policy_config["hidden_dim"],
            h=policy_config["nheads"],
            d_ff=policy_config["dim_feedforward"],
            num_layers=policy_config["dec_layers"],
            dropout=policy_config["dropout"],
        )

        style_encoder = StyleEncoder(
            act_len=policy_config["action_length"],
            hidden_dim=policy_config["hidden_dim"],
            latent_dim=policy_config["latent_dim"],
            h=policy_config["nheads"],
            d_ff=policy_config["dim_feedforward"],
            num_layers=policy_config["enc_layers"],
            dropout=policy_config["dropout"],
        )

        model = EgoMimicModel(
            backbones=backbones,
            transformer=transformer,
            encoder=style_encoder,
            latent_dim=policy_config["latent_dim"],
            a_dim=policy_config["a_dim"],
            state_dim=policy_config["state_dim"],
            num_queries=policy_config["num_queries"],
            camera_names=policy_config["camera_names"],
            num_channels=num_channels,
        )

        model.cuda()

        return model

    def _create_networks(self):
        super(EgoMimic, self)._create_networks()
        self.proprio_keys_hand = (
            self.global_config.observation_hand.modalities.obs.low_dim.copy()
        )

        self.ac_key_hand = self.global_config.train.ac_key_hand
        self.ac_key_robot = self.global_config.train.ac_key

        # self.proprio_dim = 0
        # for k in self.proprio_keys_hand:
        #     self.proprio_dim_hand += self.obs_key_shapes[k][0]
    
    def process_batch_for_training(self, batch, ac_key):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """

        input_batch = dict()
        input_batch["obs"] = {
            k: batch["obs"][k][:, 0, :]
            for k in batch["obs"]
            if k != "pad_mask" and k != "type"
        }
        input_batch["obs"]["pad_mask"] = batch["obs"]["pad_mask"]
        input_batch["goal_obs"] = batch.get(
            "goal_obs", None
        )  # goals may not be present

        if self.ac_key_hand in batch:
            input_batch[self.ac_key_hand] = batch[self.ac_key_hand]
        if self.ac_key_robot in batch:
            input_batch[self.ac_key_robot] = batch[self.ac_key_robot]

        if "type" in batch:
            input_batch["type"] = batch["type"]

        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def _robomimic_to_act_data(self, batch, cam_keys, proprio_keys):
        qpos, images, env_state, actions, is_pad = super()._robomimic_to_act_data(batch, cam_keys, proprio_keys)
        actions_hand = batch.get(self.ac_key_hand, None)
        actions_robot = batch[self.ac_key_robot] if self.ac_key_robot in batch else None

        return qpos, images, env_state, actions_hand, actions_robot, is_pad

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            predictions (dict): dictionary containing network outputs
        """

        modality = self._modality_check(batch)
        cam_keys = (
            self.camera_keys if modality == "robot" else self.camera_keys[:1]
        )  # TODO Simar rm hardcoding
        proprio_keys = (
            self.proprio_keys_hand if modality == "hand" else self.proprio_keys
        )
        qpos, images, env_state, actions_hand, actions_robot, is_pad = self._robomimic_to_act_data(
            batch, cam_keys, proprio_keys
        )
    
        actions = actions_hand if modality == "hand" else actions_robot

        a_hat, is_pad_hat, (mu, logvar) = self.nets["policy"](
            qpos, images, env_state, modality, actions=actions, is_pad=is_pad
        )
        total_kld, dim_wise_kld, mean_kld = self.kl_divergence(mu, logvar)
        loss_dict = dict()

        if modality == "hand":
            all_l1 = F.l1_loss(actions_hand, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean() * self.global_config.algo.sp.hand_lambda
            total_kld = total_kld * self.global_config.algo.sp.hand_lambda
        elif modality == "robot":
            all_l1_robot = F.l1_loss(actions_robot, a_hat[0], reduction="none")
            all_l1_hand = F.l1_loss(actions_hand, a_hat[1], reduction="none")
            l1 = (all_l1_robot * ~is_pad.unsqueeze(-1)).mean() + (all_l1_hand * ~is_pad.unsqueeze(-1)).mean()

        loss_dict["l1"] = l1
        loss_dict["kl"] = total_kld[0]

        predictions = OrderedDict(
            actions=actions,
            kl_loss=loss_dict["kl"],
            reconstruction_loss=loss_dict["l1"],
        )

        return predictions

    def forward_eval(self, batch, unnorm_stats):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            predictions (dict): dictionary containing network outputs
        """

        modality = self._modality_check(batch)

        cam_keys = (
            self.camera_keys if modality == "robot" else self.camera_keys[:1]
        )  # TODO Simar rm hardcoding
        proprio_keys = (
            self.proprio_keys_hand if modality == "hand" else self.proprio_keys
        )
        qpos, images, env_state, _, _, is_pad = self._robomimic_to_act_data(
            batch, cam_keys, proprio_keys
        )
        a_hat, is_pad_hat, (mu, logvar) = self.nets["policy"](
            qpos, images, env_state, modality, actions=None, is_pad=is_pad
        )

        # a_hat = a_hat[0] if modality == "robot" else a_hat
        if modality == "robot":
            predictions = OrderedDict()
            predictions[self.ac_key_robot] = a_hat[0]
            predictions[self.ac_key_hand] = a_hat[1]
            predictions = ObsUtils.unnormalize_batch(predictions, unnorm_stats)
        else:
            predictions = OrderedDict()
            predictions[self.ac_key_hand] = a_hat
            predictions = ObsUtils.unnormalize_batch(predictions, unnorm_stats)

        return predictions
