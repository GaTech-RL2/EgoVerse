import torch
import torch.nn as nn
from torchvision import transforms

class polciy_wrapper(torch.nn.Module):
    def __init__(self, policy, norm_stats):
        super().__init__()
        self.policy = policy
        self.norm_stats = norm_stats
        for k, stats in self.norm_stats.items():
            for stat_name, tensor in stats.items():
                self.norm_stats[k][stat_name] = tensor.cuda()
        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
        self.image_std = torch.tensor([0.229, 0.224, 0.225]).cuda()

    def normalize(self, data):
        normalized = {}
        for key, value in data.items():
            norm_key = None
            if key in self.norm_stats:
                norm_key = key
            elif key.startswith("state_") and key[6:] in self.norm_stats:
                norm_key = key[6:]

            if norm_key is not None:
                mean = self.norm_stats[norm_key]["mean"]
                std = self.norm_stats[norm_key]["std"]
                normalized[key] = (value - mean) / (std+1e-5)
            else:
                normalized[key] = value

        return normalized

    def unnormalize(self, data):
        unnormalized = {}
        for key, value in data.items():
            norm_key = None
            if key in self.norm_stats:
                norm_key = key
            elif key.startswith("state_") and key[6:] in self.norm_stats:
                norm_key = key[6:]

            if norm_key is not None:
                mean = self.norm_stats[norm_key]["mean"]
                std = self.norm_stats[norm_key]["std"]
                unnormalized[key] = value * (std+1e-5) + mean
            else:
                unnormalized[key] = value

        return unnormalized

    def image_aug(self, image):
        mean = self.image_mean.view(1, -1, 1, 1)
        std = self.image_std.view(1, -1, 1, 1)
        return (image - mean)/std

    def forward(self, data):
        domain = 'ucsd_h1_bimanual'
        data = {
            'h1_ee_pose': data['observations.state.ee_pose'].unsqueeze(1),
            'h1_left_wrist_6d_rot': data['observations.state.ucsd_h1.left_wrist_6d_rot'].unsqueeze(1),
            'h1_right_wrist_6d_rot': data['observations.state.ucsd_h1.right_wrist_6d_rot'].unsqueeze(1),
            'h1_left_hand_kpts': data['observations.state.ucsd_h1.left_hand_kpts'].unsqueeze(1),
            'h1_right_hand_kpts': data['observations.state.ucsd_h1.right_hand_kpts'].unsqueeze(1),
            'front_img_1': self.image_aug(data['observations.images.front_img_1']).unsqueeze(1).unsqueeze(1),
            'front_img_2': self.image_aug(data['observations.images.front_img_2']).unsqueeze(1).unsqueeze(1),
            'actions_cartesian': data['actions_cartesian'],
            'left_hand_kpts': data['actions.ucsd_h1.left_hand_kpts'],
            'right_hand_kpts': data['actions.ucsd_h1.right_hand_kpts'],
            'left_wrist_6d_rot': data['actions.ucsd_h1.left_wrist_6d_rot'],
            'right_wrist_6d_rot': data['actions.ucsd_h1.right_wrist_6d_rot'],
            'head_6d_rot': data['actions.ucsd_h1.head_6d_rot'],
            'embodiment': data['metadata.embodiment'],
        }
        B, S, _ = data['actions_cartesian'].shape
        data['pad_mask'] = torch.ones(B, S, 1)
        
        # normalize data
        data = self.normalize(data)
        data['action'] = data.pop('actions_cartesian')
        data = {
            ("state_" + k) if k.startswith("h1") else k: v
            for k, v in data.items()
        }

        # policy
        action = self.policy.forward(domain, data)
        action['actions_cartesian'] = action.pop('ucsd_h1_bimanual')

        # unnormalize data
        action = self.unnormalize(action)

        return action