"""Single-GPU driver for RLinf's released DSRL networks and SAC objective.

The pi0.5 decoder remains the exact native LeRobot model. The released DSRL
networks/methods are imported from the pinned source; no checkpoint conversion
is required for this arm. Scheduling/replay are local, not RLinf's Ray harness.
"""

import copy
import importlib
import types

import numpy as np
import torch
from torch import nn


def noise_network(horizon, device):
    dsrl = importlib.import_module("rlinf.models.embodiment.openpi.tasks.dsrl")
    cfg = dsrl.Pi0DSRLConfig()
    model = nn.Module()
    model.dsrl_cfg = cfg
    model.dsrl_action_noise_net = dsrl.GaussianPolicy(
        input_dim=cfg.state_latent_dim + cfg.image_latent_dim,
        output_dim=cfg.action_noise_dim,
        hidden_dims=cfg.hidden_dims,
        low=None,
        high=None,
        action_horizon=horizon,
    )
    model.actor_image_encoder = dsrl.LightweightImageEncoder64(
        num_images=1, latent_dim=cfg.image_latent_dim, image_size=64
    )
    model.actor_state_encoder = dsrl.CompactStateEncoder(
        state_dim=cfg.state_dim, hidden_dim=cfg.state_latent_dim
    )
    model.critic_image_encoder = dsrl.LightweightImageEncoder64(
        num_images=1, latent_dim=cfg.image_latent_dim, image_size=64
    )
    model.critic_state_encoder = dsrl.CompactStateEncoder(
        state_dim=cfg.state_dim, hidden_dim=cfg.state_latent_dim
    )
    model.q_head = dsrl.CompactMultiQHead(
        state_dim=cfg.state_latent_dim,
        image_dim=cfg.image_latent_dim,
        action_dim=cfg.action_noise_dim,
        hidden_dims=cfg.hidden_dims,
        num_q_heads=cfg.num_q_heads,
        output_dim=1,
    )
    for name in (
        "_normalize_dsrl_obs",
        "_preprocess_dsrl_images",
        "_preprocess_states",
        "_actor_features",
        "sac_forward",
        "sac_q_forward",
    ):
        setattr(model, name, types.MethodType(getattr(dsrl.Pi0DSRL, name), model))
    # Float32 master weights/Adam states; upstream forward casts use bf16 under
    # autocast, preventing small optimizer steps being rounded away in storage.
    return model.to(device=device, dtype=torch.float32)


def stack_observations(observations, device):
    return {
        "images": [
            torch.as_tensor(
                np.stack([o["observation/image"] for o in observations]), device=device
            )
        ],
        "states": torch.as_tensor(
            np.stack([o["observation/state"] for o in observations]),
            device=device,
            dtype=torch.float32,
        ),
    }


class DSRLLearner:
    def __init__(self, policy, *, seed, batch_size=64, updates=200):
        torch.manual_seed(seed)
        self.policy, self.device = policy, policy.device
        self.network = noise_network(policy.horizon, self.device)
        self.target = copy.deepcopy(self.network).eval().requires_grad_(False)
        self.actor_parameters = [
            p
            for name, p in self.network.named_parameters()
            if name.startswith(("actor_", "dsrl_action_noise_net"))
        ]
        self.critic_parameters = [
            p
            for name, p in self.network.named_parameters()
            if name.startswith(("critic_", "q_head"))
        ]
        self.actor_optimizer = torch.optim.Adam(self.actor_parameters, lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=3e-4)
        temperature = importlib.import_module(
            "rlinf.models.embodiment.modules.entropy_tunning"
        ).EntropyTemperature
        self.temperature = temperature(1.0, alpha_type="softplus", device=self.device)
        self.alpha_optimizer = torch.optim.Adam(self.temperature.parameters(), lr=3e-4)
        self.rng, self.replay = np.random.default_rng(seed), []
        self.batch_size, self.update_steps = batch_size, updates
        self.critic_steps, self.version = 0, 0

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)

    @torch.no_grad()
    def noise(self, observation, *, evaluation):
        with self.autocast():
            noise, _, _ = self.network.sac_forward(
                obs=stack_observations([observation], self.device),
                mode="eval" if evaluation else "train",
            )
        return noise.float()

    def update(self, transitions):
        if not transitions or any(not row["executed_actions"] for row in transitions):
            raise ValueError("SAC requires real, nonempty executed transitions")
        self.replay.extend(transitions)
        self.replay = self.replay[-15000:]
        if len(self.replay) < 10:
            return {"updated": False, "replay_transitions": len(self.replay)}
        history = []
        for _ in range(self.update_steps):
            rows = [
                self.replay[i]
                for i in self.rng.integers(0, len(self.replay), self.batch_size)
            ]
            current = stack_observations([r["observation"] for r in rows], self.device)
            following = stack_observations(
                [r["next_observation"] for r in rows], self.device
            )
            actions = torch.stack([r["noise"][0] for r in rows]).to(self.device)
            reward = torch.tensor([r["sac_reward"] for r in rows], device=self.device)[
                :, None
            ]
            duration = torch.tensor(
                [len(r["executed_actions"]) for r in rows], device=self.device
            )[:, None]
            terminal = torch.tensor([r["terminal"] for r in rows], device=self.device)[
                :, None
            ]
            with torch.no_grad(), self.autocast():
                next_noise, _, _ = self.network.sac_forward(obs=following, train=True)
                next_q = (
                    self.target.sac_q_forward(
                        obs=following, actions=next_noise, train=True
                    )
                    .float()
                    .mean(-1, keepdim=True)
                )
                # Released DSRL template: mean over 10 Q heads, no entropy in
                # the Bellman backup, gamma=.999, standard termination mask.
                target = reward + (~terminal) * (0.999**duration) * next_q
            self.critic_optimizer.zero_grad(set_to_none=True)
            with self.autocast():
                qs = self.network.sac_q_forward(
                    obs=current, actions=actions, train=True
                ).float()
                critic_loss = (qs - target).square().mean()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                self.critic_parameters, 10, error_if_nonfinite=True
            )
            self.critic_optimizer.step()
            self.critic_steps += 1
            actor_loss, alpha_loss = None, None
            if self.critic_steps > 10:
                for p in self.critic_parameters:
                    p.requires_grad_(False)
                try:
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    with self.autocast():
                        noise, logprob, _ = self.network.sac_forward(
                            obs=current, train=True
                        )
                        q = (
                            self.network.sac_q_forward(
                                obs=current,
                                actions=noise,
                                detach_encoder=True,
                                train=True,
                            )
                            .float()
                            .mean(-1)
                        )
                        actor_loss = (
                            self.temperature.alpha * logprob.float() - q
                        ).mean()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actor_parameters, 10, error_if_nonfinite=True
                    )
                    self.actor_optimizer.step()
                finally:
                    for p in self.critic_parameters:
                        p.requires_grad_(True)
                self.alpha_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad(), self.autocast():
                    _, logprob, _ = self.network.sac_forward(obs=current, train=True)
                alpha_loss = -self.temperature.compute_alpha() * (
                    logprob.float().mean() - 16
                )
                alpha_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.temperature.parameters(), 10, error_if_nonfinite=True
                )
                self.alpha_optimizer.step()
            with torch.no_grad():
                for target_p, online_p in zip(
                    self.target.parameters(), self.network.parameters()
                ):
                    target_p.lerp_(online_p, 0.005)
            history.append(
                {
                    "critic_loss": float(critic_loss.detach()),
                    "actor_loss": None
                    if actor_loss is None
                    else float(actor_loss.detach()),
                    "alpha_loss": None
                    if alpha_loss is None
                    else float(alpha_loss.detach()),
                    "alpha": self.temperature.alpha,
                }
            )
        self.version += 1
        return {
            "updated": True,
            "policy_version": self.version,
            "replay_transitions": len(self.replay),
            "critic_steps": self.critic_steps,
            "history": history,
        }

    def save(self, path):
        torch.save(
            {
                "network": self.network.state_dict(),
                "target": self.target.state_dict(),
                "temperature": self.temperature.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "version": self.version,
            },
            path,
        )
