"""RLinf's stochastic flow sampler/recompute with a serial PPO/GAE driver.

Uses the strict converted checkpoint and documented GELU compatibility context.
The objective matches released clipped PPO and Huber value regression. This is
an integration harness, not a claim to reproduce RLinf's distributed throughput.
"""

import importlib
import random
import types

import numpy as np
import torch

from .rlinf_bridge import observation_from_native


def advantages_and_returns(rewards, values, terminals, gamma=0.99, gae_lambda=0.95):
    if values.shape != (len(rewards) + 1,) or terminals.shape != rewards.shape:
        raise ValueError("GAE requires N rewards/terminals and N+1 values")
    advantages = torch.empty_like(rewards)
    carry = torch.zeros((), device=rewards.device)
    for i in reversed(range(len(rewards))):
        alive = (~terminals[i]).float()
        delta = rewards[i] + gamma * values[i + 1] * alive - values[i]
        carry = delta + gamma * gae_lambda * alive * carry
        advantages[i] = carry
    returns = advantages + values[:-1]
    if len(rewards) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
    return advantages, returns


def clipped_losses(logprob, old_logprob, advantage, value, old_value, returns):
    ratio = torch.exp(logprob - old_logprob)
    actor = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(0.8, 1.2)).mean()
    clipped_value = old_value + (value - old_value).clamp(-0.2, 0.2)

    def huber(error):
        return torch.where(
            error.abs() < 10, 0.5 * error.square(), 10 * (error.abs() - 5)
        )

    critic = torch.maximum(
        huber(returns - value), huber(returns - clipped_value)
    ).mean()
    return actor, critic


def attach_released_ppo(core, model, native):
    """Attach the released task methods/heads after strict base-weight loading."""
    module = importlib.import_module("rlinf.models.embodiment.openpi.tasks.rl")
    # Pi0RL adds task attributes and a value head to the same Pi0 backbone.
    # Reuse that exact class, avoiding a second allocation of its full weights.
    model.__class__ = module.Pi0RL
    model.model_action_dim = model.action_dim
    model.global_step = 0
    model.action_chunk, model.action_env_dim, model.num_steps = 5, 7, 10
    model.rl_cfg = module.Pi0RLConfig(
        add_value_head=True,
        noise_method="flow_sde",
        noise_level=0.5,
        value_after_vlm=True,
        train_expert_only=True,
        config_name="pi05_libero",
    )
    model.value_head = module.ValueHead(
        input_dim=2048,
        hidden_sizes=(1024, 512, 256),
        output_dim=1,
        activation="relu",
        bias_last=True,
    ).to(native.device)
    model.decode_actions = types.MethodType(
        lambda self, actions, state: native.postprocessor(actions[:, :, :7])[:, :5],
        model,
    )
    model.requires_grad_(True)
    model.freeze_vlm(freeze_action_expert=False)
    return model


class PPOLearner:
    def __init__(self, core, model, native, *, seed, epochs=1, optimizer_batch=8):
        random.seed(seed)
        torch.manual_seed(seed)
        self.core, self.policy = core, native
        self.device = native.device
        self.model = attach_released_ppo(core, model, native)
        self.parameters = {
            name: p for name, p in self.model.named_parameters() if p.requires_grad
        }
        actor = [
            p for n, p in self.parameters.items() if not n.startswith("value_head.")
        ]
        critic = [p for n, p in self.parameters.items() if n.startswith("value_head.")]
        self.optimizer = torch.optim.AdamW(
            [{"params": actor, "lr": 5e-6}, {"params": critic, "lr": 1e-4}],
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01,
        )
        self.model.eval().requires_grad_(False)
        self.rng = np.random.default_rng(seed)
        self.epochs, self.optimizer_batch, self.version = epochs, optimizer_batch, 0

    def observation(self, observation, instruction):
        batch = self.policy._preprocess({**observation, "prompt": instruction})
        return observation_from_native(self.core, self.policy, batch)

    @torch.no_grad()
    def proposal(self, observation, instruction, *, evaluation, rng):
        obs = self.observation(observation, instruction)
        noise = self.policy.noise(rng)
        if evaluation:
            actions = self.model.sample_actions(obs, num_steps=10, noise=noise)
            return actions, {"noise": noise.cpu(), "ppo": None}
        _, result = self.model._predict_train(
            obs, noise=noise, rng=None, compute_values=True
        )
        forward = result["forward_inputs"]
        forward.update({f"obs_image__{k}": v for k, v in obs.images.items()})
        forward.update({f"obs_image_mask__{k}": v for k, v in obs.image_masks.items()})
        forward["obs_state"] = obs.state
        record = {
            "forward_inputs": {k: v.detach().cpu() for k, v in forward.items()},
            "old_logprobs": result["prev_logprobs"].detach().cpu(),
            "old_value": float(result["prev_values"][0, 0]),
        }
        return result["model_actions"], {"noise": noise.cpu(), "ppo": record}

    def recompute(self, record):
        inputs = {k: v.to(self.device) for k, v in record["forward_inputs"].items()}
        return self.model.default_forward(inputs, compute_values=True)

    def preflight(self, observation, instruction):
        _, receipt = self.proposal(
            observation, instruction, evaluation=False, rng=np.random.default_rng(173)
        )
        with torch.no_grad():
            current = self.recompute(receipt["ppo"])
        old = receipt["ppo"]["old_logprobs"].to(self.device)
        error = float((old - current["logprobs"]).abs().max())
        if error > 1e-4 or not torch.isfinite(current["logprobs"]).all():
            raise RuntimeError(
                "Released PPO rollout and recomputed log-probabilities differ"
            )
        try:
            for p in self.parameters.values():
                p.requires_grad_(True)
            current = self.recompute(receipt["ppo"])
            loss = -current["logprobs"].mean() + current["values"].square().mean()
            loss.backward()
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.parameters.values(), 1, error_if_nonfinite=True
                )
            )
            if norm <= 0:
                raise RuntimeError("PPO backward has no gradient")
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.requires_grad_(False)
        return {
            "logprob_max_abs": error,
            "gradient_norm": norm,
            "policy_updates": 0,
            "environment_actions": 0,
            "trainable_parameters": sum(p.numel() for p in self.parameters.values()),
        }

    def update(self, transitions):
        if not transitions or any(not row["executed_actions"] for row in transitions):
            raise ValueError("PPO requires actual executed transitions")
        rewards = torch.tensor(
            [r["ppo_reward"] for r in transitions], device=self.device
        )
        values = torch.tensor(
            [r["ppo"]["old_value"] for r in transitions] + [0.0], device=self.device
        )
        terminal = torch.tensor(
            [r["terminal"] for r in transitions], device=self.device
        )
        advantages, returns = advantages_and_returns(rewards, values, terminal)
        history = []
        try:
            for p in self.parameters.values():
                p.requires_grad_(True)
            for _ in range(self.epochs):
                order = self.rng.permutation(len(transitions))
                for start in range(0, len(order), self.optimizer_batch):
                    batch = order[start : start + self.optimizer_batch]
                    self.optimizer.zero_grad(set_to_none=True)
                    actor_losses, critic_losses = [], []
                    for index in batch:
                        row = transitions[index]
                        result = self.recompute(row["ppo"])
                        count = len(row["executed_actions"])
                        lp = result["logprobs"][:, :count].sum()
                        old_lp = (
                            row["ppo"]["old_logprobs"][:, :count].to(self.device).sum()
                        )
                        actor, critic = clipped_losses(
                            lp,
                            old_lp,
                            advantages[index],
                            result["values"].reshape(()),
                            values[index],
                            returns[index],
                        )
                        loss = (actor + critic) / len(batch)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Non-finite PPO loss")
                        loss.backward()
                        actor_losses.append(float(actor.detach()))
                        critic_losses.append(float(critic.detach()))
                    norm = torch.nn.utils.clip_grad_norm_(
                        self.parameters.values(), 1, error_if_nonfinite=True
                    )
                    self.optimizer.step()
                    history.append(
                        {
                            "actor_loss": float(np.mean(actor_losses)),
                            "critic_loss": float(np.mean(critic_losses)),
                            "gradient_norm": float(norm),
                        }
                    )
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.eval().requires_grad_(False)
        self.version += 1
        return {
            "updated": True,
            "policy_version": self.version,
            "transitions": len(transitions),
            "epochs": self.epochs,
            "history": history,
        }

    def save(self, path):
        torch.save(
            {
                "parameters": {
                    name: p.detach().cpu() for name, p in self.parameters.items()
                },
                "version": self.version,
                "rl_cfg": self.model.rl_cfg.__dict__,
                "optimizer_state_omitted": True,
            },
            path,
        )
