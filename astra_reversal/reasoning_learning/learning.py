"""Native PI05 flow matching on observed-useful windows, using action-expert LoRA.

Adapters live inside the existing policy. Targets are fixed executed commands;
fresh noise and the checkpoint's native time distribution are sampled per update.
No teacher input, original demonstrations, residual action controller or FRS is
needed for autonomous inference after updating these parameters.
"""

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from astra_reversal.lerobot_policy import prepare_velocity
from astra_reversal.records import digest, file_sha256, to_numpy

from .evidence import balanced_window_indices


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        if not isinstance(base, nn.Linear) or type(rank) is not int or rank < 1:
            raise ValueError("LoRA requires a linear layer and positive rank")
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Parameter(
            torch.empty(
                rank, base.in_features, device=base.weight.device, dtype=torch.float32
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                base.out_features, rank, device=base.weight.device, dtype=torch.float32
            )
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = 1.0

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        native = self.base(x)
        delta = torch.nn.functional.linear(
            torch.nn.functional.linear(x.float(), self.lora_a), self.lora_b
        )
        return (native.float() + self.scale * delta).to(native.dtype)


def install_action_adapters(policy, rank=8):
    """Only action transformer layers; zero B preserves the starting policy."""
    model = policy.model
    model.requires_grad_(False)
    root = model.paligemma_with_expert.gemma_expert.model
    names = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    replaced = []
    for path, module in list(root.named_modules()):
        if path.rsplit(".", 1)[-1] in names and isinstance(module, nn.Linear):
            parent_path, _, child = path.rpartition(".")
            setattr(root.get_submodule(parent_path), child, LoRALinear(module, rank))
            replaced.append(path)
    if not replaced:
        raise ValueError("No native action-expert linear layers matched")
    policy.policy.eval().requires_grad_(False)
    policy.metadata["adaptation"] = {
        "kind": "action_expert_lora",
        "rank": rank,
        "modules": replaced,
        "initially_zero": True,
    }
    return replaced


class NativeLearner:
    def __init__(self, policy, *, rank=8, learning_rate=1e-4, seed=173):
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Positive finite learning rate required")
        self.policy = policy
        torch.manual_seed(seed)
        self.modules = install_action_adapters(policy, rank)
        self.parameters = {
            name: p
            for name, p in policy.model.named_parameters()
            if name.endswith(("lora_a", "lora_b"))
        }
        self.optimizer = torch.optim.AdamW(
            self.parameters.values(), lr=learning_rate, weight_decay=0
        )
        self.rng = np.random.default_rng(seed)
        self.version = 0
        self.seed, self.rank = seed, rank

    def update(self, windows, observations, *, updates=20):
        """Train at rollout boundaries. Complete executed windows only (beta=0)."""
        if any(
            row.get("source") != "executed_commands"
            or row.get("evidence") != "observed_useful"
            for row in windows
        ):
            raise ValueError("Default learner admits only observed-useful real windows")
        indices = balanced_window_indices(windows, updates, self.rng)
        before = digest({name: to_numpy(p) for name, p in self.parameters.items()})
        history = []
        try:
            for p in self.parameters.values():
                p.requires_grad_(True)
            # eval() removes dropout; it does not turn off autograd.
            self.policy.policy.eval()
            for index in indices:
                row = windows[index]
                raw = observations[row["observation_id"]]
                if digest(raw) != row["observation_id"]:
                    raise ValueError("Training observation identity changed")
                actions = np.asarray(row["actions"], dtype=np.float32)
                if actions.shape != (self.policy.horizon, 7):
                    raise ValueError(
                        "Native training requires a complete action horizon"
                    )
                batch = self.policy._preprocess({**raw, "actions": actions})
                target = self.policy.policy.prepare_action(batch).detach()
                # Prefix is detached and never changes during an action-expert
                # update; suffix velocity is exactly the native model's head.
                velocity = prepare_velocity(
                    self.policy.policy, batch, differentiable=True
                )
                noise = self.policy.model.sample_noise(target.shape, target.device)
                times = self.policy.model.sample_time(target.shape[0], target.device)
                if times.shape != (1,):
                    raise ValueError("This pilot uses one observation per update")
                t = float(times[0])
                x_t = t * noise + (1 - t) * target
                prediction = velocity(x_t, t)
                loss = (prediction - (noise - target)).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite native flow-matching loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    list(self.parameters.values()), 1.0, error_if_nonfinite=True
                )
                self.optimizer.step()
                history.append(
                    {
                        "window_id": row["window_id"],
                        "flow_time": t,
                        "loss": float(loss.detach()),
                        "gradient_norm": float(norm),
                    }
                )
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.policy.policy.eval().requires_grad_(False)
        after = digest({name: to_numpy(p) for name, p in self.parameters.items()})
        if after == before:
            raise RuntimeError("No action-expert parameter changed during the update")
        self.version += 1
        self.policy.metadata["frozen"] = False
        self.policy.metadata["adaptation"]["initially_zero"] = False
        self.policy.metadata["adaptation"]["policy_version"] = self.version
        return {
            "policy_version": self.version,
            "before_sha256": before,
            "after_sha256": after,
            "trainable_parameters": sum(p.numel() for p in self.parameters.values()),
            "objective": "native_flow_matching_noise_minus_actions",
            "replay_beta": 0,
            "history": history,
        }

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "action_expert_adapters.pt"
        torch.save(
            {
                "schema": "reasoning-action-lora-1",
                "rank": self.rank,
                "seed": self.seed,
                "version": self.version,
                "modules": self.modules,
                "parameters": {
                    name: p.detach().cpu() for name, p in self.parameters.items()
                },
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        return {
            "file": str(path),
            "sha256": file_sha256(path),
            "policy_version": self.version,
        }


def load_adapters(policy, path, expected_sha256):
    """Autonomous deployment needs the base checkpoint plus this small artifact."""
    if file_sha256(path) != expected_sha256:
        raise ValueError("Learned adapter checkpoint hash differs")
    saved = torch.load(path, map_location=policy.device, weights_only=True)
    if saved["schema"] != "reasoning-action-lora-1":
        raise ValueError("Unknown adapter schema")
    names = install_action_adapters(policy, saved["rank"])
    if names != saved["modules"]:
        raise ValueError("Learned adapter architecture differs")
    current = {
        name: p
        for name, p in policy.model.named_parameters()
        if name.endswith(("lora_a", "lora_b"))
    }
    if current.keys() != saved["parameters"].keys():
        raise ValueError("Learned adapter keys differ")
    with torch.no_grad():
        for name, p in current.items():
            p.copy_(saved["parameters"][name])
    policy.policy.eval().requires_grad_(False)
    policy.metadata["frozen"] = False
    policy.metadata["adaptation"].update(
        initially_zero=False, policy_version=saved["version"]
    )
    return saved["version"]
