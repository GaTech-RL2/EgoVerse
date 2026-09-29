"""Distill recorded text choices into a small native-observation selector.

No provider is imported or called. The learned gate imitates where successful
teachers edited conditioning; it does not estimate a calibrated failure risk.
"""

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .records import digest, file_sha256, to_numpy

SCHEMA = "recipe-selector-1.0"
FEATURE_SCHEMA = "native-last-prefix-visual-instruction-state8-1.0"


def canonical_choice(choice):
    """Preserve exact TEI/TLI operators, including reversed-pair equivalence."""
    if choice == {"operator": "native"}:
        return dict(choice)
    if set(choice) != {"operator", "source_a_id", "source_b_id", "alpha"}:
        raise ValueError("Unexpected selector choice fields")
    operator = choice["operator"]
    a, b, alpha = choice["source_a_id"], choice["source_b_id"], choice["alpha"]
    if (
        operator not in ("tei", "tli")
        or not isinstance(a, str)
        or not isinstance(b, str)
        or not a
        or not b
        or type(alpha) not in (int, float)
        or not np.isfinite(alpha)
        or not 0 <= alpha <= 1
    ):
        raise ValueError("Invalid selector choice")
    if operator == "tli" and (alpha == 0.5 or a == b):
        return {"operator": "native"}
    if a > b:
        a, b, alpha = b, a, 1.0 - alpha
    return {
        "operator": operator,
        "source_a_id": a,
        "source_b_id": b,
        "alpha": float(alpha),
    }


def class_key(choice):
    value = canonical_choice(choice)
    return tuple(value[k] for k in ("operator", "source_a_id", "source_b_id"))


def pooled_features(hidden, padding, instruction, state):
    """Pool only valid native visual slots and original instruction slots."""
    hidden = np.asarray(hidden, np.float32)
    padding, instruction = np.asarray(padding), np.asarray(instruction)
    state = np.asarray(state, np.float32)
    if (
        hidden.ndim != 3
        or hidden.shape[0] != 1
        or padding.shape != hidden.shape[:2]
        or padding.dtype != np.bool_
        or instruction.ndim != 2
        or instruction.shape[0] != 1
        or instruction.dtype != np.bool_
        or state.shape != (8,)
        or not np.isfinite(hidden).all()
        or not np.isfinite(state).all()
    ):
        raise ValueError("Invalid native selector features")
    start = hidden.shape[1] - instruction.shape[1]
    if (
        start <= 0
        or not padding[0, :start].any()
        or not instruction.any()
        or np.any(instruction & ~padding[:, start:])
    ):
        raise ValueError("Selector masks do not describe native prefix slots")
    return np.concatenate(
        (
            hidden[0, :start][padding[0, :start]].mean(axis=0),
            hidden[0, start:][instruction[0]].mean(axis=0),
            state,
        )
    ).astype(np.float32)


def prepare_native_features(policy, observation, prompt):
    """Capture one native prefix without changing its embeddings or velocity."""
    from .interpolation_conditioning import scoped_post_block_hooks
    from .lerobot_policy import prepare_velocity
    from .policy_adapter import Condition

    started = time.perf_counter()
    raw, batch, instruction, layers, _ = policy._interpolation_inputs(
        observation, prompt
    )
    captured = {}

    def prefix(embeddings, padding, attention):
        captured["padding"] = to_numpy(padding).copy()
        return embeddings

    def hidden(index, values):
        captured["hidden"] = to_numpy(values).copy()

    with scoped_post_block_hooks([layers[-1]], hidden):
        velocity = prepare_velocity(policy.policy, batch, prefix_transform=prefix)
    features = pooled_features(
        captured["hidden"],
        captured["padding"],
        instruction,
        observation["observation/state"],
    )
    elapsed = time.perf_counter() - started
    condition = Condition(
        digest(raw),
        digest(observation),
        prompt,
        raw,
        batch["observation.state"],
        velocity,
        elapsed,
    )
    return (
        condition,
        features,
        {
            "schema_version": FEATURE_SCHEMA,
            "feature_sha256": digest(features),
            "raw_observation_sha256": digest(observation),
            "original_prompt_sha256": digest(prompt),
            "condition_id": condition.condition_id,
            "prefix_evaluations": 1,
            "velocity_evaluations": 0,
            "seconds": elapsed,
        },
    )


def selector_config():
    return {
        "steps": 1000,
        "batch_size": 64,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "hidden_widths": [128, 64],
        "seed": 61,
        "gate_threshold": 0.5,
        "gate_tie": "native",
        "alpha_loss_weight": 1.0,
        "feature_std_floor": 0.05,
        "normalized_feature_clip": 10.0,
        "sampling": "equal correction/native groups; equal trajectory weights within each group",
        "objective": "gate_BCE + positive_pair_CE + positive_alpha_MSE",
    }


class _Network(nn.Module):
    def __init__(self, dimension, classes):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(dimension, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU()
        )
        self.gate = nn.Linear(64, 1)
        self.pair = nn.Linear(64, classes)
        self.alpha = nn.Linear(64, classes)

    def forward(self, features):
        hidden = self.shared(features)
        return self.gate(hidden).squeeze(-1), self.pair(hidden), self.alpha(hidden)


class PhaseSelector:
    def __init__(self, dimension, classes, *, device="cuda"):
        if dimension < 1 or not classes or len(set(classes)) != len(classes):
            raise ValueError("Selector requires a dimension and distinct classes")
        self.device = torch.device(device)
        self.classes = tuple(tuple(c) for c in classes)
        for operator, a, b in self.classes:
            canonical_choice(
                {
                    "operator": operator,
                    "source_a_id": a,
                    "source_b_id": b,
                    "alpha": 0.25,
                }
            )
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(61)
            self.model = _Network(dimension, len(classes)).to(self.device)
        self.mean = np.zeros(dimension, np.float32)
        self.scale = np.ones(dimension, np.float32)
        self.training = None

    @classmethod
    def for_choices(cls, features, choices, *, device="cuda"):
        classes = sorted(
            {
                class_key(c)
                for c in choices
                if canonical_choice(c)["operator"] != "native"
            }
        )
        return cls(np.asarray(features).shape[1], classes, device=device)

    def _inputs(self, features):
        array = np.asarray(features, np.float32)
        if (
            array.ndim != 2
            or array.shape[1:] != self.mean.shape
            or not np.isfinite(array).all()
        ):
            raise ValueError("Selector feature dimension or values differ")
        values = np.clip((array - self.mean) / self.scale, -10, 10)
        return torch.as_tensor(values, device=self.device)

    def fit(
        self, features, choices, trajectory_ids, *, source_sha256, _test_steps=None
    ):
        if self.training is not None:
            raise ValueError("Refusing to silently refit a frozen selector")
        if self.device.type != "cuda" and _test_steps is None:
            raise RuntimeError("Production selector training requires an allocated GPU")
        values = np.asarray(features, np.float32)
        if (
            values.ndim != 2
            or values.shape[1:] != self.mean.shape
            or len(values) != len(choices)
            or len(values) != len(trajectory_ids)
            or not np.isfinite(values).all()
            or len(source_sha256) != 64
        ):
            raise ValueError("Invalid selector training corpus")
        choices = [canonical_choice(c) for c in choices]
        positive = np.array([c["operator"] != "native" for c in choices])
        if not positive.any() or positive.all():
            raise ValueError("Selector needs both intervention and native anchors")
        self.mean = values.mean(axis=0)
        self.scale = np.maximum(values.std(axis=0), 0.05)
        x = self._inputs(values)
        y_gate = torch.as_tensor(positive, dtype=torch.float32, device=self.device)
        class_ids = np.array(
            [
                self.classes.index(class_key(c)) if p else 0
                for c, p in zip(choices, positive, strict=True)
            ]
        )
        y_pair = torch.as_tensor(class_ids, dtype=torch.long, device=self.device)
        y_alpha = torch.as_tensor(
            [c.get("alpha", 0.5) for c in choices],
            dtype=torch.float32,
            device=self.device,
        )
        rng = np.random.default_rng(61)
        groups = [np.flatnonzero(positive), np.flatnonzero(~positive)]
        probabilities = []
        for indexes in groups:
            counts = {}
            for i in indexes:
                counts[trajectory_ids[i]] = counts.get(trajectory_ids[i], 0) + 1
            weights = np.array([1.0 / counts[trajectory_ids[i]] for i in indexes])
            probabilities.append(weights / weights.sum())
        steps = 1000 if _test_steps is None else _test_steps
        if type(steps) is not int or not 1 <= steps <= 1000:
            raise ValueError("Invalid selector test budget")
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.001, weight_decay=0.0001
        )
        started, trace = time.perf_counter(), []
        self.model.train()
        for step in range(steps):
            chosen = np.concatenate(
                [
                    rng.choice(g, 32, p=p)
                    for g, p in zip(groups, probabilities, strict=True)
                ]
            )
            indexes = torch.as_tensor(chosen, device=self.device)
            gate, pair, alpha = self.model(x[indexes])
            gate_loss = nn.functional.binary_cross_entropy_with_logits(
                gate, y_gate[indexes]
            )
            pair_loss = nn.functional.cross_entropy(pair[:32], y_pair[indexes[:32]])
            predicted_alpha = (
                torch.sigmoid(alpha[:32])
                .gather(1, y_pair[indexes[:32], None])
                .squeeze(1)
            )
            alpha_loss = nn.functional.mse_loss(predicted_alpha, y_alpha[indexes[:32]])
            loss = gate_loss + pair_loss + alpha_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite selector loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            if step % 50 == 0 or step + 1 == steps:
                trace.append(
                    {
                        "step": step + 1,
                        "loss": float(loss.detach()),
                        "gate_bce": float(gate_loss.detach()),
                        "pair_ce": float(pair_loss.detach()),
                        "alpha_mse": float(alpha_loss.detach()),
                    }
                )
        self.model.eval().requires_grad_(False)
        self.training = {
            "schema_version": SCHEMA,
            "config": selector_config(),
            "source_sha256": source_sha256,
            "training_input_sha256": digest(
                {
                    "features": values,
                    "choices": choices,
                    "trajectories": list(trajectory_ids),
                }
            ),
            "samples": len(values),
            "positive_samples": int(positive.sum()),
            "trajectories": len(set(trajectory_ids)),
            "optimizer_steps": steps,
            "test_only": _test_steps is not None,
            "loss_trace": trace,
            "wall_seconds": time.perf_counter() - started,
        }
        return copy.deepcopy(self.training)

    def predict(self, features):
        if self.training is None:
            raise ValueError("Selector is not fitted")
        with torch.no_grad():
            gate, pair, alpha = self.model(self._inputs(np.asarray(features)[None]))
            if not all(bool(torch.isfinite(v).all()) for v in (gate, pair, alpha)):
                raise FloatingPointError("Nonfinite selector prediction")
            probability = float(torch.sigmoid(gate)[0])
            index = int(pair[0].argmax())
            value = float(torch.sigmoid(alpha)[0, index])
        operator, a, b = self.classes[index]
        choice = (
            canonical_choice(
                {
                    "operator": operator,
                    "source_a_id": a,
                    "source_b_id": b,
                    "alpha": value,
                }
            )
            if probability > 0.5
            else {"operator": "native"}
        )
        return {
            "choice": choice,
            "gate_probability": probability,
            "gate_active": probability > 0.5,
        }

    def save(self, directory):
        if self.training is None:
            raise ValueError("Cannot publish an unfitted selector")
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state, root / "weights.pt")
        np.savez(root / "normalization.npz", mean=self.mean, scale=self.scale)
        meta = {
            "schema_version": SCHEMA,
            "feature_schema": FEATURE_SCHEMA,
            "dimension": len(self.mean),
            "classes": self.classes,
            "training": self.training,
            "files": {
                name: file_sha256(root / name)
                for name in ("weights.pt", "normalization.npz")
            },
            "implementation_sha256": file_sha256(__file__),
        }
        (root / "metadata.json").write_text(
            json.dumps(meta, indent=2, allow_nan=False) + "\n"
        )
        return meta

    @classmethod
    def load(cls, directory, *, device="cuda", allow_test_only=False):
        root = Path(directory)
        meta = json.loads((root / "metadata.json").read_text())
        if (
            meta["schema_version"] != SCHEMA
            or meta["feature_schema"] != FEATURE_SCHEMA
            or meta["implementation_sha256"] != file_sha256(__file__)
        ):
            raise ValueError("Selector checkpoint implementation mismatch")
        if set(meta["files"]) != {"weights.pt", "normalization.npz"}:
            raise ValueError("Selector checkpoint file inventory differs")
        training = meta["training"]
        if (
            training["config"] != selector_config()
            or (training["test_only"] and not allow_test_only)
            or (not training["test_only"] and training["optimizer_steps"] != 1000)
        ):
            raise ValueError("Selector checkpoint training scope differs")
        for name, expected in meta["files"].items():
            if Path(name).name != name or file_sha256(root / name) != expected:
                raise ValueError("Selector checkpoint bytes differ")
        result = cls(
            meta["dimension"], [tuple(c) for c in meta["classes"]], device=device
        )
        result.model.load_state_dict(
            torch.load(root / "weights.pt", map_location=device, weights_only=True),
            strict=True,
        )
        result.model.eval().requires_grad_(False)
        if not all(bool(torch.isfinite(p).all()) for p in result.model.parameters()):
            raise ValueError("Nonfinite selector checkpoint parameters")
        with np.load(root / "normalization.npz", allow_pickle=False) as values:
            result.mean, result.scale = values["mean"].copy(), values["scale"].copy()
        if (
            result.mean.shape != (meta["dimension"],)
            or result.scale.shape != result.mean.shape
            or not np.isfinite(result.mean).all()
            or not np.isfinite(result.scale).all()
            or np.any(result.scale < 0.05)
        ):
            raise ValueError("Invalid selector normalization")
        result.training = meta["training"]
        return result
