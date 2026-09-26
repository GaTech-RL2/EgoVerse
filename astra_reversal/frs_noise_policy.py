"""Task-specific auxiliary noise learning; the pi05 policy is never an argument.

This is a DSBC-inspired approximation, not the paper's Gaussian-NLL policy:
raw observed-noise MSE + 0.001 mean-square regularization, a deterministic mean
bounded by tanh*5, and no learned variance. Each Astra-``better`` rollout adds
its executed, repeated first-seven noise labels to replay and triggers exactly
1,000 Adam updates. Native deferrals are labels too, provided their executed
noise actually has the same repeated representation. No simulator label enters
selection. At most three accepted updates are allowed per task.

Actual training requires CUDA and CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8)
before CUDA initialization. A visibly test-only CPU override exists solely for
small synthetic regression tests and produces incompatible test checkpoints.
"""

import copy
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import PIL
import torch
from PIL import Image
from torch import nn

from .frs_operators import ACTION_DIM, HORIZON, PHYSICAL_DIM
from .records import digest, file_sha256, to_numpy

CAMERAS = ("observation/image", "observation/wrist_image")
IMAGE_SIZE = 84
NOISE_BOUND = 5.0
UPDATES_PER_ROUND = 1000
MAX_ROUNDS = 3
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
MEAN_REGULARIZATION = 1e-3


def training_config():
    return {
        "architecture": "shared_CNN16_32_32_pair84_proprio8_MLP128_128_128",
        "cnn_kernels_strides": [[5, 2], [3, 2], [3, 2]],
        "cnn_pool": "fixed_AvgPool2d_kernel2_stride2_9x9_to4x4",
        "camera_order": list(CAMERAS),
        "image_preprocessing": "Pillow_bilinear_resize84_no_crop_or_flip_then_float32_minus1_to1",
        "proprio_preprocessing": "raw_float32_8vector_no_fitted_statistics",
        "mean_bound": NOISE_BOUND,
        "horizon": HORIZON,
        "action_dim": ACTION_DIM,
        "loss": "raw_observed_noise_MSE_plus_mean_square_regularization",
        "mean_regularization": MEAN_REGULARIZATION,
        "loss_targets_clipped": False,
        "learned_variance": False,
        "paper_relation": "DSBC_inspired_MSE_approximation_not_Gaussian_NLL_replication",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "updates_per_accepted_rollout": UPDATES_PER_ROUND,
        "maximum_accepted_updates": MAX_ROUNDS,
        "replay": "all_Astra_better_same_task_rollouts",
        "sample_kinds": ["frs_edit", "native_defer"],
        "selection": "Astra_judge_verdict_better_only_no_simulator_label",
    }


def _hex(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _repeated_noise(value):
    noise = np.array(to_numpy(value), copy=True)
    if (
        noise.shape != (1, HORIZON, ACTION_DIM)
        or noise.dtype != np.float32
        or not np.isfinite(noise).all()
    ):
        raise ValueError("Executed noise must be finite float32 [1, 10, 32]")
    if not np.array_equal(
        noise[0, :, :7], np.broadcast_to(noise[0, 0, :7], (HORIZON, 7))
    ):
        raise ValueError(
            "Executed first7 noise must be identical across all horizon rows"
        )
    return noise


def preprocess_observation(observation):
    """Return copied paired RGB uint8 [2,84,84,3] and raw float32 proprio [8]."""
    images = []
    for camera in CAMERAS:
        frame = np.asarray(observation[camera])
        if (
            frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or min(frame.shape[:2]) < 1
        ):
            raise ValueError(f"{camera} must be nonempty uint8 HWC RGB")
        images.append(
            np.asarray(
                Image.fromarray(frame).resize(
                    (IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR
                ),
                dtype=np.uint8,
            ).copy()
        )
    raw = np.asarray(observation["observation/state"])
    if raw.shape != (8,) or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError("Proprioception must be a finite numeric 8-vector")
    state = raw.astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError("Proprioception overflows float32")
    return np.stack(images), state


def make_training_sample(
    observation, executed_noise, *, kind, observation_step, source_id
):
    """Capture an executed loop chunk. This does not approve or train on it."""
    if kind not in ("frs_edit", "native_defer"):
        raise ValueError("Training kind must be frs_edit or native_defer")
    if type(observation_step) is not int or observation_step < 0:
        raise ValueError("observation_step must be a nonnegative integer")
    _text(source_id, "source_id")
    noise = _repeated_noise(executed_noise)
    images, state = preprocess_observation(observation)
    # Repeated values survive float64 mean -> float32 without any label mismatch.
    target = noise[0, :, :7].astype(np.float64).mean(axis=0).astype(np.float32)
    metadata = {
        "kind": kind,
        "observation_step": observation_step,
        "source_id": source_id,
        "observation_sha256": digest(observation),
        "original_camera_sha256": {
            camera: digest(np.asarray(observation[camera])) for camera in CAMERAS
        },
        "images84_sha256": digest(images),
        "state_sha256": digest(state),
        "executed_noise_sha256": digest(noise),
        "target_sha256": digest(target),
        "label_matches_executed_first7_exactly": True,
        "pillow_version": PIL.__version__,
    }
    metadata["sample_id"] = digest(metadata)
    return {
        "images": images,
        "state": state,
        "executed_noise": noise,
        "target": target,
        "metadata": metadata,
    }


def _validate_sample(value):
    if not isinstance(value, dict) or set(value) != {
        "images",
        "state",
        "executed_noise",
        "target",
        "metadata",
    }:
        raise ValueError("Use make_training_sample to construct training records")
    sample = copy.deepcopy(value)
    metadata = sample["metadata"]
    if not isinstance(metadata, dict) or metadata.get("kind") not in (
        "frs_edit",
        "native_defer",
    ):
        raise ValueError("Invalid training sample kind")
    if (
        type(metadata.get("observation_step")) is not int
        or metadata["observation_step"] < 0
    ):
        raise ValueError("Invalid training observation step")
    _text(metadata.get("source_id"), "source_id")
    expected_id = digest(
        {key: value for key, value in metadata.items() if key != "sample_id"}
    )
    if metadata.get("sample_id") != expected_id:
        raise ValueError("Training sample metadata hash mismatch")
    noise = _repeated_noise(sample["executed_noise"])
    expected = {
        "images": ((2, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8, "images84_sha256"),
        "state": ((8,), np.float32, "state_sha256"),
        "executed_noise": (
            (1, HORIZON, ACTION_DIM),
            np.float32,
            "executed_noise_sha256",
        ),
        "target": ((7,), np.float32, "target_sha256"),
    }
    for key, (shape, dtype, hash_key) in expected.items():
        array = sample[key]
        if (
            not isinstance(array, np.ndarray)
            or array.shape != shape
            or array.dtype != dtype
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"Invalid training array {key}")
        if metadata.get(hash_key) != digest(array):
            raise ValueError(f"Training array hash mismatch: {key}")
    if metadata.get(
        "label_matches_executed_first7_exactly"
    ) is not True or not np.array_equal(sample["target"], noise[0, 0, :7]):
        raise ValueError(
            "Training target differs from the actual executed repeated noise"
        )
    _hex(metadata.get("observation_sha256"), "observation_sha256")
    if set(metadata.get("original_camera_sha256", {})) != set(CAMERAS):
        raise ValueError("Missing original camera hashes")
    for camera_hash in metadata["original_camera_sha256"].values():
        _hex(camera_hash, "original camera hash")
    return sample


class VisualNoiseMean(nn.Module):
    """Small shared CNN, paired-view concatenation, and three 128-wide layers."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            # Fixed 84px input gives 9x9 here. AdaptiveAvgPool2d backward
            # is prohibited by PyTorch's deterministic CUDA mode.
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * 32 * 4 * 4 + 8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, PHYSICAL_DIM),
            nn.Tanh(),
        )

    def forward(self, images, state):
        if (
            images.ndim != 5
            or images.shape[1:] != (2, 3, IMAGE_SIZE, IMAGE_SIZE)
            or state.shape != (images.shape[0], 8)
        ):
            raise ValueError("CNN expects paired [B,2,3,84,84] images and [B,8] state")
        features = self.encoder(images.reshape(-1, 3, IMAGE_SIZE, IMAGE_SIZE)).reshape(
            images.shape[0], -1
        )
        return NOISE_BOUND * self.head(torch.cat((features, state), dim=-1))


def noise_loss(predicted, target):
    """Raw-label loss; bounded output does not silently clip observed labels."""
    if (
        predicted.ndim != 2
        or predicted.shape != target.shape
        or predicted.shape[1] != PHYSICAL_DIM
    ):
        raise ValueError("Noise loss needs matching [B,7] arrays")
    mse = (predicted - target).square().mean()
    mean_square = predicted.square().mean()
    return mse + MEAN_REGULARIZATION * mean_square, mse, mean_square


def _tree(value, *, tensors=False):
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu()
        return array.clone() if tensors else array.numpy().copy()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy()) if tensors else value.copy()
    if isinstance(value, dict):
        return {key: _tree(item, tensors=tensors) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree(item, tensors=tensors) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree(item, tensors=tensors) for item in value)
    return copy.deepcopy(value)


@contextmanager
def _deterministic_training(device):
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (
        ":4096:8",
        ":16:8",
    ):
        raise RuntimeError(
            "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization"
        )
    old = {
        "algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.use_deterministic_algorithms(
            old["algorithms"], warn_only=old["warn_only"]
        )
        torch.backends.cudnn.benchmark = old["benchmark"]
        torch.backends.cudnn.deterministic = old["cudnn_deterministic"]
        torch.backends.cudnn.allow_tf32 = old["cudnn_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = old["matmul_tf32"]


class AuxiliaryNoisePolicy:
    """Isolated auxiliary parameters, approved replay, and transactional state."""

    def __init__(self, task_id, seed, device="cuda", *, _test_only_cpu_steps=None):
        self.task_id = _text(task_id, "task_id")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0,2**63)")
        self.seed = seed
        self.device = torch.device(device)
        if _test_only_cpu_steps is not None and (
            self.device.type != "cpu"
            or type(_test_only_cpu_steps) is not int
            or not 1 <= _test_only_cpu_steps <= UPDATES_PER_ROUND
        ):
            raise ValueError(
                "Test-only CPU steps must be an integer in [1,1000] on CPU"
            )
        self._test_only_cpu_steps = _test_only_cpu_steps
        self.effective_seed = int(digest({"task": task_id, "seed": seed})[:16], 16) % (
            2**63
        )
        # Construct on CPU in a forked RNG scope, leaving global/model RNG alone.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.effective_seed)
            self.model = VisualNoiseMean().to(self.device, dtype=torch.float32).eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.replay = []
        self.history = []
        self.accepted_rollouts = []

    @property
    def trained(self):
        return bool(self.history)

    @property
    def rounds(self):
        return len(self.history)

    def _sources(self):
        return {
            "noise_policy_sha256": file_sha256(Path(__file__)),
            "operators_sha256": file_sha256(
                Path(__file__).with_name("frs_operators.py")
            ),
            "records_sha256": file_sha256(Path(__file__).with_name("records.py")),
        }

    def metadata(self):
        values = {
            "schema_version": "auxiliary-frs-noise-policy-1.0",
            "task_id": self.task_id,
            "seed": self.seed,
            "effective_seed": self.effective_seed,
            "config": training_config(),
            "sources": self._sources(),
            "preprocessing_pillow_version": PIL.__version__,
            "trained": self.trained,
            "rounds": self.rounds,
            "accepted_rollouts": list(self.accepted_rollouts),
            "replay_samples": len(self.replay),
            "replay_sha256": digest(self.replay),
            "history_sha256": digest(self.history),
            "parameter_sha256": digest(_tree(self.model.state_dict())),
            "optimizer_sha256": digest(_tree(self.optimizer.state_dict())),
            "synthetic_cpu_test_only": self._test_only_cpu_steps is not None,
            "test_only_cpu_steps": self._test_only_cpu_steps,
        }
        values["state_id"] = digest(values)
        return values

    def _inputs(self, images, states):
        pixels = (
            torch.as_tensor(images, device=self.device).permute(0, 1, 4, 2, 3).float()
        )
        return pixels.div(127.5).sub(1.0), torch.as_tensor(
            states, device=self.device, dtype=torch.float32
        )

    def _predictions(self, samples):
        result = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(samples), BATCH_SIZE):
                batch = samples[start : start + BATCH_SIZE]
                images, states = self._inputs(
                    np.stack([s["images"] for s in batch]),
                    np.stack([s["state"] for s in batch]),
                )
                result.append(self.model(images, states).cpu().numpy().copy())
        return np.concatenate(result)

    def predict(self, observation, rng, base_noise):
        """Before fit: exact base fallback, without preprocessing or RNG draws."""
        base = _repeated_noise(base_noise)
        if not isinstance(rng, np.random.Generator):
            raise ValueError("Expected a NumPy Generator")
        identity = self.metadata()
        before = digest(rng.bit_generator.state)
        if not self.trained:
            return {
                "noise": base,
                "mean_first7": base[0, 0, :7].copy(),
                "receipt": {
                    "source": "exact_untrained_base_fallback",
                    "state_id": identity["state_id"],
                    "trained_rounds": 0,
                    "base_noise_sha256": digest(base),
                    "generation_noise_sha256": digest(base),
                    "rng_state_before_sha256": before,
                    "rng_state_after_sha256": before,
                    "auxiliary_forward_performed": False,
                },
            }
        images, state = preprocess_observation(observation)
        self.model.eval()
        with torch.no_grad(), _deterministic_training(self.device):
            mean = (
                self.model(*self._inputs(images[None], state[None]))[0]
                .cpu()
                .numpy()
                .copy()
            )
        if (
            mean.dtype != np.float32
            or not np.isfinite(mean).all()
            or np.any(np.abs(mean) > NOISE_BOUND)
        ):
            raise FloatingPointError("Invalid bounded auxiliary noise prediction")
        noise = np.empty((1, HORIZON, ACTION_DIM), np.float32)
        noise[..., :7] = mean
        noise[..., 7:] = rng.standard_normal(
            (1, HORIZON, ACTION_DIM - PHYSICAL_DIM)
        ).astype(np.float32)
        return {
            "noise": noise,
            "mean_first7": mean,
            "receipt": {
                "source": "learned_deterministic_mean_fresh_gaussian_padding",
                "state_id": identity["state_id"],
                "trained_rounds": self.rounds,
                "observation_sha256": digest(observation),
                "images84_sha256": digest(images),
                "state_sha256": digest(state),
                "base_noise_sha256": digest(base),
                "mean_first7_sha256": digest(mean),
                "generation_noise_sha256": digest(noise),
                "rng_state_before_sha256": before,
                "rng_state_after_sha256": digest(rng.bit_generator.state),
                "auxiliary_forward_performed": True,
                "learned_variance": False,
            },
        }

    def fit_accepted_rollout(self, rollout_id, samples, *, judge_verdict, judge_sha256):
        """Add one Astra-approved rollout and perform a fixed update, or no-op.

        The caller supplies executed sample bindings and the recorded judge
        response's content digest. This module does not authenticate that judge; offline
        request/provider/event audits must verify that binding independently.
        """
        _text(rollout_id, "rollout_id")
        _text(judge_verdict, "judge_verdict")
        _hex(judge_sha256, "judge_sha256")
        before = self.metadata()
        if judge_verdict != "better":
            return {
                "status": "not_updated",
                "reason": "judge_verdict_not_better",
                "judge_verdict": judge_verdict,
                "judge_sha256": judge_sha256,
                "rollout_id": rollout_id,
                "state_before": before["state_id"],
                "state_after": before["state_id"],
                "optimizer_updates": 0,
            }
        if self.rounds >= MAX_ROUNDS:
            raise ValueError("At most three accepted update rounds are allowed")
        if rollout_id in self.accepted_rollouts:
            raise ValueError("Rollout was already included in approved replay")
        incoming = [_validate_sample(sample) for sample in samples]
        if not incoming:
            return {
                "status": "not_updated",
                "reason": "no_executed_samples",
                "judge_verdict": judge_verdict,
                "judge_sha256": judge_sha256,
                "rollout_id": rollout_id,
                "state_before": before["state_id"],
                "state_after": before["state_id"],
                "optimizer_updates": 0,
            }
        ids = [sample["metadata"]["sample_id"] for sample in self.replay + incoming]
        sources = [sample["metadata"]["source_id"] for sample in self.replay + incoming]
        steps = [sample["metadata"]["observation_step"] for sample in incoming]
        if (
            len(ids) != len(set(ids))
            or len(sources) != len(set(sources))
            or len(steps) != len(set(steps))
        ):
            raise ValueError("Duplicate executed sample/source/step in approved replay")
        if steps != sorted(steps):
            raise ValueError("Executed samples must follow observation-step order")
        if self.device.type != "cuda" and self._test_only_cpu_steps is None:
            raise RuntimeError("Real auxiliary-policy training requires CUDA")
        replay = self.replay + incoming
        targets = np.stack([sample["target"] for sample in replay])
        iterations = self._test_only_cpu_steps or UPDATES_PER_ROUND
        round_seed = int(
            digest({"task_seed": self.effective_seed, "round": self.rounds + 1})[:16],
            16,
        ) % (2**63)
        generator = torch.Generator(device="cpu").manual_seed(round_seed)
        previous_model = _tree(self.model.state_dict(), tensors=True)
        previous_optimizer = _tree(self.optimizer.state_dict(), tensors=True)
        began = time.perf_counter()
        losses = []
        try:
            with torch.enable_grad(), _deterministic_training(self.device):
                predictions_before = self._predictions(replay)
                # Store compact uint8 frames on device; normalize just each batch.
                pixels = torch.as_tensor(
                    np.stack([sample["images"] for sample in replay]),
                    device=self.device,
                )
                states = torch.as_tensor(
                    np.stack([sample["state"] for sample in replay]), device=self.device
                )
                labels = torch.as_tensor(targets, device=self.device)
                self.model.train()
                for _ in range(iterations):
                    indices = torch.randint(
                        len(replay), (BATCH_SIZE,), generator=generator
                    ).to(self.device)
                    images = (
                        pixels[indices]
                        .permute(0, 1, 4, 2, 3)
                        .float()
                        .div(127.5)
                        .sub(1.0)
                    )
                    predicted = self.model(images, states[indices])
                    loss, mse, regularization = noise_loss(predicted, labels[indices])
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Non-finite auxiliary training loss")
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if any(
                        p.grad is not None and not bool(torch.isfinite(p.grad).all())
                        for p in self.model.parameters()
                    ):
                        raise FloatingPointError("Non-finite auxiliary gradient")
                    self.optimizer.step()
                    losses.append(
                        [
                            float(loss.detach().cpu()),
                            float(mse.detach().cpu()),
                            float(regularization.detach().cpu()),
                        ]
                    )
                predictions_after = self._predictions(replay)
                if not np.isfinite(predictions_after).all():
                    raise FloatingPointError("Non-finite post-update predictions")
        except BaseException:
            self.model.load_state_dict(previous_model, strict=True)
            self.optimizer.load_state_dict(previous_optimizer)
            self.optimizer.zero_grad(set_to_none=True)
            self.model.eval()
            raise
        elapsed = time.perf_counter() - began
        delta = predictions_after.astype(np.float64) - targets
        outside = np.abs(targets) > NOISE_BOUND
        receipt = {
            "status": "updated",
            "round": self.rounds + 1,
            "rollout_id": rollout_id,
            "judge_verdict": judge_verdict,
            "judge_sha256": judge_sha256,
            "state_before": before["state_id"],
            "task_id": self.task_id,
            "added_samples": len(incoming),
            "replay_samples": len(replay),
            "sample_ids": ids,
            "sample_kinds": {
                kind: sum(s["metadata"]["kind"] == kind for s in replay)
                for kind in ("frs_edit", "native_defer")
            },
            "optimizer_updates": iterations,
            "training_seed": round_seed,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "mean_regularization": MEAN_REGULARIZATION,
            "loss_targets_clipped": False,
            "targets_raw": targets,
            "targets_clipped_for_diagnostics_only": np.clip(
                targets, -NOISE_BOUND, NOISE_BOUND
            ),
            "target_out_of_range_mask": outside,
            "target_out_of_range_fraction": float(outside.mean()),
            "target_out_of_range_fraction_per_sample": outside.mean(axis=1),
            "predictions_before": predictions_before,
            "predictions_after": predictions_after,
            "mean_target_error": delta,
            "mean_target_rmse_per_sample": np.sqrt(np.mean(delta**2, axis=1)),
            "mean_target_rmse": float(np.sqrt(np.mean(delta**2))),
            "mean_target_max_abs": float(np.max(np.abs(delta))),
            "loss_trace_columns": ["total", "raw_target_mse", "unweighted_mean_square"],
            "loss_trace": np.asarray(losses, np.float64),
            "wall_seconds": elapsed,
            "device": str(self.device),
            "torch_version": str(torch.__version__),
            "pillow_version": PIL.__version__,
            "cuda_version": torch.version.cuda,
            "deterministic_algorithms": True,
            "tf32_training": False,
            "synthetic_cpu_test_only": self._test_only_cpu_steps is not None,
        }
        self.replay = replay
        self.accepted_rollouts.append(rollout_id)
        # Avoid a circular state digest: history omits the returned state_after.
        self.history.append(copy.deepcopy(receipt))
        receipt["state_after"] = self.metadata()["state_id"]
        return receipt

    def save(self, directory):
        """Write a new self-contained weights/Adam/replay checkpoint and manifest."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "state.pt"
        identity = self.metadata()
        payload = {
            "model": _tree(self.model.state_dict(), tensors=True),
            "optimizer": _tree(self.optimizer.state_dict(), tensors=True),
            "replay": _tree(self.replay, tensors=True),
            "history": _tree(self.history, tensors=True),
            "accepted_rollouts": list(self.accepted_rollouts),
        }
        torch.save(payload, path)
        manifest = {
            "schema_version": "frs-noise-checkpoint-1.0",
            "policy": identity,
            "files": {
                "state.pt": {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            },
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        return {**manifest, "manifest_sha256": file_sha256(directory / "manifest.json")}

    def load(self, directory):
        """Validate all state in a candidate object, then atomically restore it."""
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("schema_version") != "frs-noise-checkpoint-1.0" or set(
            manifest.get("files", {})
        ) != {"state.pt"}:
            raise ValueError("Invalid auxiliary checkpoint schema")
        expected = manifest["policy"]
        current = self.metadata()
        for key in (
            "task_id",
            "seed",
            "effective_seed",
            "config",
            "sources",
            "preprocessing_pillow_version",
            "synthetic_cpu_test_only",
            "test_only_cpu_steps",
        ):
            if expected.get(key) != current[key]:
                raise ValueError(f"Checkpoint compatibility mismatch: {key}")
        path = directory / "state.pt"
        if manifest["files"]["state.pt"] != {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }:
            raise ValueError("Checkpoint state file hash/size mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if set(payload) != {
            "model",
            "optimizer",
            "replay",
            "history",
            "accepted_rollouts",
        }:
            raise ValueError("Invalid checkpoint state keys")
        candidate = type(self)(
            self.task_id,
            self.seed,
            self.device,
            _test_only_cpu_steps=self._test_only_cpu_steps,
        )
        candidate.model.load_state_dict(payload["model"], strict=True)
        candidate.optimizer.load_state_dict(payload["optimizer"])
        candidate.replay = [_validate_sample(s) for s in _tree(payload["replay"])]
        candidate.history = _tree(payload["history"])
        candidate.accepted_rollouts = list(payload["accepted_rollouts"])
        if (
            len(candidate.history) != len(candidate.accepted_rollouts)
            or len(candidate.history) > MAX_ROUNDS
            or len(set(candidate.accepted_rollouts)) != len(candidate.accepted_rollouts)
        ):
            raise ValueError("Invalid checkpoint update history")
        if bool(candidate.history) != bool(candidate.replay):
            raise ValueError("Checkpoint replay and training history disagree")
        if not all(bool(torch.isfinite(p).all()) for p in candidate.model.parameters()):
            raise ValueError("Non-finite checkpoint parameter")
        if candidate.metadata() != expected:
            raise ValueError("Checkpoint policy/state identity mismatch")
        self.model, self.optimizer = candidate.model, candidate.optimizer
        self.replay, self.history, self.accepted_rollouts = (
            candidate.replay,
            candidate.history,
            candidate.accepted_rollouts,
        )
        return {
            "status": "restored",
            "state_before": current["state_id"],
            "state_after": expected["state_id"],
            "rounds": self.rounds,
            "manifest_sha256": file_sha256(directory / "manifest.json"),
        }
