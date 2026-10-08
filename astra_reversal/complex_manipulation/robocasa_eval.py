"""Native RoboCasa pi0.5 evaluation, with explicit reset/segment accounting.

Run in the pinned RoboCasa/OpenPI environment on an allocated GPU. The loader
uses the released model and inference transforms without importing the training
dataset stack. No LIBERO weights, action mapping, or horizon are reused.
"""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np

from astra_reversal.complex_manipulation.accounting import (
    EpisodeCounts,
    checked_chunk,
    summarize_episodes,
)
from astra_reversal.records import digest, file_sha256


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def policy_observation(obs):
    from openpi_client import image_tools

    state = np.concatenate(
        [
            obs["state.end_effector_position_relative"],
            obs["state.end_effector_rotation_relative"],
            obs["state.base_position"],
            obs["state.base_rotation"],
            obs["state.gripper_qpos"],
        ]
    )
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("RoboCasa native policy requires finite 16-dimensional state")
    value = {
        "observation/state": state,
        "prompt": obs["annotation.human.task_description"],
    }
    for target, source in (
        ("observation/image", "video.robot0_agentview_left"),
        ("observation/right_image", "video.robot0_agentview_right"),
        ("observation/wrist_image", "video.robot0_eye_in_hand"),
    ):
        # The upstream gym wrapper already flips the render vertically.
        image = np.ascontiguousarray(obs[source])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Expected an original uint8 RGB camera observation")
        value[target] = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(image, 224, 224)
        )
    return value


def load_native_policy(checkpoint, seed):
    import jax
    import jax.numpy as jnp
    from openpi import transforms
    from openpi.models import model, pi0_config, tokenizer
    from openpi.policies import policy, robocasa_policy
    from openpi.shared import normalize

    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("This experiment requires the allocated OSMO GPU")
    checkpoint = Path(checkpoint)
    config = pi0_config.Pi0Config(pi05=True, max_token_len=200)
    if (config.action_horizon, config.action_dim, config.discrete_state_input) != (
        50,
        32,
        True,
    ):
        raise ValueError("Pinned pi05_pretrain_human300 model contract changed")
    stats = normalize.load(checkpoint / "assets")
    if set(stats) != {"state", "actions"} or any(
        len(s.mean) != 32 for s in stats.values()
    ):
        raise ValueError(
            "Checkpoint normalization dimensions do not match the native model"
        )
    native = config.load(
        model.restore_params(checkpoint / "params", dtype=jnp.bfloat16)
    )
    loaded = policy.Policy(
        native,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            robocasa_policy.RobocasaInputs(config.action_dim, config.model_type),
            # This fork's DataConfig defaults to mean/std even for this pi0.5 run.
            transforms.Normalize(stats, use_quantiles=False),
            transforms.InjectDefaultPrompt(None),
            transforms.ResizeImages(224, 224),
            transforms.TokenizePrompt(
                tokenizer.PaligemmaTokenizer(200), discrete_state_input=True
            ),
            transforms.PadStatesAndActions(32),
        ],
        output_transforms=[
            transforms.Unnormalize(stats, use_quantiles=False),
            robocasa_policy.RobocasaOutputs(),
        ],
        sample_kwargs={"num_steps": 10},
    )
    # The pinned upstream constructor uses `rng or ...`, which tries to coerce
    # a typed JAX key to bool. Assign after construction to preserve native RNG.
    loaded._rng = jax.random.key(seed)
    return loaded


def rollout(
    env, policy, *, task, split, seed, horizon, output, smoke_actions=None, guide=None
):
    import imageio.v2 as imageio
    from robocasa.utils.env_utils import convert_action

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    method = "native_pi05" if guide is None else guide.method_name
    identity_method = "native" if guide is None else guide.identity_method
    identity = f"{os.environ.get('ASTRA_RUN_ID', output.parent.name)}:robocasa:{task}:{split}:seed{seed}:{identity_method}:attempt0"
    counts = EpisodeCounts(identity, horizon, 5)
    write_json(output / "started.json", {"episode_id": identity, "seed": seed})
    obs, _ = env.reset(seed=seed)
    raw_env = env.unwrapped.env
    initial = policy_observation(obs)
    initial_success = bool(raw_env._check_success())
    state = raw_env.sim.get_state().flatten()
    xml = raw_env.sim.model.get_xml().encode()
    np.savez_compressed(
        output / "initial_observation.npz", **initial, simulator_state=state
    )
    with gzip.open(output / "initial_model.xml.gz", "wb") as stream:
        stream.write(xml)
    write_json(
        output / "reset.json",
        {
            "seed": seed,
            "instruction": initial["prompt"],
            "observation_sha256": digest(initial),
            "state_sha256": digest(state),
            "model_sha256": hashlib.sha256(xml).hexdigest(),
            "initial_success": initial_success,
            "horizon": horizon,
            "execution_prefix": 5,
            "extra_libero_stabilization_steps": 0,
        },
    )
    imageio.imwrite(
        output / "starting_image.png",
        np.concatenate(
            [
                obs["video.robot0_agentview_left"],
                obs["video.robot0_agentview_right"],
                obs["video.robot0_eye_in_hand"],
            ],
            axis=1,
        ),
    )
    if initial_success:
        raise ValueError(
            "Initially successful reset; retain evidence without claiming success"
        )
    stop, success = "horizon", False
    total_policy_seconds = 0.0
    started = time.perf_counter()
    limit = horizon if smoke_actions is None else min(horizon, smoke_actions)
    with (
        imageio.get_writer(output / "rollout.mp4", fps=10) as video,
        (output / "executed_steps.jsonl").open("w") as steps,
        (output / "predictions.jsonl").open("w") as predictions,
    ):
        try:
            if guide is not None:
                guide.bind_reset(
                    json.loads((output / "reset.json").read_text()), initial, xml
                )
            while counts.executed_actions < limit:
                before = policy_observation(obs)
                model_input, guidance = (before, {"assisted": False})
                if guide is not None:
                    model_input, guidance = guide.prepare(
                        before, step=counts.executed_actions, episode_id=identity
                    )
                sample_started = time.perf_counter()
                inference_kwargs = (
                    {"intervention": guidance["intervention"]}
                    if "intervention" in guidance
                    else {}
                )
                prediction = policy.infer(model_input, **inference_kwargs)
                if prediction.get("interpolation"):
                    guidance["assisted"] = prediction["interpolation"]["has_effect"]
                actions = checked_chunk(
                    prediction["actions"], horizon=50, action_dim=12
                )
                elapsed = time.perf_counter() - sample_started
                total_policy_seconds += elapsed
                counts.generated_chunks += 1
                predictions.write(
                    json.dumps(
                        {
                            "step": counts.executed_actions,
                            "observation_sha256": digest(before),
                            "model_input_sha256": digest(model_input),
                            "model_prompt": model_input["prompt"],
                            "guidance": guidance,
                            "interpolation": prediction.get("interpolation"),
                            "actions": actions.tolist(),
                            "policy_seconds": elapsed,
                        }
                    )
                    + "\n"
                )
                predictions.flush()
                prefix_start = counts.executed_actions
                executed = 0
                try:
                    for action in actions[: min(5, limit - prefix_start)]:
                        obs, _, terminated, truncated, info = env.step(
                            convert_action(action)
                        )
                        executed += 1
                        success = bool(info["success"])
                        steps.write(
                            json.dumps(
                                {
                                    "step": prefix_start + executed - 1,
                                    "command": action.tolist(),
                                    "success": success,
                                    "terminated": bool(terminated),
                                    "truncated": bool(truncated),
                                }
                            )
                            + "\n"
                        )
                        if (prefix_start + executed) % 2 == 0 or success:
                            video.append_data(
                                np.concatenate(
                                    [
                                        obs["video.robot0_agentview_left"],
                                        obs["video.robot0_agentview_right"],
                                        obs["video.robot0_eye_in_hand"],
                                    ],
                                    axis=1,
                                )
                            )
                        if success or terminated or truncated:
                            stop = "success" if success else "environment_terminated"
                            break
                finally:
                    if executed:
                        counts.execute(executed, assisted=guidance["assisted"])
                steps.flush()
                write_json(
                    output / "progress.json",
                    {
                        "episode_id": identity,
                        "executed_actions": counts.executed_actions,
                        "reset_free_segments": counts.reset_free_segments,
                        "policy_seconds": total_policy_seconds,
                    },
                )
                if success or stop == "environment_terminated":
                    break
            if not success and stop == "horizon" and limit < horizon:
                stop = "smoke_limit"
        except BaseException as exc:
            stop = (
                "interrupted"
                if isinstance(exc, KeyboardInterrupt)
                else "infrastructure_or_policy_error"
            )
            write_json(
                output / "error.json",
                {"type": type(exc).__name__, "message": str(exc)[:1000]},
            )
            raise
        finally:
            result = counts.result(success=success, stop_reason=stop)
            result.update(
                task=task,
                split=split,
                seed=seed,
                method=method,
                wall_seconds=time.perf_counter() - started,
                policy_seconds=total_policy_seconds,
                teacher_calls=0,
                teacher_tokens=0,
            )
            if guide is not None:
                result.update(guide.summary())
            write_json(output / "result.json", result)
    return result


def episode_schedule(tasks, seeds, plan=None):
    pairs = (
        json.loads(Path(plan).read_text())
        if plan
        else [[t, s] for t in tasks for s in seeds]
    )
    if (
        not isinstance(pairs, list)
        or not pairs
        or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or pair[0] not in tasks
            or type(pair[1]) is not int
            or pair[1] not in seeds
            for pair in pairs
        )
    ):
        raise ValueError("Episode plan must contain declared task/seed pairs")
    if len({tuple(p) for p in pairs}) != len(pairs):
        raise ValueError("Episode plan must not repeat reset pairs")
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tasks", nargs="+", default=["LoadPreparedFood", "PackIdenticalLunches"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1000])
    parser.add_argument("--split", choices=["pretrain", "target"], default="pretrain")
    parser.add_argument("--smoke-actions", type=int)
    parser.add_argument("--guidance-baselines", type=Path)
    parser.add_argument("--representation-method", choices=("tei", "tli"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--episode-plan", type=Path)
    args = parser.parse_args()
    if args.smoke_actions is not None and args.smoke_actions < 1:
        parser.error("Smoke probes must execute at least one action")
    if args.guidance_baselines and args.smoke_actions:
        parser.error("Guidance comparison uses full native horizons")
    if args.representation_method and not args.guidance_baselines:
        parser.error("Representation comparison requires paired native evidence")
    if args.preflight_only and not args.representation_method:
        parser.error("Preflight-only requires a representation method")
    episode_pairs = episode_schedule(args.tasks, args.seeds, args.episode_plan)
    args.output.mkdir(parents=True, exist_ok=False)
    prior_failures = {}
    if args.guidance_baselines and not args.preflight_only:
        from astra_reversal.complex_manipulation.language_guidance import prior_failure

        # imageio seeks by starting ffmpeg subprocesses. Finish this work before
        # JAX initializes its multithreaded runtime; forking afterward can leave
        # the decoder with invalid inherited gRPC descriptors.
        preparation = []
        for task, seed in episode_pairs:
            baseline = prior_failure(args.guidance_baselines / f"{task}_seed{seed}")
            prior_failures[task, seed] = baseline
            preparation.append(
                {
                    "task": task,
                    "seed": seed,
                    "video_sha256": baseline["video_sha256"],
                    "frame_steps": [f["step"] for f in baseline["frames"]],
                }
            )
            write_json(args.output / "prior_failure_preparation.json", preparation)
    import gymnasium as gym
    import jax
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    versions = {
        name: importlib.metadata.version(name)
        for name in ("jax", "flax", "numpy", "mujoco", "robosuite")
    }
    write_json(
        args.output / "runtime.json",
        {
            "versions": versions,
            "devices": [str(d) for d in jax.devices()],
            "checkpoint_norm_sha256": file_sha256(
                args.checkpoint / "assets/norm_stats.json"
            ),
            "tasks": args.tasks,
            "seeds": args.seeds,
            "episode_schedule": episode_pairs,
            "split": args.split,
            "model_horizon": 50,
            "internal_action_dim": 32,
            "output_action_dim": 12,
            "normalization": "checkpoint_mean_std",
            "sample_steps": 10,
            "execution_prefix": 5,
            "scope": (
                "checkpoint_interpolation_preflight"
                if args.preflight_only
                else "development_guidance_comparison"
                if args.guidance_baselines
                else "smoke_only"
                if args.smoke_actions
                else "development_baseline"
            ),
            "guidance": (
                {
                    "model": "gpt-6-astra",
                    "reasoning_effort": "medium",
                    "backend": "codex_relay",
                    "maximum_calls_per_episode": 16,
                    "method": args.representation_method or "language_phase_prompt",
                    "policy_updates": 0,
                }
                if args.guidance_baselines
                else None
            ),
            "constructor_resets": (
                "Zero: preflight uses saved observations without constructing an environment."
                if args.preflight_only
                else "The upstream gym wrapper performs one setup reset per environment construction; these are not policy rollouts."
            ),
        },
    )
    if args.guidance_baselines:
        from astra_reversal.codex_relay import ensure_server

        ensure_server()
    policy = load_native_policy(args.checkpoint, args.seeds[0])
    if args.representation_method:
        from astra_reversal.complex_manipulation.jax_text_interpolation import (
            TextInterpolationPolicy,
        )

        policy = TextInterpolationPolicy(policy)
        # Inference-only probe on a saved native reset: no simulator reset/actions.
        task, seed = episode_pairs[0]
        with np.load(
            args.guidance_baselines / f"{task}_seed{seed}" / "initial_observation.npz"
        ) as data:
            probe = {key: data[key] for key in data.files if key != "simulator_state"}
        probe["prompt"] = str(probe["prompt"])
        policy.preflight(
            probe,
            active_method=None if args.preflight_only else args.representation_method,
            publish=lambda value: write_json(
                args.output / "interpolation_preflight.json", value
            ),
        )
        if args.preflight_only:
            return
    results = []
    constructors = []
    write_json(args.output / "constructors.json", constructors)
    for task, seed in episode_pairs:
        # Recreate the env for each seed so simulator reset RNG history is explicit.
        np.random.seed(seed)
        constructor = {
            "task": task,
            "seed": seed,
            "status": "started",
            "setup_resets": None,
        }
        constructors.append(constructor)
        write_json(args.output / "constructors.json", constructors)
        env = gym.make(
            f"robocasa/{task}",
            split=args.split,
            seed=seed,
            disable_env_checker=True,
        )
        constructor.update(status="ready", setup_resets=1)
        write_json(args.output / "constructors.json", constructors)
        try:
            policy._rng = jax.random.key(seed)
            guide = None
            if args.guidance_baselines:
                from astra_reversal.codex_relay import CodexRelayClient
                from astra_reversal.complex_manipulation.language_guidance import (
                    LanguageGuide,
                )

                directory = args.output / f"{task}_seed{seed}"
                client = CodexRelayClient(
                    model="gpt-6-astra",
                    family=(
                        "complex_representation"
                        if args.representation_method
                        else "complex_language"
                    ),
                    response_log=str(directory / "guidance/provider.jsonl"),
                    timeout=300,
                )
                guide_kwargs = {}
                if args.representation_method:
                    from astra_reversal.complex_manipulation import (
                        representation_teacher,
                    )

                    guide_kwargs = {
                        "teacher_module": representation_teacher,
                        "representation_method": args.representation_method,
                    }
                guide = LanguageGuide(
                    client=client,
                    baseline=prior_failures[task, seed],
                    output=directory / "guidance",
                    **guide_kwargs,
                )
            results.append(
                rollout(
                    env,
                    policy,
                    task=task,
                    split=args.split,
                    seed=seed,
                    horizon=get_task_horizon(task),
                    output=args.output / f"{task}_seed{seed}",
                    smoke_actions=args.smoke_actions,
                    guide=guide,
                )
            )
            write_json(args.output / "summary.json", summarize_episodes(results))
        finally:
            env.close()


if __name__ == "__main__":
    main()
