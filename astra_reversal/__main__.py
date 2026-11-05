"""Run with ``python -m astra_reversal --help``."""

import argparse
import json
import shlex
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .action_adapter import ActionAdapter, ActionSpec
from .agent import AstraAgent, CommandBackend, ReplayBackend
from .checkpoint import inspect_checkpoint
from .config import AGENT_FREE_METHODS, METHODS, BenchmarkConfig, RunConfig
from .controller import Controller
from .diagnostics import diagnose, require_diagnostics
from .evaluate import compare_runs, read_episodes, summarize, trace_costs
from .libero_runner import (
    LiberoEpisode,
    configure_libero,
    create_task_manifest,
    load_task_manifest,
    preflight,
    revision,
    shard_entries,
)
from .policy_adapter import load_policy
from .records import Recorder, RunManifest


def load_config(path):
    if path is None:
        return RunConfig()
    import yaml

    return RunConfig.from_dict(yaml.safe_load(Path(path).read_text()))


def enable_local_openpi():
    root = Path(__file__).resolve().parents[1] / "external" / "openpi"
    for source in (root / "src", root / "packages" / "openpi-client" / "src"):
        if source.is_dir():
            sys.path.insert(0, str(source))
    return root


def parser():
    root = argparse.ArgumentParser(
        description="Frozen pi0.5 Astra reversal experiments; see astra_reversal/README.md"
    )
    commands = root.add_subparsers(dest="command", required=True)
    config = commands.add_parser(
        "config", help="Print executable default configuration"
    )
    config.add_argument(
        "--suite",
        default="libero_10",
        choices=("libero_10", "libero_goal_ood", "libero_spatial_ood"),
    )
    config.add_argument("--method", default="reversal", choices=METHODS)
    inspect = commands.add_parser(
        "inspect-checkpoint",
        help="Inspect model layout and dimensions without loading weights",
    )
    inspect.add_argument("path")
    check = commands.add_parser(
        "preflight",
        help="Inspect local dependencies without loading weights or a simulator",
    )
    check.add_argument("--config")
    check.add_argument("--libero-root")
    manifest = commands.add_parser(
        "make-manifest",
        help="Freeze prescribed or seeded reset states before evaluation",
    )
    manifest.add_argument("--config")
    manifest.add_argument("--libero-root", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--tasks", type=int, nargs="+")
    manifest.add_argument("--trials", type=int)
    for name in ("run", "diagnostics"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--manifest", required=True)
        command.add_argument("--output", required=True)
        command.add_argument(
            "--libero-root", help="Override source location, preserving pinned revision"
        )
        if name == "run":
            command.add_argument("--shard-index", type=int, default=0)
            command.add_argument("--num-shards", type=int, default=1)
            backend = command.add_mutually_exclusive_group()
            backend.add_argument(
                "--agent-command",
                help="Astra client executable receiving JSON on stdin and returning response JSON",
            )
            backend.add_argument(
                "--agent-replay", help="Observation-bound JSONL replay records"
            )
            command.add_argument(
                "--progress-command",
                help="Optional image progress client returning {positive: bool, evidence: str}",
            )
            command.add_argument(
                "--diagnostics",
                help="Passing checkpoint diagnostics JSON; required for inversion methods",
            )
        else:
            command.add_argument("--action-atol", type=float, required=True)
            command.add_argument("--noise-atol", type=float, required=True)
            command.add_argument("--parity-atol", type=float, required=True)
            command.add_argument(
                "--resolutions", type=int, nargs="+", default=[10, 20, 50]
            )
            command.add_argument(
                "--controller-actions",
                help="Optional [H,7] .npy file from a recorded Astra proposal",
            )
    summary = commands.add_parser("summarize")
    summary.add_argument("run_directory", nargs="+")
    compare = commands.add_parser("compare")
    compare.add_argument("left")
    compare.add_argument("right")
    compare.add_argument("--samples", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=0)
    return root


def run(args, config):
    source = enable_local_openpi()
    task_manifest = load_task_manifest(args.manifest, config)
    entries = shard_entries(task_manifest, args.shard_index, args.num_shards)
    root = args.libero_root or task_manifest["libero_root"]
    directory = Path(args.output)
    recorder = Recorder(directory)
    _, create = configure_libero(root, config.benchmark, directory / "libero_config")
    policy = load_policy(
        config.policy.checkpoint,
        config.policy.config_name,
        config.policy.device,
        config.policy.checkpoint_provenance,
        config.policy.training_overlap,
        tokenizer_path=config.policy.tokenizer_path,
        input_profile=config.policy.input_profile,
        reference_assets=config.policy.reference_assets,
    )
    config.validate(policy.horizon)
    agent = None
    if config.method not in AGENT_FREE_METHODS:
        if args.agent_replay:
            backend = ReplayBackend(args.agent_replay)
        elif args.agent_command:
            backend = CommandBackend(
                shlex.split(args.agent_command),
                config.agent.model_version,
                config.agent.request_timeout_seconds,
            )
        else:
            raise ValueError("This method needs --agent-command or --agent-replay")
        agent = AstraAgent(backend, config.agent, recorder)
    assessor = None
    if args.progress_command:
        progress_backend = CommandBackend(
            shlex.split(args.progress_command),
            config.agent.model_version,
            config.agent.request_timeout_seconds,
        )

        def assessor(observation, criterion):
            return json.loads(
                progress_backend.generate(
                    {"observation": observation, "criterion": criterion}
                )
            )

    diagnostic_report = (
        json.loads(Path(args.diagnostics).read_text()) if args.diagnostics else None
    )
    inverse_methods = {
        "reversal",
        "same_condition",
        "inversion_only",
        "stage2",
        "language_only",
        "observation_only",
        "conditioning_transfer",
    }
    if config.method in inverse_methods and diagnostic_report is None:
        raise ValueError(
            "Run checkpoint diagnostics with declared tolerances before an inversion experiment"
        )
    summaries = []
    for index, entry in enumerate(entries):
        env, task, bddl = create(entry["task_id"], entry["seed"])
        episode = None
        try:
            import hashlib

            if (
                hashlib.sha256(bddl.read_bytes()).hexdigest() != entry["bddl_sha256"]
                or task.language != entry["instruction"]
            ):
                raise ValueError("Task assets differ from the frozen manifest")
            spec = ActionSpec.from_environment(env, policy.horizon, policy.action_dim)
            actions = ActionAdapter(
                spec, policy.input_transform, policy.output_transform
            )
            if diagnostic_report is not None:
                require_diagnostics(diagnostic_report, policy, actions, config)
            if index == 0:
                versions = {
                    "repository": revision(Path(__file__).resolve().parents[1]),
                    "openpi": revision(source),
                    "libero": revision(root),
                    "python": sys.version,
                    "evaluation_shard": {
                        "index": args.shard_index,
                        "count": args.num_shards,
                    },
                }
                recorder.manifest(
                    RunManifest(
                        asdict(config),
                        versions,
                        spec.as_dict(),
                        policy.metadata,
                        task_manifest["sha256"],
                        time.time(),
                    )
                )
                if diagnostic_report is not None:
                    recorder.event("accepted_diagnostics", report=diagnostic_report)
            video = (
                directory / f"episode_{index:05d}.mp4"
                if config.evaluation.save_rollout_videos
                else None
            )
            episode = LiberoEpisode(
                env,
                entry,
                config.benchmark,
                video_path=video,
                policy_image_size=policy.observation_image_size,
            )
            controller = Controller(config, policy, actions, recorder, agent, assessor)
            metadata = {
                key: entry[key]
                for key in (
                    "suite",
                    "task_id",
                    "initial_state_id",
                    "reset_state_sha256",
                )
            }
            metadata.update(
                seed=config.seed,
                protocol=config.benchmark.protocol,
                task_action_budget=config.benchmark.task_action_budget,
                split=config.evaluation.split,
            )
            episode_began = time.perf_counter()
            try:
                summary = controller.run_episode(
                    episode,
                    episode_id=entry["episode_id"],
                    instruction=entry["instruction"],
                    seed=config.seed,
                    metadata=metadata,
                )
            except Exception as exc:
                # Keep failed episodes in the denominator; never drop exceptions
                # and report a success rate only over the remaining runs.
                summary = {
                    "episode_id": entry["episode_id"],
                    "method": config.method,
                    "success": False,
                    "actions": controller.step,
                    "wall_seconds": time.perf_counter() - episode_began,
                    "velocity_evaluations": controller.velocity_evaluations,
                    "fallbacks": 0,
                    "failure": f"{type(exc).__name__}: {exc}",
                    **metadata,
                }
                recorder.event("episode_end", **summary)
            summaries.append(summary)
            print(
                json.dumps(
                    {
                        "episode": index + 1,
                        "total": len(entries),
                        "success": summary["success"],
                        "failure": summary["failure"],
                    }
                ),
                flush=True,
            )
        finally:
            if episode is not None:
                episode.close()
            else:
                env.close()
    output = {**summarize(summaries), "costs": trace_costs(directory)}
    (directory / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


def run_diagnostics(args, config):
    enable_local_openpi()
    task_manifest = load_task_manifest(args.manifest, config)
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    _, create = configure_libero(
        args.libero_root or task_manifest["libero_root"],
        config.benchmark,
        destination.parent / (destination.stem + "_libero_config"),
    )
    policy = load_policy(
        config.policy.checkpoint,
        config.policy.config_name,
        config.policy.device,
        config.policy.checkpoint_provenance,
        config.policy.training_overlap,
        tokenizer_path=config.policy.tokenizer_path,
        input_profile=config.policy.input_profile,
        reference_assets=config.policy.reference_assets,
    )
    entry = task_manifest["episodes"][0]
    env, _, _ = create(entry["task_id"], entry["seed"])
    episode = None
    try:
        spec = ActionSpec.from_environment(env, policy.horizon, policy.action_dim)
        actions = ActionAdapter(spec, policy.input_transform, policy.output_transform)
        episode = LiberoEpisode(
            env,
            entry,
            config.benchmark,
            policy_image_size=policy.observation_image_size,
        )
        report = diagnose(
            policy,
            actions,
            episode.observe(),
            entry["instruction"],
            seed=config.seed,
            resolutions=args.resolutions,
            solver=config.flow.integrator,
            solver_options=config.flow.solver_options,
            action_atol=args.action_atol,
            noise_atol=args.noise_atol,
            parity_atol=args.parity_atol,
            controller_actions=np.load(args.controller_actions, allow_pickle=False)
            if args.controller_actions
            else None,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        return report
    finally:
        if episode is not None:
            episode.close()
        else:
            env.close()


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "inspect-checkpoint":
        result = inspect_checkpoint(args.path)
    elif args.command == "config":
        config = RunConfig(
            method=args.method, benchmark=BenchmarkConfig.preset(args.suite)
        )
        if args.method == "direct_astra":
            config = replace(
                config,
                agent=replace(config.agent, refresh_env_steps=10),
                controller=replace(config.controller, latent_max_age_env_steps=10),
            )
        result = asdict(config)
    elif args.command == "summarize":
        result = {
            **summarize(
                [
                    episode
                    for directory in args.run_directory
                    for episode in read_episodes(directory)
                ]
            ),
            "costs_by_run": {
                directory: trace_costs(directory) for directory in args.run_directory
            },
        }
    elif args.command == "compare":
        result = compare_runs(
            args.left,
            args.right,
            samples=args.samples,
            seed=args.seed,
        )
    else:
        config = load_config(args.config)
        config.validate()
        if args.command == "preflight":
            result = preflight(config, args.libero_root)
        elif args.command == "make-manifest":
            enable_local_openpi()
            result = create_task_manifest(
                args.libero_root,
                config,
                args.output,
                task_ids=args.tasks,
                trials=args.trials,
            )
            result = {
                "path": args.output,
                "sha256": result["sha256"],
                "episodes": len(result["episodes"]),
            }
        elif args.command == "diagnostics":
            result = run_diagnostics(args, config)
        else:
            result = run(args, config)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
