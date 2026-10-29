"""Yam station rollout: EgoVerse checkpoint inference or read-only recorded action replay."""

import argparse
import copy
import json
import time
from collections import deque
from collections.abc import Mapping

import numpy as np
from omegaconf import OmegaConf

from egomimic.robot.yam.rollout_ik import validate_ik_recovery
from egomimic.robot.yam_station.cameras import CameraView
from egomimic.robot.yam_station.ik_options import (
    validate_ik_relaxation,
    validate_ik_solver,
)
from egomimic.robot.yam_station.interface import ARM_OFFSET, create_robot, joint_vector
from egomimic.robot.yam_station.rollout_episode import (
    RolloutEpisodeRecorder,
    validate_episode_recording,
)
from egomimic.robot.yam_station.rollout_video import (
    RolloutVideoRecorder,
    validate_video_recording,
)


def load_policy(config):
    config = dict(config)
    kind = config.pop("kind")
    if kind == "egoverse":
        from egomimic.robot.yam_station.egoverse_policy import load_egoverse_policy

        return load_egoverse_policy(config)
    if kind == "zarr_replay":
        from egomimic.robot.yam_station.replay_policy import ZarrReplayPolicy

        return ZarrReplayPolicy(**config)
    if kind == "hdf5_replay":
        from egomimic.robot.yam_station.replay_policy import Hdf5ReplayPolicy

        return Hdf5ReplayPolicy(**config)
    raise ValueError(
        "Use kind=egoverse, zarr_replay, or hdf5_replay for recorded actions"
    )


def _max_steps(config):
    """``None`` runs until the operator stops; otherwise a positive step count."""
    value = config.get("max_steps")
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError("max_steps must be a positive integer or null")
    return value


def validate_rollout_config(config):
    """Validate browser-only rollout settings before creating robot hardware."""
    _max_steps(config)
    solver_settings = validate_ik_solver(config.get("ik_solver"))
    recovery = validate_ik_recovery(config.get("ik_recovery"))
    if recovery["enabled"] and config.get("robot", {}).get("kind") != "yam":
        raise ValueError("Differential IK recovery is supported only for Yam")
    relaxation = validate_ik_relaxation(config.get("ik_relaxation"))
    if relaxation["enabled"] and config.get("robot", {}).get("kind") != "yam":
        raise ValueError(
            "IK pose relaxation is supported only for Yam Cartesian rollout"
        )
    if solver_settings["backend"] == "pytracik":
        if config.get("robot", {}).get("kind") != "yam":
            raise ValueError("TRAC-IK is supported only for Yam Cartesian rollout")
        from egomimic.robot.yam.tracik import require_pytracik

        require_pytracik()
    for name in ("reset_on_start", "reset_home_on_restart"):
        value = config.get(name, False)
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if "execute_steps" in config:
        raise ValueError("execute_steps moved to policy.replan_every")
    preview = dict(config.get("preview", {}))
    validate_video_recording(config.get("video_recording"))
    validate_episode_recording(config.get("episode_recording"))
    policy = config.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("Rollout configuration needs a policy mapping")
    from egomimic.robot.yam_station.rollout_dashboard import validate_model_browser

    validate_model_browser(config.get("model_browser"), policy)
    if preview.get("mode") != "dashboard":
        return
    robot = config.get("robot")
    if not isinstance(robot, dict):
        raise ValueError("Dashboard rollout configuration needs a robot mapping")
    cameras = robot.get("cameras")
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError("Dashboard rollout configuration needs configured cameras")
    from egomimic.robot.yam_station.rollout_dashboard import validate_rollout_preview

    validate_rollout_preview(preview, cameras=set(cameras))


def create_preview_view(
    camera_res,
    preview,
    inference_controls=None,
    video_recording=None,
    model_browser=None,
    policy=None,
    episode_recording=None,
):
    """Select the legacy OpenCV preview or the local browser dashboard."""
    preview = dict(preview)
    if preview.get("mode") == "dashboard":
        from egomimic.robot.yam_station.rollout_dashboard import RolloutDashboard

        return RolloutDashboard(
            camera_res,
            inference_controls=inference_controls,
            video_recording=video_recording,
            episode_recording=episode_recording,
            model_browser=model_browser,
            policy=policy,
            **preview,
        )
    return CameraView(camera_res, **preview)


def _set_view_status(view, status):
    set_status = getattr(view, "set_status", None)
    if callable(set_status):
        set_status(status)


def _velocity_decision(view, details):
    """Resolve an explicit response to a whole-plan velocity-limit rejection."""
    choose = getattr(view, "choose_velocity_action", None)
    if not callable(choose):
        return "resample"
    decision = choose(details)
    if decision not in {
        "execute",
        "model_selection",
        "reconnect",
        "resample",
        "restart",
        "stop",
    }:
        raise ValueError(f"Unknown velocity-limit decision: {decision!r}")
    return decision


def _take_camera_reconnect_request(view) -> bool:
    """Consume one browser recovery request without adding a robot command path."""
    take = getattr(view, "take_camera_reconnect_request", None)
    return bool(take()) if callable(take) else False


def _take_video_recording_request(view) -> bool:
    """Consume one browser-only request to toggle display-only video capture."""
    take = getattr(view, "take_video_recording_request", None)
    return bool(take()) if callable(take) else False


def _take_model_selection_request(view):
    """Consume one dashboard-selected model without exposing a command path."""
    take = getattr(view, "take_model_selection_request", None)
    return take() if callable(take) else None


def _set_model_checkpoint(view, checkpoint) -> None:
    publish = getattr(view, "set_model_checkpoint", None)
    if callable(publish):
        publish(checkpoint)


def _policy_inference_controls(policy):
    """Read one policy's serializable, explicitly exposed runtime controls."""
    controls = getattr(policy, "inference_controls", None)
    if not callable(controls):
        return {}
    result = controls()
    if not isinstance(result, Mapping):
        raise TypeError("Policy inference_controls() must return a mapping")
    return result


def _set_inference_controls(view, controls) -> None:
    publish = getattr(view, "set_inference_controls", None)
    if callable(publish):
        publish(controls)


def _take_inference_override_request(view):
    take = getattr(view, "take_inference_override_request", None)
    return take() if callable(take) else None


def _apply_inference_overrides(policy, overrides):
    apply = getattr(policy, "apply_inference_overrides", None)
    if not callable(apply):
        raise ValueError("The selected policy exposes no inference overrides")
    return apply(overrides)


def _policy_execution_plan(policy, prediction):
    """Let the policy choose its executable prefix without model logic here."""
    select = getattr(policy, "execution_plan", None)
    actions = np.asarray(select(prediction) if callable(select) else prediction)
    if (
        actions.ndim != 2
        or actions.shape[1] != 14
        or not len(actions)
        or len(actions) > len(prediction)
        or not np.isfinite(actions).all()
    ):
        raise ValueError(
            "Policy execution plan must be a finite nonempty (N, 14) prefix"
        )
    return actions


def _set_video_recording(view, recording: bool, saved=None) -> None:
    """Publish capture state without letting the dashboard write robot state."""
    set_recording = getattr(view, "set_video_recording", None)
    if callable(set_recording):
        set_recording(recording, saved=saved)


def _take_episode_request(view):
    """Consume one browser-only episode request: start, save with an outcome, or discard."""
    take = getattr(view, "take_episode_request", None)
    request = take() if callable(take) else None
    return request if isinstance(request, Mapping) else None


def _set_episode_recording(view, recording: bool, frames: int = 0, saved=None) -> None:
    """Publish episode-recorder state without giving the dashboard a write path."""
    publish = getattr(view, "set_episode_recording", None)
    if callable(publish):
        publish(recording, frames=frames, saved=saved)


def _episode_actions(robot, commands, row, action_type):
    """Demo-format action columns for one executed tick.

    As the GELLO collector records: the joint command sent to each arm and the
    forward kinematics of that command (+ gripper); unselected arms stay zero
    like ``get_obs``. The policy's own Cartesian target is kept separately.
    """
    joints, eepose = np.zeros(14), np.zeros(14)
    for arm, command in commands.items():
        offset = ARM_OFFSET[arm]
        joints[offset : offset + 7] = command
        pose = np.asarray(robot.forward_kinematics(command[:6], arm), dtype=float)
        eepose[offset : offset + 7] = np.r_[pose, command[6]]
    target = (
        np.asarray(row, dtype=float)
        if action_type == "cartesian"
        else np.full(14, np.nan)
    )
    return joints, eepose, target


def _plan_info(policy) -> dict:
    """Inference settings and decoder stats that produced one plan, for the episode."""
    controls = {
        name: spec.get("value")
        for name, spec in _policy_inference_controls(policy).items()
        if isinstance(spec, Mapping)
    }
    decoder = getattr(getattr(policy, "adapter", None), "decoder", None)
    stats = getattr(decoder, "last_stats", None)
    return {
        "controls": controls,
        "decoder": dict(stats) if isinstance(stats, Mapping) else None,
    }


def _reset_policy_state(policy) -> None:
    """Reset inference-owned history without exposing model details to rollout."""
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()


def run_rollout(robot, policy, config, view=None):
    solver_settings = validate_ik_solver(config.get("ik_solver"))
    recovery = validate_ik_recovery(config.get("ik_recovery"))
    if recovery["enabled"] and (
        policy.action_type != "cartesian"
        or not callable(getattr(robot, "solve_ik_step", None))
    ):
        raise ValueError(
            "Differential IK recovery needs a supported Cartesian robot/policy"
        )
    relaxation = validate_ik_relaxation(config.get("ik_relaxation"))
    relaxation_available = policy.action_type == "cartesian" and callable(
        getattr(robot, "solve_ik_relaxed", None)
    )
    if relaxation["enabled"] and not relaxation_available:
        raise ValueError("This robot/policy does not support IK pose relaxation")
    set_backend = getattr(robot, "set_ik_backend", None)
    solver_available = policy.action_type == "cartesian" and callable(set_backend)
    if solver_settings["backend"] != "mujoco" and not solver_available:
        raise ValueError("TRAC-IK needs a supported Cartesian robot/policy")
    if solver_available:
        set_backend(**solver_settings)
    frequency = float(config["frequency"])
    max_steps = _max_steps(config)
    limit = float(config["max_joint_velocity"]) / frequency
    max_velocity_replans = config.get("max_velocity_replans", 0)
    if min(frequency, limit) <= 0 or not np.isfinite(limit):
        raise ValueError("Rollout frequency and velocity must be positive")
    if type(max_velocity_replans) is not int or not 0 <= max_velocity_replans <= 32:
        raise ValueError("max_velocity_replans must be an integer in [0, 32]")
    if policy.action_type not in ("joints", "cartesian"):
        raise ValueError("Unknown policy action representation")
    queue, last, step = deque(), None, 0
    policy_config = copy.deepcopy(config.get("policy"))
    waiting_since, velocity_replans = None, 0
    ik_rejections = 0
    recovery_row, recovery_ticks = None, 0
    paused = False
    pending_inference_overrides = {}
    view = view or create_preview_view(
        robot.camera_res,
        config["preview"],
        inference_controls=_policy_inference_controls(policy),
        video_recording=config.get("video_recording"),
        model_browser=config.get("model_browser"),
        policy=policy_config,
        episode_recording=config.get("episode_recording"),
    )
    video_config = validate_video_recording(config.get("video_recording"))
    video_recorder = (
        RolloutVideoRecorder(robot.camera_res, video_config)
        if video_config["enabled"]
        else None
    )
    video_recording = False
    episode_config = validate_episode_recording(config.get("episode_recording"))
    episode_recorder = (
        RolloutEpisodeRecorder(robot.camera_res, episode_config)
        if episode_config["enabled"]
        else None
    )
    episode_end_reason = "stop"
    reset_on_start = config.get("reset_on_start", False)
    reset_home_on_restart = config.get("reset_home_on_restart", False)
    wait_for_start = bool(config.get("preview", {}).get("wait_for_start", False))
    started = not wait_for_start
    model_ready = True

    def publish_ik_relaxation():
        publish = getattr(view, "set_ik_relaxation", None)
        if callable(publish):
            publish({**relaxation, "available": relaxation_available})

    publish_ik_relaxation()

    # Live gripper force limit: only robots that can change it (YamInterface).
    set_force = getattr(robot, "set_gripper_force_limit", None)
    force_limits = (
        {arm: float(robot.gripper_force_limit) for arm in robot.arms}
        if callable(set_force) and hasattr(robot, "gripper_force_limit")
        else {}
    )

    def publish_gripper_force():
        publish = getattr(view, "set_gripper_force_limits", None)
        if callable(publish) and force_limits:
            publish(dict(force_limits))

    publish_gripper_force()

    def publish_ik_solver(error=None):
        publish = getattr(view, "set_ik_solver", None)
        if callable(publish):
            publish({**solver_settings, "available": solver_available, "error": error})

    publish_ik_solver()

    def reset_to_ready():
        nonlocal last, step, waiting_since, velocity_replans, started, paused
        nonlocal ik_rejections
        queue.clear()
        last, step, waiting_since, velocity_replans, started = (
            None,
            0,
            None,
            0,
            False,
        )
        ik_rejections = 0
        paused = False
        _reset_policy_state(policy)
        clear_plan = getattr(view, "clear_action_plan", None)
        if callable(clear_plan):
            clear_plan()
        if reset_home_on_restart:
            _set_view_status(view, "Resetting YAM to configured home")
            robot.set_home()
        _set_view_status(view, "Ready — press c to start")

    def finish_video_recording():
        """Finalize recording without letting a disk failure interrupt robot control."""
        nonlocal video_recording
        if not video_recording or video_recorder is None:
            return None
        video_recording = False
        try:
            saved = video_recorder.close()
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Could not finalize rollout video: {error}")
            _set_video_recording(view, False)
            return None
        _set_video_recording(view, False, saved=saved)
        if saved is not None:
            print(
                f"Saved rollout video {saved['filename']} "
                f"({saved['frames']} frames at {saved['fps']} fps)"
            )
        return saved

    def finish_episode(outcome="unlabeled", complete=True, end_reason="saved"):
        """Stop episode capture; the recorder finalizes the file off the control loop."""
        if episode_recorder is None or not episode_recorder.recording:
            return None
        try:
            episode_id = episode_recorder.close(
                outcome=outcome, complete=complete, end_reason=end_reason
            )
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Could not finalize rollout episode: {error}")
            episode_id = None
        _set_episode_recording(view, False, frames=episode_recorder.frames)
        return episode_id

    def publish_finished_episodes():
        if episode_recorder is None:
            return
        for result in episode_recorder.take_finished():
            saved = result["saved"]
            if result["error"]:
                print(
                    f"Could not save rollout episode {result['id']}: {result['error']}"
                )
            elif result["discarded"]:
                print(f"Discarded rollout episode {result['id']}")
            elif saved is not None:
                state = "complete" if saved["complete"] else "incomplete"
                print(
                    f"Saved rollout episode {saved['filename']} ({saved['frames']} "
                    f"frames, {saved['outcome']}, {state}: {saved['end_reason']})"
                )
            _set_episode_recording(
                view,
                episode_recorder.recording,
                frames=episode_recorder.frames,
                saved=saved,
            )

    try:
        _reset_policy_state(policy)
        if reset_on_start:
            _set_view_status(view, "Resetting YAM to configured home")
            robot.set_home()
        if wait_for_start:
            _set_view_status(view, "Ready — press c to start")
        while max_steps is None or step < max_steps:
            tick = time.monotonic()
            obs = robot.get_obs()
            control = view.update(obs)
            if control in ("q", "\x1b"):
                break
            take_solver = getattr(view, "take_ik_solver_request", None)
            requested_solver = take_solver() if callable(take_solver) else None
            if solver_available and requested_solver in ("mujoco", "pytracik"):
                if requested_solver != solver_settings["backend"]:
                    # Do not retain old-backend waypoints or stale observations
                    # across model construction. A failed switch sends no command.
                    queue.clear()
                    _reset_policy_state(policy)
                    clear_plan = getattr(view, "clear_action_plan", None)
                    if callable(clear_plan):
                        clear_plan()
                    candidate = {**solver_settings, "backend": requested_solver}
                    try:
                        set_backend(**candidate)
                    except (ImportError, OSError, RuntimeError, ValueError) as error:
                        publish_ik_solver(str(error))
                        _set_view_status(view, f"IK backend unchanged: {error}")
                    else:
                        solver_settings.update(candidate)
                        publish_ik_solver()
                    continue
                publish_ik_solver()
            take_relaxation = getattr(view, "take_ik_relaxation_request", None)
            requested_relaxation = (
                take_relaxation() if callable(take_relaxation) else None
            )
            if relaxation_available and type(requested_relaxation) is bool:
                if relaxation["enabled"] != requested_relaxation:
                    relaxation["enabled"] = requested_relaxation
                    queue.clear()
                    _reset_policy_state(policy)
                publish_ik_relaxation()
            take_force = getattr(view, "take_gripper_force_request", None)
            force_request = take_force() if callable(take_force) else None
            if force_request is not None and force_limits:
                try:
                    force_limits.update(
                        set_force(force_request["value"], arm=force_request["arm"])
                    )
                except (KeyError, TypeError, ValueError, RuntimeError) as error:
                    _set_view_status(view, f"Gripper force unchanged: {error}")
                publish_gripper_force()
            bundle = _take_model_selection_request(view)
            if bundle is not None:
                # Freeze the current measured pose before the potentially slow
                # GPU checkpoint load. Old-model chunks are discarded and the
                # operator must explicitly start the new model after it loads.
                finish_video_recording()
                finish_episode(complete=False, end_reason="model_change")
                queue.clear()
                pending_inference_overrides.clear()
                last = np.asarray(obs["joint_positions"], dtype=float).copy()
                for arm in robot.arms:
                    offset = ARM_OFFSET[arm]
                    robot.set_joints(last[offset : offset + 7], arm)
                started, paused = False, False
                model_ready = False
                clear_plan = getattr(view, "clear_action_plan", None)
                if callable(clear_plan):
                    clear_plan()
                _set_view_status(view, f"Loading checkpoint {bundle.checkpoint.name}")
                try:
                    if not isinstance(policy_config, dict):
                        raise ValueError("Rollout policy configuration is unavailable")
                    candidate_config = copy.deepcopy(policy_config)
                    candidate_config["checkpoint"] = str(bundle.checkpoint)
                    candidate = load_policy(candidate_config)
                    if candidate.action_type != policy.action_type:
                        raise ValueError("Selected model changed action representation")
                except Exception as error:
                    _set_view_status(
                        view,
                        f"MODEL LOAD FAILED — rollout locked: {bundle.checkpoint.name}: {error}",
                    )
                    print(f"Could not load selected model {bundle.checkpoint}: {error}")
                else:
                    policy, policy_config = candidate, candidate_config
                    model_ready = True
                    _set_model_checkpoint(view, bundle)
                    _set_inference_controls(view, _policy_inference_controls(candidate))
                    _set_view_status(view, "Selected model loaded — press c to start")
                time.sleep(max(0.0, 1 / frequency - (time.monotonic() - tick)))
                continue
            overrides = _take_inference_override_request(view)
            if overrides:
                pending_inference_overrides.update(overrides)
            if pending_inference_overrides and (not started or paused or not queue):
                current_controls = _policy_inference_controls(policy)
                changed = {
                    name: value
                    for name, value in pending_inference_overrides.items()
                    if current_controls.get(name, {}).get("value") != value
                }
                try:
                    controls = (
                        _apply_inference_overrides(policy, changed)
                        if changed
                        else current_controls
                    )
                except (TypeError, ValueError) as error:
                    print(f"Could not apply inference override: {error}")
                    _set_view_status(view, f"Inference setting rejected: {error}")
                    _set_inference_controls(view, _policy_inference_controls(policy))
                else:
                    _set_inference_controls(view, controls)
                    if changed:
                        suffix = (
                            "next plan will use them"
                            if started and not paused
                            else "they will be used when rollout resumes"
                        )
                        _set_view_status(view, f"Inference settings updated — {suffix}")
                    else:
                        _set_view_status(view, "Inference settings unchanged")
                pending_inference_overrides.clear()
            elif pending_inference_overrides:
                _set_view_status(
                    view, "Inference settings queued — applying at next replan"
                )
            if _take_camera_reconnect_request(view):
                # Never execute a target sampled before a camera recovery. The
                # YAM method stops and reopens RGB streams only; it does not
                # close or command either follower driver.
                queue.clear()
                last, waiting_since, velocity_replans = None, None, 0
                ik_rejections = 0
                started, paused = False, False
                _reset_policy_state(policy)
                clear_plan = getattr(view, "clear_action_plan", None)
                if callable(clear_plan):
                    clear_plan()
                finish_video_recording()
                finish_episode(complete=False, end_reason="camera_reconnect")
                _set_view_status(
                    view, "Reconnecting RGB cameras — rollout control is paused"
                )
                reconnect = getattr(robot, "reconnect_cameras", None)
                try:
                    if not callable(reconnect):
                        raise RuntimeError(
                            "This robot runtime does not support camera recovery"
                        )
                    reconnect()
                except Exception as error:
                    print(f"Camera reconnect failed; replug and try again: {error}")
                    _set_view_status(view, "Camera reconnect failed — try again")
                else:
                    _set_view_status(view, "Cameras reconnected — press c to start")
                time.sleep(max(0.0, 1 / frequency - (time.monotonic() - tick)))
                continue
            if control in ("r", "R"):
                # Restart never reuses a queued target. The HPT profile also
                # returns both followers to their configured home before c can
                # begin the next policy rollout. An episode still recording is
                # kept as a complete, unlabeled take.
                finish_video_recording()
                finish_episode(end_reason="restart")
                reset_to_ready()
                time.sleep(max(0.0, 1 / frequency - (time.monotonic() - tick)))
                continue
            is_paused = getattr(view, "is_paused", None)
            requested_pause = bool(is_paused()) if callable(is_paused) else False
            if requested_pause != paused:
                queue.clear()
                _reset_policy_state(policy)
                clear_plan = getattr(view, "clear_action_plan", None)
                if callable(clear_plan):
                    clear_plan()
                if requested_pause:
                    # Replace any previously commanded target with the measured
                    # pose once. This holds both followers without advancing the
                    # policy queue or retaining an unsafe stale plan.
                    last = np.asarray(obs["joint_positions"], dtype=float).copy()
                    for arm in robot.arms:
                        offset = ARM_OFFSET[arm]
                        robot.set_joints(last[offset : offset + 7], arm)
                    _set_view_status(view, "Paused — holding current joint positions")
                else:
                    # A resumed rollout always infers from a fresh observation.
                    last = None
                    _set_view_status(view, "Running")
                paused = requested_pause
            if _take_video_recording_request(view):
                if video_recording:
                    finish_video_recording()
                elif not started:
                    _set_view_status(view, "Press c to start before recording video")
                elif video_recorder is not None:
                    try:
                        video_id = video_recorder.start()
                    except (OSError, RuntimeError, ValueError) as error:
                        print(f"Could not start rollout video recording: {error}")
                        _set_view_status(view, "Could not start local video recording")
                    else:
                        video_recording = True
                        _set_video_recording(view, True)
                        print(f"Recording rollout video {video_id}.mp4")
            if video_recording and video_recorder is not None:
                try:
                    video_recorder.append(obs)
                except (OSError, RuntimeError, ValueError) as error:
                    print(f"Rollout video recording stopped: {error}")
                    finish_video_recording()
            episode_request = _take_episode_request(view)
            if episode_request is not None and episode_recorder is not None:
                action = episode_request.get("action")
                if action == "start" and not episode_recorder.recording:
                    if not started:
                        _set_view_status(
                            view, "Press c to start before recording an episode"
                        )
                    elif not callable(getattr(robot, "forward_kinematics", None)):
                        _set_view_status(
                            view,
                            "This robot runtime has no forward kinematics; "
                            "cannot record a demo-format episode",
                        )
                    else:
                        metadata = {
                            "checkpoint": (policy_config or {}).get("checkpoint"),
                            "policy_config": policy_config,
                            "action_type": policy.action_type,
                            "frequency": frequency,
                        }
                        try:
                            episode_id = episode_recorder.start(metadata)
                        except (OSError, RuntimeError, ValueError) as error:
                            print(f"Could not start rollout episode recording: {error}")
                            _set_view_status(view, f"Episode not started: {error}")
                        else:
                            _set_episode_recording(view, True, frames=0)
                            print(f"Recording rollout episode {episode_id}.hdf5")
                elif action == "save":
                    finish_episode(
                        outcome=episode_request.get("outcome", "unlabeled"),
                        end_reason="saved",
                    )
                elif action == "discard" and episode_recorder.discard() is not None:
                    _set_episode_recording(view, False, frames=0)
            if episode_recorder is not None:
                if episode_recorder.recording and episode_recorder.failure:
                    reason = episode_recorder.failure
                    finish_episode(complete=False, end_reason=reason)
                    _set_view_status(
                        view,
                        "Episode recording stopped: "
                        + (
                            "disk space is low"
                            if reason == "low_disk"
                            else "writer error"
                        ),
                    )
                publish_finished_episodes()
            if paused:
                time.sleep(max(0.0, 1 / frequency - (time.monotonic() - tick)))
                continue
            if not started:
                if control in ("c", "C"):
                    if not model_ready:
                        _set_view_status(
                            view,
                            "MODEL LOAD FAILED — rollout locked; select a valid model",
                        )
                        continue
                    started = True
                    _set_view_status(view, "Running")
                else:
                    time.sleep(max(0.0, 1 / frequency - (time.monotonic() - tick)))
                    continue
            if not all(obs.get(name) is not None for name in robot.camera_res):
                waiting_since = tick if waiting_since is None else waiting_since
                if tick - waiting_since > float(config.get("camera_timeout", 30.0)):
                    raise TimeoutError("Rollout cameras are unavailable")
                # Resume from a fresh plan after camera loss; do not execute stale queued actions.
                queue.clear()
                time.sleep(1 / frequency)
                continue
            waiting_since = None
            if last is None:
                last = np.asarray(obs["joint_positions"], dtype=float).copy()
            if not queue:
                try:
                    inference_started = time.perf_counter()
                    prediction = np.asarray(policy.predict(obs), dtype=float)
                    inference_seconds = time.perf_counter() - inference_started
                except StopIteration:
                    break
                record_inference = getattr(view, "record_inference", None)
                if callable(record_inference):
                    record_inference(inference_seconds)
                if (
                    prediction.ndim != 2
                    or prediction.shape[1] != 14
                    or not len(prediction)
                    or not np.isfinite(prediction).all()
                ):
                    raise ValueError(
                        "Policy must return a nonempty finite (H, 14) action chunk"
                    )
                set_plan = getattr(view, "set_action_plan", None)
                if callable(set_plan):
                    # Browser overlays are display-only. The unchanged command queue
                    # below remains the sole source of robot actuation.
                    set_plan(prediction, policy.action_type)
                # Each policy owns its execution/replanning semantics. Replay
                # returns the full chunk; graph inference returns the profile's
                # operator-selected executable prefix.
                plan = _policy_execution_plan(policy, prediction)
                queue.extend(plan)
                if episode_recorder is not None and episode_recorder.recording:
                    episode_recorder.mark_plan(
                        prediction, len(plan), inference_seconds, _plan_info(policy)
                    )
            row = queue.popleft()
            commands = {}
            violations = []
            unreachable = []
            recovering = recovery["enabled"] and recovery_row is row
            for arm in () if recovering else robot.arms:
                offset = ARM_OFFSET[arm]
                target = row[offset : offset + 7]
                if policy.action_type == "cartesian":
                    try:
                        try:
                            joints = robot.solve_ik(target[:6], arm)
                        except ValueError:
                            if not relaxation["enabled"]:
                                raise
                            joints = robot.solve_ik_relaxed(
                                target[:6],
                                arm,
                                position_tolerance_m=relaxation["position_tolerance_m"],
                                rotation_tolerance_deg=relaxation[
                                    "rotation_tolerance_deg"
                                ],
                            )
                    except ValueError as error:
                        # An unreachable Cartesian target is a property of this
                        # sampled plan, not a fault. Record it and drop the plan
                        # below; the robot holds position and the next tick asks
                        # the policy for a fresh one.
                        unreachable.append((arm, str(error)))
                        continue
                    target = np.r_[joints, target[6]]
                command = joint_vector(target)
                joint_step = float(
                    np.max(np.abs(command[:6] - last[offset : offset + 6]))
                )
                if joint_step > limit + 1e-8:
                    violations.append((arm, joint_step))
                commands[arm] = command
            if recovery["enabled"] and (recovering or unreachable):
                # Once one endpoint fails, both arms track this same paired row
                # locally. Keep it queued until both endpoints are reached.
                commands, violations, unreachable = {}, [], []
                if recovery_row is not row:
                    recovery_ticks = 0
                reached = True
                recovery_limit = min(limit, recovery["max_joint_velocity"] / frequency)
                for arm in robot.arms:
                    offset = ARM_OFFSET[arm]
                    try:
                        if recovery_ticks >= recovery["max_ticks_per_target"]:
                            raise ValueError(
                                "Differential IK target exceeded its recovery tick budget"
                            )
                        result = robot.solve_ik_step(
                            row[offset : offset + 6],
                            arm,
                            measured=np.asarray(obs["joint_positions"])[
                                offset : offset + 6
                            ],
                            last_command=last[offset : offset + 6],
                            max_step=recovery_limit,
                            settings=recovery,
                        )
                        command = joint_vector(np.r_[result.joints, row[offset + 6]])
                        # Defense in depth, including measured-state tracking lag.
                        measured = np.asarray(obs["joint_positions"])[
                            offset : offset + 6
                        ]
                        if (
                            np.max(np.abs(command[:6] - last[offset : offset + 6]))
                            > recovery_limit + 1e-8
                            or np.max(np.abs(command[:6] - measured))
                            > recovery_limit + 1e-8
                        ):
                            raise ValueError(
                                "Differential IK exceeded the per-tick joint bound"
                            )
                        reached = reached and result.reached
                        commands[arm] = command
                    except ValueError as error:
                        unreachable.append((arm, str(error)))
                if not unreachable:
                    recovery_ticks += 1
                    if not reached:
                        queue.appendleft(row)
                        recovery_row = row
                        # Do not release/close a gripper before reaching its row.
                        for arm, command in commands.items():
                            command[6] = last[ARM_OFFSET[arm] + 6]
                    else:
                        recovery_row, recovery_ticks = None, 0
                else:
                    recovery_row, recovery_ticks = None, 0
            if unreachable:
                # Send neither arm, discard the rest of the chunk and wait for
                # the next plan. This is deliberately not budgeted: an
                # unreachable sample must never end the session.
                queue.clear()
                last = np.asarray(obs["joint_positions"], dtype=float).copy()
                ik_rejections += 1
                _set_view_status(
                    view,
                    f"Unreachable IK target on {[arm for arm, _ in unreachable]}; "
                    f"plan discarded, awaiting next action ({ik_rejections})",
                )
                time.sleep(max(0.0, 1 / frequency - (time.monotonic() - tick)))
                continue
            if violations:
                details = {
                    "arms": [arm for arm, _ in violations],
                    "max_joint_step": max(step_size for _, step_size in violations),
                    "limit": limit,
                }
                decision = _velocity_decision(view, details)
                if decision == "stop":
                    break
                if decision == "reconnect":
                    # The dashboard has retained the explicit recovery event.
                    # Consume it on the next safe loop tick rather than
                    # charging it against the velocity-resample budget.
                    time.sleep(max(0.0, 1 / frequency - (time.monotonic() - tick)))
                    continue
                if decision == "model_selection":
                    # The next tick consumes the validated checkpoint request
                    # before another policy target can be considered.
                    time.sleep(max(0.0, 1 / frequency - (time.monotonic() - tick)))
                    continue
                if decision == "restart":
                    finish_episode(end_reason="restart")
                    reset_to_ready()
                    time.sleep(max(0.0, 1 / frequency - (time.monotonic() - tick)))
                    continue
                if decision == "execute":
                    # An operator explicitly accepted this one complete paired
                    # target. It is never selected automatically.
                    _set_view_status(view, "Executing operator-approved plan")
                else:
                    assert decision == "resample"
                    # Do not send either arm; re-observe and request a fresh plan.
                    queue.clear()
                    last = np.asarray(obs["joint_positions"], dtype=float).copy()
                    velocity_replans += 1
                    if velocity_replans > max_velocity_replans:
                        raise ValueError(
                            f"{details['arms']} target exceeds the configured joint "
                            f"velocity after {max_velocity_replans} safe replan "
                            "attempt(s). No unsafe command was sent."
                        )
                    _set_view_status(
                        view,
                        f"Rejected velocity-unsafe plan; resampling "
                        f"{velocity_replans}/{max_velocity_replans}",
                    )
                    time.sleep(max(0.0, 1 / frequency - (time.monotonic() - tick)))
                    continue
            # Validate both arms before sending either command.
            for arm, command in commands.items():
                robot.set_joints(command, arm)
                offset = ARM_OFFSET[arm]
                last[offset : offset + 7] = command
            if episode_recorder is not None and episode_recorder.recording:
                # One demo-format row per executed tick: this tick's observation
                # and the paired command just sent. Recording never blocks or
                # interrupts control; a failure only ends the episode.
                try:
                    joints, eepose, target = _episode_actions(
                        robot, commands, row, policy.action_type
                    )
                    episode_recorder.append(obs, joints, eepose, target)
                except (
                    Exception
                ) as error:  # the command is already sent; only the episode ends
                    print(f"Rollout episode recording stopped: {error}")
                    finish_episode(complete=False, end_reason="writer_error")
                else:
                    _set_episode_recording(view, True, frames=episode_recorder.frames)
            velocity_replans = 0
            ik_rejections = 0
            step += 1
            _set_view_status(view, "Running")
            time.sleep(max(0.0, 1 / frequency - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        raise  # an operator Ctrl-C ends the session like q
    except BaseException:
        episode_end_reason = "error"
        raise
    finally:
        finish_video_recording()
        finish_episode(complete=False, end_reason=episode_end_reason)
        if episode_recorder is not None:
            if not episode_recorder.wait(timeout=60):
                print("Rollout episode is still being written; it will finish on exit")
            publish_finished_episodes()
        view.close()
        close_policy = getattr(policy, "close", None)
        if callable(close_policy):
            close_policy()
    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument(
        "--check-config",
        action="store_true",
        help="validate dashboard/calibration configuration only; open no devices",
    )
    checks.add_argument(
        "--check-model",
        action="store_true",
        help="load the configured EgoVerse checkpoint on CPU; open no devices",
    )
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if args.check_model:
        if config.get("policy", {}).get("kind") != "egoverse":
            raise ValueError("--check-model requires an EgoVerse checkpoint policy")
        config["policy"]["device"] = "cpu"
    validate_rollout_config(config)
    if args.check_config:
        print(
            "Rollout configuration is valid; no robot, CAN, or camera device was opened."
        )
        return
    # Load/validate policy artifacts before opening hardware.
    policy = load_policy(config["policy"])
    if args.check_model:
        try:
            parameters = list(policy.model.nets.parameters())
            if any(parameter.device.type != "cpu" for parameter in parameters):
                raise RuntimeError("Model check must restore every parameter on CPU")
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "checkpoint": config["policy"]["checkpoint"],
                        "device": "cpu",
                        "parameters": sum(
                            parameter.numel() for parameter in parameters
                        ),
                        "action_type": policy.action_type,
                        "hardware_opened": False,
                    },
                    sort_keys=True,
                )
            )
        finally:
            close_policy = getattr(policy, "close", None)
            if callable(close_policy):
                close_policy()
        return
    robot = create_robot(config["robot"])
    try:
        run_rollout(robot, policy, config)
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()


if __name__ == "__main__":
    main()
