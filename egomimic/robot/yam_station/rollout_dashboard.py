"""Loopback-only browser view for YAM rollout cameras and action overlays.

This module exposes display and high-level rollout requests only. It cannot arm,
home, or otherwise send robot commands; :mod:`egomimic.robot.yam_station.rollout` remains
the sole command path.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
import webbrowser
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from egomimic.robot.yam_station.interface import ARM_OFFSET
from egomimic.robot.yam_station.rollout_episode import (
    OUTCOMES,
    list_rollout_episodes,
    validate_episode_recording,
)
from egomimic.robot.yam_station.rollout_video import (
    list_rollout_videos,
    rollout_video_path,
    validate_video_recording,
)

DEFAULTS = {
    "enabled": True,
    "mode": "dashboard",
    "width": 640,
    "host": "127.0.0.1",
    "port": 8081,
    "open_browser": True,
    "camera_hz": 12,
    "jpeg_quality": 80,
    "wait_for_start": False,
    "action_overlay": {
        "initial_enabled": False,
        "camera": "front_img_1",
        "calibration_path": None,
        "arm_channels": None,
    },
}
STATIC = Path(__file__).with_name("rollout_dashboard_static")
# Private-range close code: this tab lost the dashboard to a newer one.
SUPERSEDED_CLOSE_CODE = 4001
CHECKPOINT_SUFFIXES = frozenset({".ckpt"})
MODEL_BROWSER_DEFAULTS = {"enabled": False, "root": None}


class _RolloutTimer:
    """Monotonic active time since Continue; callers hold the dashboard lock."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.elapsed = 0.0
        self.running_since = None

    def resume(self):
        if self.running_since is None:
            self.running_since = self.clock()

    def pause(self):
        if self.running_since is not None:
            self.elapsed += max(0.0, self.clock() - self.running_since)
            self.running_since = None

    def reset(self):
        self.elapsed = 0.0
        self.running_since = None

    def snapshot(self):
        elapsed = self.elapsed
        if self.running_since is not None:
            elapsed += max(0.0, self.clock() - self.running_since)
        return {"elapsed_seconds": elapsed, "running": self.running_since is not None}


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _validate_inference_controls(controls: Mapping[str, object] | None) -> dict:
    """Validate the policy-produced schema before it reaches browser code."""
    controls = {} if controls is None else controls
    if not isinstance(controls, Mapping):
        raise ValueError("Inference controls must be a mapping")
    result = {}
    for name, spec in controls.items():
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or name.startswith("_")
            or not isinstance(spec, Mapping)
        ):
            raise ValueError("Inference controls must have safe named mappings")
        label, description = spec.get("label"), spec.get("description", "")
        from egomimic.robot.yam_station.egoverse_policy import validate_control_value

        if not isinstance(label, str) or not label or not isinstance(description, str):
            raise ValueError(f"Inference control {name!r} has an invalid schema")
        validate_control_value(name, spec, spec.get("value"))
        result[name] = {
            key: deepcopy(value)
            for key, value in spec.items()
            if key
            in {
                "label",
                "description",
                "type",
                "min",
                "max",
                "step",
                "choices",
                "value",
            }
        }
    return result


@dataclass(frozen=True)
class CheckpointBundle:
    """One EgoVerse checkpoint; its config tree and norm stats are stored inside."""

    checkpoint: Path


class CheckpointBrowser:
    """Filesystem browser constrained to one explicitly configured checkpoint root."""

    def __init__(self, root: str | Path) -> None:
        if not isinstance(root, (str, Path)):
            raise ValueError(
                "model_browser.root must be an existing absolute directory"
            )
        root = Path(root)
        if not root.is_absolute() or not root.is_dir():
            raise ValueError(
                "model_browser.root must be an existing absolute directory"
            )
        self.root = root.resolve(strict=True)

    def _resolve(self, relative: object) -> Path:
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
        ):
            raise ValueError("Checkpoint browser paths must be nonempty and relative")
        try:
            path = (self.root / relative).resolve(strict=True)
        except OSError as error:
            raise ValueError("Checkpoint browser path does not exist") from error
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError(
                "Checkpoint browser path escapes model_browser.root"
            ) from error
        return path

    def relative(self, path: str | Path) -> str:
        path = Path(path).resolve(strict=True)
        try:
            relative = path.relative_to(self.root)
        except ValueError as error:
            raise ValueError("Checkpoint is outside model_browser.root") from error
        return "." if not relative.parts else relative.as_posix()

    def resolve_bundle(self, relative: object) -> CheckpointBundle:
        path = self._resolve(relative)
        if path.suffix.lower() not in CHECKPOINT_SUFFIXES or not path.is_file():
            raise ValueError("Selected model must be an existing .ckpt file")
        return CheckpointBundle(checkpoint=path)

    def validate_policy(self, policy: Mapping[str, object]) -> CheckpointBundle:
        checkpoint = policy.get("checkpoint")
        if not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute():
            raise ValueError("policy.checkpoint must be an absolute path")
        return self.resolve_bundle(self.relative(checkpoint))

    def list_directory(self, relative: object = ".", query: object = "") -> dict:
        directory = self._resolve(relative)
        if not directory.is_dir():
            raise ValueError("Checkpoint browser path is not a directory")
        if not isinstance(query, str) or len(query) > 200:
            raise ValueError(
                "Checkpoint search must be a string of at most 200 characters"
            )
        needle = query.strip().casefold()
        entries = []
        try:
            children = tuple(directory.iterdir())
        except OSError as error:
            raise ValueError("Could not read checkpoint browser directory") from error
        for child in sorted(
            children, key=lambda entry: (not entry.is_dir(), entry.name.lower())
        ):
            if child.name.startswith("."):
                continue
            if needle and needle not in child.name.casefold():
                continue
            try:
                resolved = child.resolve(strict=True)
                child_relative = self.relative(resolved)
            except (OSError, ValueError):
                # Broken links and links outside the configured root never
                # become browser-visible candidates.
                continue
            if resolved.is_dir():
                entries.append(
                    {"type": "directory", "name": child.name, "path": child_relative}
                )
            elif resolved.is_file() and resolved.suffix.lower() in CHECKPOINT_SUFFIXES:
                try:
                    self.resolve_bundle(child_relative)
                except ValueError:
                    continue
                entries.append(
                    {"type": "checkpoint", "name": child.name, "path": child_relative}
                )
        parent = None if directory == self.root else self.relative(directory.parent)
        return {"path": self.relative(directory), "parent": parent, "entries": entries}


def validate_model_browser(
    config: Mapping[str, object] | None, policy: Mapping[str, object] | None = None
) -> CheckpointBrowser | None:
    """Validate the optional, local-only model-picker root before hardware opens."""
    config = {} if config is None else dict(config)
    unknown = config.keys() - MODEL_BROWSER_DEFAULTS.keys()
    if unknown:
        raise ValueError(
            "Unknown model_browser option(s): " + ", ".join(sorted(unknown))
        )
    result = {**MODEL_BROWSER_DEFAULTS, **config}
    if not _require_bool(result["enabled"], "model_browser.enabled"):
        return None
    browser = CheckpointBrowser(result["root"])
    if policy is not None:
        browser.validate_policy(policy)
    return browser


def _matrix(value: object, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"{name} must be a homogeneous rigid transform")
    return matrix


@dataclass(frozen=True)
class ActionOverlay:
    """Projection parameters for physical Cartesian YAM action plans."""

    camera: str
    intrinsics: np.ndarray
    distortion: np.ndarray
    camera_T_base: dict[str, np.ndarray]

    def _project(
        self, points_base: np.ndarray, arm: str
    ) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        points_base = np.asarray(points_base, dtype=float)
        if points_base.ndim != 2 or points_base.shape[1] != 3:
            raise ValueError("Action-overlay points must have shape (N, 3)")
        points_h = np.c_[points_base, np.ones(len(points_base))]
        points_camera = (self.camera_T_base[arm] @ points_h.T).T[:, :3]
        visible = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > 0.02)
        pixels = np.full((len(points_base), 2), np.nan, dtype=float)
        if np.any(visible):
            projected, _ = cv2.projectPoints(
                points_camera[visible].reshape(-1, 1, 3),
                np.zeros(3),
                np.zeros(3),
                self.intrinsics,
                self.distortion,
            )
            pixels[visible] = projected.reshape(-1, 2)
        return pixels, visible

    @staticmethod
    def _draw_path(
        image: np.ndarray, pixels: np.ndarray, visible: np.ndarray, color, label: str
    ) -> None:
        import cv2

        height, width = image.shape[:2]
        start = 0
        for index in range(len(pixels) + 1):
            if index < len(pixels) and visible[index]:
                continue
            segment = pixels[start:index]
            if len(segment):
                points = np.round(segment).astype(np.int32)
                if len(points) > 1:
                    cv2.polylines(image, [points], False, color, 2, cv2.LINE_AA)
                x, y = points[0]
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(image, (x, y), 5, color, -1, cv2.LINE_AA)
                    cv2.putText(
                        image,
                        label,
                        (x + 7, max(14, y - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
            start = index + 1

    def draw(self, image: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Draw a Cartesian plan on a BGR top-camera frame without changing it."""
        image = np.asarray(image)
        actions = np.asarray(actions, dtype=float)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError("Action-overlay image must be a uint8 HxWx3 BGR frame")
        if (
            actions.ndim != 2
            or actions.shape[1] != 14
            or not len(actions)
            or not np.isfinite(actions).all()
        ):
            raise ValueError(
                "Action-overlay plan must be a nonempty finite (H, 14) array"
            )
        output = image.copy()
        for arm, color, label in (
            ("left", (0, 190, 255), "left"),
            ("right", (255, 185, 80), "right"),
        ):
            offset = ARM_OFFSET[arm]
            pixels, visible = self._project(actions[:, offset : offset + 3], arm)
            self._draw_path(output, pixels, visible, color, label)
        return output


def load_action_overlay(
    config: Mapping[str, object], cameras: set[str] | None = None
) -> ActionOverlay:
    """Load the pinned overhead-camera calibration without opening a device."""
    options = dict(config)
    expected = {"initial_enabled", "camera", "calibration_path", "arm_channels"}
    unknown = options.keys() - expected
    if unknown:
        raise ValueError(
            f"Unknown preview.action_overlay option(s): {', '.join(sorted(unknown))}"
        )
    if not _require_bool(
        options.get("initial_enabled"), "preview.action_overlay.initial_enabled"
    ):
        # The calibration is still required: the browser can turn the overlay on later.
        pass
    camera = options.get("camera")
    if not isinstance(camera, str) or not camera:
        raise ValueError("preview.action_overlay.camera must be a nonempty camera name")
    if cameras is not None and camera not in cameras:
        raise ValueError(
            "preview.action_overlay.camera is not a configured rollout camera"
        )
    path = options.get("calibration_path")
    if not isinstance(path, str) or not path:
        raise ValueError("preview.action_overlay.calibration_path is required")
    arm_channels = options.get("arm_channels")
    if not isinstance(arm_channels, Mapping) or set(arm_channels) != set(ARM_OFFSET):
        raise ValueError(
            "preview.action_overlay.arm_channels must map exactly left/right"
        )
    if not all(
        isinstance(channel, str) and channel for channel in arm_channels.values()
    ):
        raise ValueError("preview.action_overlay.arm_channels values must be nonempty")

    try:
        payload = yaml.safe_load(Path(path).read_text())
    except OSError as error:
        raise ValueError(
            f"Could not read action-overlay calibration: {path}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError("Action-overlay calibration must be a YAML mapping")
    if payload.get("convention") != "point_base = base_T_camera @ point_camera":
        raise ValueError("Action-overlay calibration convention is unsupported")
    intrinsics = np.asarray(payload.get("K"), dtype=float)
    distortion = np.asarray(payload.get("dist"), dtype=float).reshape(-1)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("Action-overlay calibration K must be a finite 3x3 matrix")
    if distortion.shape not in ((4,), (5,), (8,)) or not np.isfinite(distortion).all():
        raise ValueError(
            "Action-overlay calibration dist must have 4, 5, or 8 finite values"
        )
    channels = payload.get("channels")
    if not isinstance(channels, Mapping):
        raise ValueError("Action-overlay calibration has no channel transforms")
    camera_T_base = {}
    for arm, channel in arm_channels.items():
        entry = channels.get(channel)
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"Action-overlay calibration has no transform for {arm}/{channel}"
            )
        base_T_camera = _matrix(entry.get("base_T_camera"), f"{arm}.base_T_camera")
        camera_T_base[arm] = np.linalg.inv(base_T_camera)
    return ActionOverlay(camera, intrinsics, distortion, camera_T_base)


def validate_rollout_preview(
    config: Mapping[str, object] | None, cameras: set[str] | None = None
) -> tuple[dict, ActionOverlay]:
    """Validate browser/overlay settings before the robot factory is called."""
    config = {} if config is None else dict(config)
    unknown = config.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown preview option(s): {', '.join(sorted(unknown))}")
    result = {**deepcopy(DEFAULTS), **config}
    for name in ("enabled", "open_browser", "wait_for_start"):
        _require_bool(result[name], f"preview.{name}")
    if result["mode"] != "dashboard":
        raise ValueError("preview.mode must be `dashboard`")
    if result["host"] not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("The rollout dashboard must bind only to loopback")
    for name, lower, upper in (
        ("width", 160, 1280),
        ("port", 1024, 65535),
        ("camera_hz", 1, 30),
        ("jpeg_quality", 1, 100),
    ):
        value = result[name]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"preview.{name} must be an integer in [{lower}, {upper}]")
    action_overlay = result["action_overlay"]
    if not isinstance(action_overlay, Mapping):
        raise ValueError("preview.action_overlay must be a mapping")
    overlay = load_action_overlay(action_overlay, cameras=cameras)
    result["action_overlay"] = dict(action_overlay)
    return result, overlay


def _jpeg_bytes(frame: np.ndarray, width: int, quality: int) -> bytes:
    import cv2

    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("Camera frame must be a uint8 HxWx3 BGR image")
    height, original_width = frame.shape[:2]
    if not height or not original_width:
        raise ValueError("Camera frame must be nonempty")
    scale = min(1.0, width / original_width)
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (round(original_width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Could not JPEG-encode camera frame")
    return encoded.tobytes()


def _jpeg_data_url(frame: np.ndarray, width: int, quality: int) -> str:
    """One still image for a caller that wants a self-contained string.

    The live view does not use this: a fresh `data:` URL per camera per frame is
    a new document resource the browser keeps, so a long rollout grows the tab
    without bound. Frames go to the browser as binary messages instead.
    """
    return "data:image/jpeg;base64," + base64.b64encode(
        _jpeg_bytes(frame, width, quality)
    ).decode("ascii")


async def _broadcast_dashboard_message(clients, message, images=()) -> set:
    """Send one frame without letting one stale browser kill the dashboard.

    Camera images follow the JSON status as binary messages, in the order named
    by `message["images"]`.
    """
    disconnected = set()
    for client in tuple(clients):
        if client.closed:
            disconnected.add(client)
            continue
        try:
            await client.send_json(message)
            for payload in images:
                await client.send_bytes(payload)
        except (ConnectionError, RuntimeError):
            # A tab may reload or a Wi-Fi/X11 bridge may reset during a frame.
            # That client reconnects independently; the local server stays alive.
            disconnected.add(client)
    return disconnected


class RolloutDashboard:
    """Three-camera rollout view with a display-only Cartesian action overlay."""

    def __init__(
        self,
        cameras,
        inference_controls=None,
        video_recording=None,
        model_browser=None,
        policy=None,
        episode_recording=None,
        **config,
    ) -> None:
        self.cameras = tuple(cameras)
        if not self.cameras:
            raise ValueError("The rollout dashboard needs at least one camera")
        self.config, self.overlay = validate_rollout_preview(config, set(self.cameras))
        self.video_recording_config = validate_video_recording(video_recording)
        self.episode_recording_config = validate_episode_recording(episode_recording)
        self.model_browser = validate_model_browser(model_browser, policy)
        if not self.config["enabled"]:
            raise ValueError(
                "preview.enabled must be true when preview.mode is dashboard"
            )
        self._frames: dict[str, np.ndarray] = {}
        self._plan: np.ndarray | None = None
        self._overlay_status = "Waiting for a Cartesian policy plan"
        self._overlay_enabled = self.config["action_overlay"]["initial_enabled"]
        self._inference_controls = _validate_inference_controls(inference_controls)
        self._pending_inference_overrides: dict[str, object] = {}
        self._inference_controls_revision = 0
        self._ik_relaxation = {"available": False, "enabled": False}
        self._ik_relaxation_request = None
        self._gripper_force_limits: dict[str, float] = {}
        self._gripper_force_request = None
        self._ik_solver = {"available": False, "backend": "mujoco", "error": None}
        self._ik_solver_request = None
        self._inference_ms = deque(maxlen=20)
        self._rollout_timer = _RolloutTimer()
        self._video_recording = False
        self._video_last_saved: dict | None = None
        self._episode_recording = False
        self._episode_frames = 0
        self._episode_last_saved: dict | None = None
        self._episode_request: dict | None = None
        self._checkpoint = (
            None
            if self.model_browser is None
            else self.model_browser.relative(Path(policy["checkpoint"]))
        )
        self._selected_model: CheckpointBundle | None = None
        self._wait_for_start = self.config["wait_for_start"]
        self._status = (
            "Ready — press c to start" if self._wait_for_start else "Starting"
        )
        self._updated_at = 0.0
        self._clients = 0
        self._lock = threading.Lock()
        self._quit_requested = threading.Event()
        self._start_requested = threading.Event()
        self._restart_requested = threading.Event()
        self._camera_reconnect_requested = threading.Event()
        self._video_record_requested = threading.Event()
        self._paused = threading.Event()
        self._velocity_decision_ready = threading.Event()
        self._velocity_prompt: dict | None = None
        self._velocity_decision: str | None = None
        if not self._wait_for_start:
            self._start_requested.set()
            self._rollout_timer.resume()
        self._shutdown = threading.Event()
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run, name="yam_rollout_dashboard", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=5):
            self.close()
            raise RuntimeError("Timed out starting local rollout dashboard")
        if self._error is not None:
            self.close()
            raise RuntimeError(
                f"Could not start local rollout dashboard: {self._error}"
            )
        host = self.config["host"]
        url_host = f"[{host}]" if ":" in host else host
        self.url = f"http://{url_host}:{self.config['port']}"
        print(f"Rollout dashboard: {self.url}")
        if self.config["open_browser"]:
            threading.Thread(
                target=webbrowser.open, args=(self.url,), daemon=True
            ).start()

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = str(status)

    def request_start(self) -> None:
        """Begin policy control on the rollout loop's next safe tick."""
        with self._lock:
            self._start_requested.set()
            if not self._paused.is_set():
                self._rollout_timer.resume()

    def request_restart(self) -> None:
        """Return to ready state and discard any displayed action plan."""
        self._start_requested.clear()
        self._paused.clear()
        self._camera_reconnect_requested.clear()
        with self._lock:
            self._rollout_timer.reset()
            # A top-level Restart may arrive while the rollout thread is blocked
            # on a velocity decision. Remove that stale warning before it returns
            # to the ready gate.
            self._velocity_prompt = None
            self._velocity_decision = None
            self._velocity_decision_ready.clear()
        self._restart_requested.set()

    def request_camera_reconnect(self) -> None:
        """Pause policy control until the loop has rebuilt RGB streams."""
        self._start_requested.clear()
        self._paused.clear()
        with self._lock:
            self._rollout_timer.pause()
            self._status = "Camera reconnect requested — control is paused"
        self._camera_reconnect_requested.set()

    def request_video_recording(self) -> None:
        """Ask the rollout loop to start or save display-only MP4 recording."""
        if self.video_recording_config["enabled"]:
            self._video_record_requested.set()

    def request_episode_action(self, action: object, outcome: object = None) -> bool:
        """Queue start, save (with an outcome label) or discard of a rollout episode.

        The rollout loop owns the recorder; the newest request replaces any
        that the loop has not consumed yet."""
        if not self.episode_recording_config["enabled"]:
            return False
        if action == "save":
            if outcome not in OUTCOMES:
                return False
            request = {"action": "save", "outcome": outcome}
        elif action in ("start", "discard"):
            request = {"action": action}
        else:
            return False
        with self._lock:
            self._episode_request = request
        return True

    def take_episode_request(self) -> dict | None:
        """Consume one browser episode request on the rollout control loop."""
        with self._lock:
            request, self._episode_request = self._episode_request, None
        return request

    def set_episode_recording(
        self,
        recording: bool,
        frames: int = 0,
        saved: Mapping[str, object] | None = None,
    ) -> None:
        """Publish recorder-owned episode state without adding a dashboard write path."""
        with self._lock:
            self._episode_recording = bool(recording)
            self._episode_frames = int(frames)
            if saved is not None:
                self._episode_last_saved = dict(saved)

    def request_model_selection(self, relative: object) -> None:
        """Queue a selected model; rollout owns the actual model load."""
        if self.model_browser is None:
            return
        try:
            bundle = self.model_browser.resolve_bundle(relative)
        except ValueError:
            return
        self._start_requested.clear()
        self._paused.clear()
        with self._lock:
            self._rollout_timer.pause()
            self._selected_model = bundle
            self._status = (
                f"Model selected: {self.model_browser.relative(bundle.checkpoint)} — "
                "control is paused"
            )

    def take_model_selection_request(self) -> CheckpointBundle | None:
        """Return one validated model selected by the focused browser tab."""
        with self._lock:
            checkpoint, self._selected_model = self._selected_model, None
        return checkpoint

    def set_model_checkpoint(self, bundle: CheckpointBundle) -> None:
        """Publish a successful rollout-owned model change to the browser."""
        if self.model_browser is None:
            return
        with self._lock:
            self._checkpoint = self.model_browser.relative(bundle.checkpoint)

    def take_video_recording_request(self) -> bool:
        """Consume one browser recording toggle on the rollout control loop."""
        if not self._video_record_requested.is_set():
            return False
        self._video_record_requested.clear()
        return True

    def set_video_recording(
        self, recording: bool, saved: Mapping[str, object] | None = None
    ) -> None:
        """Publish recorder-owned state without adding a dashboard write path."""
        with self._lock:
            self._video_recording = bool(recording)
            if saved is not None:
                self._video_last_saved = dict(saved)

    def take_camera_reconnect_request(self) -> bool:
        """Return one explicit browser request to the single rollout loop."""
        if not self._camera_reconnect_requested.is_set():
            return False
        self._camera_reconnect_requested.clear()
        return True

    def request_pause(self, paused: bool) -> None:
        """Pause only an active rollout; the loop owns the physical hold."""
        with self._lock:
            if self._velocity_prompt is not None:
                return
            if paused and not self._start_requested.is_set():
                return
            if paused:
                self._paused.set()
                self._rollout_timer.pause()
            else:
                self._paused.clear()
                if self._start_requested.is_set():
                    self._rollout_timer.resume()

    def is_paused(self) -> bool:
        """Return the browser's requested policy-control pause state."""
        return self._paused.is_set()

    def request_inference_overrides(self, overrides: object) -> bool:
        """Atomically queue values explicitly exposed by the selected profile."""
        from egomimic.robot.yam_station.egoverse_policy import validate_control_value

        if not isinstance(overrides, Mapping) or not overrides:
            return False
        with self._lock:
            validated = {}
            for name, value in overrides.items():
                if not isinstance(name, str):
                    return False
                spec = self._inference_controls.get(name)
                if spec is None:
                    return False
                try:
                    validate_control_value(name, spec, value)
                except (ValueError, TypeError):
                    return False
                validated[name] = value
            self._pending_inference_overrides.update(validated)
        return True

    def set_ik_solver(self, settings) -> None:
        with self._lock:
            self._ik_solver = dict(settings)

    def request_ik_solver(self, backend) -> bool:
        with self._lock:
            if (
                backend not in ("mujoco", "pytracik")
                or not self._ik_solver["available"]
            ):
                return False
            self._ik_solver_request = backend
            return True

    def take_ik_solver_request(self):
        with self._lock:
            request, self._ik_solver_request = self._ik_solver_request, None
            return request

    def set_ik_relaxation(self, settings) -> None:
        """Publish settings applied by the rollout thread, never command a robot."""
        with self._lock:
            self._ik_relaxation = dict(settings)

    def request_ik_relaxation(self, enabled) -> bool:
        with self._lock:
            if type(enabled) is not bool or not self._ik_relaxation["available"]:
                return False
            self._ik_relaxation_request = enabled
        return True

    def take_ik_relaxation_request(self):
        with self._lock:
            request, self._ik_relaxation_request = self._ik_relaxation_request, None
        return request

    def set_gripper_force_limits(self, limits) -> None:
        """Publish the limits the rollout thread applied, never command a robot."""
        with self._lock:
            self._gripper_force_limits = {arm: float(v) for arm, v in limits.items()}

    def request_gripper_force(self, request) -> bool:
        """Queue one arm's new force limit (1-50 N) for the rollout thread."""
        if not isinstance(request, Mapping):
            return False
        try:
            arm, value = request.get("arm"), float(request.get("value"))
        except (TypeError, ValueError):
            return False
        with self._lock:
            if arm not in self._gripper_force_limits or not 1 <= value <= 50:
                return False
            self._gripper_force_request = {"arm": arm, "value": value}
        return True

    def take_gripper_force_request(self):
        with self._lock:
            request, self._gripper_force_request = self._gripper_force_request, None
        return request

    def request_inference_override(self, name: object, value: object) -> None:
        """Compatibility wrapper for callers submitting one declared value."""
        self.request_inference_overrides({name: value})

    def take_inference_override_request(self) -> dict[str, object] | None:
        """Consume the latest validated UI values on the rollout thread."""
        with self._lock:
            if not self._pending_inference_overrides:
                return None
            result, self._pending_inference_overrides = (
                self._pending_inference_overrides,
                {},
            )
        return result

    def set_inference_controls(self, controls) -> None:
        """Publish the active model profile and discard stale model values."""
        controls = _validate_inference_controls(controls)
        with self._lock:
            self._inference_controls = controls
            self._pending_inference_overrides.clear()
            self._inference_controls_revision += 1

    def record_inference(self, seconds: float) -> None:
        """Record end-to-end wall time for one plan becoming command-ready."""
        milliseconds = float(seconds) * 1000
        if not np.isfinite(milliseconds) or milliseconds < 0:
            return
        with self._lock:
            self._inference_ms.append(milliseconds)

    def clear_action_plan(self) -> None:
        """Remove the display-only overlay after a restart."""
        with self._lock:
            self._plan = None
            self._overlay_status = "Waiting for a Cartesian policy plan"

    def choose_velocity_action(self, details) -> str:
        """Wait for an explicit dashboard decision before an unsafe plan runs."""
        prompt = {
            "arms": list(details["arms"]),
            "max_joint_step": float(details["max_joint_step"]),
            "limit": float(details["limit"]),
        }
        with self._lock:
            self._velocity_prompt = prompt
            self._velocity_decision = None
            self._status = "Velocity limit reached — choose an action"
            self._velocity_decision_ready.clear()
        while not self._shutdown.is_set():
            if self._quit_requested.is_set():
                return "stop"
            if self._camera_reconnect_requested.is_set():
                # Let the regular loop consume the recovery action rather than
                # treating it as a velocity-limit override or a home reset.
                return "reconnect"
            with self._lock:
                if self._selected_model is not None:
                    self._velocity_prompt = None
                    return "model_selection"
            if self._restart_requested.is_set():
                self._restart_requested.clear()
                return "restart"
            if self._velocity_decision_ready.wait(timeout=0.1):
                with self._lock:
                    decision = self._velocity_decision
                    self._velocity_prompt = None
                    self._velocity_decision = None
                    self._velocity_decision_ready.clear()
                if decision is not None:
                    return decision
        return "stop"

    def set_action_plan(self, actions: np.ndarray, action_type: str) -> None:
        """Publish a plan for drawing only; it never changes the command queue."""
        with self._lock:
            if action_type != "cartesian":
                self._plan = None
                self._overlay_status = (
                    "Action overlay is available for Cartesian policies"
                )
                return
            actions = np.asarray(actions, dtype=float)
            if (
                actions.ndim != 2
                or actions.shape[1] != 14
                or not len(actions)
                or not np.isfinite(actions).all()
            ):
                self._plan = None
                self._overlay_status = "No finite Cartesian action plan is available"
                return
            self._plan = actions.copy()
            self._overlay_status = "Cartesian action plan"

    def update(self, obs) -> str | None:
        """Publish observations and return the existing quit key, if requested."""
        if self._error is not None:
            # A server-wide failure would otherwise leave the robot running with
            # no operator view or stop control. A single browser failure is
            # handled inside _broadcast_dashboard_message instead.
            return "q"
        with self._lock:
            if self._paused.is_set():
                self._status = "Paused — holding current joint positions"
            elif not self._start_requested.is_set():
                self._status = "Ready — press c to start"
            if self._clients:
                self._frames = {
                    name: np.ascontiguousarray(frame).copy()
                    for name in self.cameras
                    if (frame := obs.get(name)) is not None
                }
                self._updated_at = time.monotonic()
        if self._quit_requested.is_set():
            return "q"
        if self._restart_requested.is_set():
            self._restart_requested.clear()
            return "r"
        return "c" if self._start_requested.is_set() else None

    def close(self) -> None:
        with self._lock:
            self._rollout_timer.pause()
        self._shutdown.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3)

    def _snapshot(self) -> dict:
        with self._lock:
            inference_ms = tuple(self._inference_ms)
            mean_inference_ms = (
                sum(inference_ms) / len(inference_ms) if inference_ms else None
            )
            return {
                "frames": self._frames.copy(),
                "plan": None if self._plan is None else self._plan.copy(),
                "overlay_enabled": self._overlay_enabled,
                "overlay_status": self._overlay_status,
                "status": self._status,
                "updated_at": self._updated_at,
                "paused": self._paused.is_set(),
                "started": self._start_requested.is_set(),
                "rollout_time": self._rollout_timer.snapshot(),
                "inference_controls": deepcopy(self._inference_controls),
                "ik_relaxation": dict(self._ik_relaxation),
                "ik_solver": dict(self._ik_solver),
                "gripper_force_limits": dict(self._gripper_force_limits),
                "inference_controls_revision": self._inference_controls_revision,
                "inference": {
                    "samples": len(inference_ms),
                    "last_ms": None if not inference_ms else inference_ms[-1],
                    "mean_ms": mean_inference_ms,
                    "plans_per_second": (
                        None if not mean_inference_ms else 1000.0 / mean_inference_ms
                    ),
                },
                "video_recording": self._video_recording,
                "video_last_saved": (
                    None
                    if self._video_last_saved is None
                    else self._video_last_saved.copy()
                ),
                "episode_recording": self._episode_recording,
                "episode_frames": self._episode_frames,
                "episode_last_saved": (
                    None
                    if self._episode_last_saved is None
                    else self._episode_last_saved.copy()
                ),
                "model_browser_enabled": self.model_browser is not None,
                "checkpoint": self._checkpoint,
                "velocity_prompt": (
                    None
                    if self._velocity_prompt is None
                    else self._velocity_prompt.copy()
                ),
            }

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except Exception as error:
            self._error = error
            self._ready.set()

    async def _serve(self) -> None:
        from aiohttp import web

        clients: set[web.WebSocketResponse] = set()
        closing: set[asyncio.Task] = set()

        async def index(_request):
            return web.FileResponse(
                STATIC / "index.html", headers={"Cache-Control": "no-store"}
            )

        async def asset(request):
            path = STATIC / request.match_info["name"]
            if not path.is_file() or path.parent != STATIC:
                raise web.HTTPNotFound()
            return web.FileResponse(path, headers={"Cache-Control": "no-store"})

        async def websocket(request):
            ws = web.WebSocketResponse(heartbeat=10, max_msg_size=1024)
            await ws.prepare(request)
            clients.add(ws)
            # One station, one operator, one view. Every rollout launch opens
            # another browser tab and a tab from a finished run reconnects to
            # the next one, so without this they accumulate, and each one costs
            # a full-rate stream and its own JPEG decoding. The newest tab takes
            # the dashboard; the rest are closed and told not to come back.
            superseded = [client for client in clients if client is not ws]
            clients.difference_update(superseded)
            with self._lock:
                self._clients = len(clients)
                overlay_enabled = self._overlay_enabled
                paused = self._paused.is_set()
                started = self._start_requested.is_set()
                inference_controls = deepcopy(self._inference_controls)
                inference_controls_revision = self._inference_controls_revision
                ik_relaxation = dict(self._ik_relaxation)
                ik_solver = dict(self._ik_solver)
                gripper_force_limits = dict(self._gripper_force_limits)
                checkpoint = self._checkpoint
                rollout_time = self._rollout_timer.snapshot()
            for client in superseded:
                # Closed in the background: an unresponsive stale tab must not
                # hold up the operator's new one.
                task = asyncio.create_task(
                    client.close(code=SUPERSEDED_CLOSE_CODE, message=b"superseded")
                )
                closing.add(task)
                task.add_done_callback(closing.discard)
            try:
                await ws.send_json(
                    {
                        "type": "config",
                        "cameras": self.cameras,
                        "overlay_camera": self.overlay.camera,
                        "overlay_enabled": overlay_enabled,
                        "wait_for_start": self._wait_for_start,
                        "paused": paused,
                        "started": started,
                        "rollout_time": rollout_time,
                        "inference_controls": inference_controls,
                        "ik_relaxation": ik_relaxation,
                        "ik_solver": ik_solver,
                        "gripper_force_limits": gripper_force_limits,
                        "inference_controls_revision": inference_controls_revision,
                        "video_recording_enabled": self.video_recording_config[
                            "enabled"
                        ],
                        "video_recording": self._video_recording,
                        "episode_recording_enabled": self.episode_recording_config[
                            "enabled"
                        ],
                        "episode_recording": self._episode_recording,
                        "episode_frames": self._episode_frames,
                        "model_browser_enabled": self.model_browser is not None,
                        "checkpoint": checkpoint,
                    }
                )
                async for message in ws:
                    if message.type.name != "TEXT":
                        continue
                    try:
                        command = json.loads(message.data)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(command, Mapping):
                        continue
                    if command.get("stop") is True:
                        with self._lock:
                            self._rollout_timer.pause()
                        self._quit_requested.set()
                    if command.get("start") is True:
                        self.request_start()
                    if command.get("restart") is True:
                        self.request_restart()
                    if command.get("reconnect_cameras") is True:
                        self.request_camera_reconnect()
                    if command.get("record_video") is True:
                        self.request_video_recording()
                    if command.get("record_episode") == "start":
                        self.request_episode_action("start")
                    if "save_episode" in command:
                        self.request_episode_action("save", command["save_episode"])
                    if command.get("discard_episode") is True:
                        self.request_episode_action("discard")
                    if "select_model" in command:
                        self.request_model_selection(command["select_model"])
                    if type(command.get("paused")) is bool:
                        self.request_pause(command["paused"])
                    if "ik_solver" in command:
                        self.request_ik_solver(command["ik_solver"])
                    if "ik_relaxation" in command:
                        self.request_ik_relaxation(command["ik_relaxation"])
                    if "gripper_force" in command:
                        self.request_gripper_force(command["gripper_force"])
                    overrides = command.get(
                        "inference_overrides", command.get("inference_override")
                    )
                    if isinstance(overrides, Mapping):
                        self.request_inference_overrides(overrides)
                    decision = command.get("velocity_action")
                    if decision in {"execute", "resample", "restart"}:
                        with self._lock:
                            if self._velocity_prompt is not None:
                                if decision == "restart":
                                    self._rollout_timer.reset()
                                self._velocity_decision = decision
                                self._velocity_decision_ready.set()
                    if type(command.get("overlay")) is bool:
                        with self._lock:
                            self._overlay_enabled = command["overlay"]
            finally:
                clients.discard(ws)
                with self._lock:
                    self._clients = len(clients)
            return ws

        @web.middleware
        async def loopback_only(request, handler):
            if request.remote not in {"127.0.0.1", "::1"}:
                raise web.HTTPForbidden()
            return await handler(request)

        app = web.Application(middlewares=[loopback_only])
        app.router.add_get("/", index)
        app.router.add_get("/{name:app.js|style.css}", asset)
        app.router.add_get("/ws", websocket)
        if self.model_browser is not None:

            async def checkpoints(request):
                try:
                    listing = self.model_browser.list_directory(
                        request.query.get("path", "."),
                        request.query.get("query", ""),
                    )
                except ValueError as error:
                    raise web.HTTPBadRequest(text=str(error)) from error
                return web.json_response(listing)

            app.router.add_get("/api/checkpoints", checkpoints)
        if self.video_recording_config["enabled"]:

            async def videos(_request):
                return web.json_response(
                    list_rollout_videos(self.video_recording_config["directory"])
                )

            async def video(request):
                try:
                    path = rollout_video_path(
                        self.video_recording_config["directory"],
                        request.match_info["video"],
                    )
                except FileNotFoundError as error:
                    raise web.HTTPNotFound(text=str(error)) from error
                return web.FileResponse(path, headers={"Cache-Control": "no-cache"})

            app.router.add_get("/api/videos", videos)
            app.router.add_get("/api/videos/{video}", video)
        if self.episode_recording_config["enabled"]:

            async def episodes(_request):
                # Metadata only: episode files are gigabytes and are read with
                # h5py on the station, not streamed to the browser.
                return web.json_response(
                    list_rollout_episodes(self.episode_recording_config["directory"])
                )

            app.router.add_get("/api/episodes", episodes)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, self.config["host"], self.config["port"]).start()
        self._ready.set()
        next_frame = 0.0
        # One in-flight send per browser. A viewer that cannot keep up loses
        # frames; it never queues them, because a queue grows for as long as the
        # rollout runs and eventually stalls the tab that the operator stops with.
        sending: dict[web.WebSocketResponse, asyncio.Task] = {}
        try:
            while not self._shutdown.is_set():
                now = time.monotonic()
                if clients and now >= next_frame:
                    snapshot = self._snapshot()
                    names, images = [], []
                    for name, frame in snapshot["frames"].items():
                        try:
                            if (
                                name == self.overlay.camera
                                and snapshot["overlay_enabled"]
                                and snapshot["plan"] is not None
                            ):
                                frame = self.overlay.draw(frame, snapshot["plan"])
                            payload = _jpeg_bytes(
                                frame,
                                self.config["width"],
                                self.config["jpeg_quality"],
                            )
                        except (RuntimeError, ValueError):
                            continue
                        names.append(name)
                        images.append(payload)
                    message = {
                        "type": "frame",
                        "images": names,
                        "status": snapshot["status"],
                        "overlay_enabled": snapshot["overlay_enabled"],
                        "overlay_status": snapshot["overlay_status"],
                        "age_ms": round(max(0.0, now - snapshot["updated_at"]) * 1000),
                        "paused": snapshot["paused"],
                        "started": snapshot["started"],
                        "rollout_time": snapshot["rollout_time"],
                        "inference_controls": snapshot["inference_controls"],
                        "ik_relaxation": snapshot["ik_relaxation"],
                        "ik_solver": snapshot["ik_solver"],
                        "gripper_force_limits": snapshot["gripper_force_limits"],
                        "inference_controls_revision": snapshot[
                            "inference_controls_revision"
                        ],
                        "inference": snapshot["inference"],
                        "video_recording": snapshot["video_recording"],
                        "video_last_saved": snapshot["video_last_saved"],
                        "episode_recording": snapshot["episode_recording"],
                        "episode_frames": snapshot["episode_frames"],
                        "episode_last_saved": snapshot["episode_last_saved"],
                        "checkpoint": snapshot["checkpoint"],
                        "velocity_prompt": snapshot["velocity_prompt"],
                    }
                    for client in tuple(sending):
                        if client not in clients:
                            sending.pop(client).cancel()
                    disconnected = set()
                    for client in tuple(clients):
                        task = sending.get(client)
                        if task is not None and not task.done():
                            # Still receiving the previous frame: skip this one.
                            continue
                        if task is not None:
                            try:
                                disconnected |= task.result()
                            except Exception:
                                # One failed socket must not end the view for the
                                # others, or leave the operator without a stop.
                                disconnected.add(client)
                        sending[client] = asyncio.create_task(
                            _broadcast_dashboard_message({client}, message, images)
                        )
                    if disconnected:
                        clients.difference_update(disconnected)
                        for client in disconnected:
                            task = sending.pop(client, None)
                            if task is not None:
                                task.cancel()
                        with self._lock:
                            self._clients = len(clients)
                    # Pace from the end of a send, not its start: measuring from
                    # `now` lets a frame that took longer than the period leave
                    # the deadline in the past, and the loop then sends as fast
                    # as the socket accepts instead of at camera_hz.
                    next_frame = time.monotonic() + 1 / self.config["camera_hz"]
                await asyncio.sleep(0.01)
        finally:
            for task in sending.values():
                task.cancel()
            for client in tuple(clients):
                await client.close()
            await runner.cleanup()
