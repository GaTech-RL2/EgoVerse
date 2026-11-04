"""Local browser dashboard for safe, camera-only YAM teleoperation previews.

The dashboard is deliberately a *view and input surface*, not another robot
controller.  It streams the existing RGB observations and places already
defined teleop key events onto a small queue consumed by ``collect_gello``.
It binds only to loopback, so it is intended for the station display (or an
explicit SSH tunnel), never as a network robot-control service.
"""

from __future__ import annotations

import asyncio
import base64
import json
import queue
import threading
import time
import webbrowser
from copy import deepcopy
from numbers import Integral
from pathlib import Path

import numpy as np


def list_demos(directory: str | Path) -> list[dict]:
    """List completed local HDF5 demos without loading image arrays."""
    import h5py

    result = []
    for path in sorted(Path(directory).glob("demo_*.hdf5")):
        try:
            demo_id = int(path.stem.removeprefix("demo_"))
            with h5py.File(path, "r") as handle:
                if bool(handle.attrs.get("complete", False)):
                    result.append({"id": demo_id, "frames": len(handle["action"])})
        except (OSError, ValueError, KeyError):
            continue
    result.sort(key=lambda demo: demo["id"])
    return result


def delete_demo(directory: str | Path, demo_id: int) -> dict:
    """Delete one completed local HDF5 demo selected by its numeric ID."""
    import h5py

    if isinstance(demo_id, bool) or not isinstance(demo_id, Integral) or demo_id < 0:
        raise ValueError("Demo ID must be a nonnegative integer")
    path = Path(directory) / f"demo_{demo_id}.hdf5"
    if not path.is_file():
        raise FileNotFoundError(f"demo_{demo_id}.hdf5 does not exist")
    with h5py.File(path, "r") as handle:
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError("Only completed demos can be deleted")
    path.unlink()
    return {"deleted": demo_id}


def demo_frame(
    directory: str | Path, demo_id: int, index: int, width: int, quality: int
):
    """Read one stored RGB HDF5 observation frame for browser-only review."""
    import cv2
    import h5py

    if (
        not isinstance(demo_id, Integral)
        or not isinstance(index, Integral)
        or demo_id < 0
    ):
        raise ValueError("Demo and frame indices must be nonnegative integers")
    path = Path(directory) / f"demo_{demo_id}.hdf5"
    with h5py.File(path, "r") as handle:
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError("Demo is incomplete")
        frames = len(handle["action"])
        if not 0 <= index < frames:
            raise ValueError("Frame index is outside this demo")
        images = {}
        for name, dataset in handle["observations/images"].items():
            # HDF5 demos intentionally store RGB, while cv2's encoder expects BGR.
            image = cv2.cvtColor(dataset[index], cv2.COLOR_RGB2BGR)
            images[name] = _jpeg_data_url(image, width, quality)
    return {"id": demo_id, "index": index, "frames": frames, "images": images}


DEFAULTS = {
    "enabled": True,
    "mode": "dashboard",
    "width": 640,
    "host": "127.0.0.1",
    "port": 8080,
    "open_browser": True,
    "camera_hz": 12,
    "jpeg_quality": 80,
}
STATIC = Path(__file__).with_name("teleop_dashboard_static")


def validate_episode_length(value: object) -> int | None:
    """A null limit means manual save only; otherwise require whole frames."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("recording.episode_length must be a positive integer or null")
    return int(value)


def validate_dashboard_config(config: dict | None) -> dict:
    """Return a complete, local-only dashboard configuration."""
    config = {} if config is None else dict(config)
    unknown = config.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown preview option(s): {', '.join(sorted(unknown))}")
    result = {**deepcopy(DEFAULTS), **config}
    for name in ("enabled", "open_browser"):
        if type(result[name]) is not bool:
            raise ValueError(f"preview.{name} must be a boolean")
    if result["mode"] != "dashboard":
        raise ValueError("preview.mode must be `dashboard`")
    if result["host"] not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("The teleop dashboard must bind only to loopback")
    for name, lower, upper in (
        ("width", 160, 1280),
        ("port", 1024, 65535),
        ("camera_hz", 1, 30),
        ("jpeg_quality", 1, 100),
    ):
        value = result[name]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"preview.{name} must be an integer in [{lower}, {upper}]")
    return result


def _jpeg_data_url(frame: np.ndarray, width: int, quality: int) -> str:
    """Encode an RGB frame for an ``<img>`` element without touching Qt."""
    import cv2

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Camera frame must be an HxWx3 RGB image")
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
    # The station RealSense stream is BGR8.  OpenCV's JPEG encoder expects BGR
    # too, and a browser decodes the resulting JPEG as the correct RGB image.
    # Converting beforehand swaps red and blue in the browser.
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Could not JPEG-encode camera frame")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


class TeleopDashboard:
    """Browser preview and existing-key bridge for three-camera GELLO teleop."""

    def __init__(
        self,
        cameras,
        keys: dict[str, str],
        recording_directory=None,
        gripper_force_limits: dict[str, float] | None = None,
        **config,
    ) -> None:
        self.config = validate_dashboard_config(config)
        self.cameras = tuple(cameras)
        if not self.cameras:
            raise ValueError("The teleop dashboard needs at least one camera")
        self.key_bindings = {
            str(action): str(key).lower() for action, key in keys.items()
        }
        self.keys = set(self.key_bindings.values())
        self.gripper_force_limits = {
            str(arm): float(force)
            for arm, force in (gripper_force_limits or {}).items()
        }
        self.recording_directory = (
            None if recording_directory is None else Path(recording_directory)
        )
        if not self.keys or any(len(key) != 1 for key in self.keys):
            raise ValueError("Dashboard controls must be single-character keys")
        self._keys: queue.Queue[object] = queue.Queue(maxsize=8)
        self._lock = threading.Lock()
        self._frames: dict[str, np.ndarray] = {}
        self._recording = False
        self._status = "Starting"
        self._episode_id: int | None = None
        self._episode_state = "next"
        self._episode_length: int | None = None
        self._recorded_frames = 0
        self._recording_rate_hz = 30.0
        self._updated_at = 0.0
        self._clients = 0
        self._camera_reconnect_requested = threading.Event()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run, name="yam_teleop_dashboard", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=5):
            self.close()
            raise RuntimeError("Timed out starting local teleop dashboard")
        if self._error is not None:
            self.close()
            raise RuntimeError(f"Could not start local teleop dashboard: {self._error}")
        host = self.config["host"]
        url_host = f"[{host}]" if ":" in host else host
        self.url = f"http://{url_host}:{self.config['port']}"
        print(f"Teleop dashboard: {self.url}")
        if self.config["open_browser"]:
            threading.Thread(
                target=webbrowser.open, args=(self.url,), daemon=True
            ).start()

    def update(self, obs, recording=False):
        """Publish the newest camera frames and return one requested key, if any."""
        with self._lock:
            self._recording = bool(recording)
            if self._clients:
                self._frames = {
                    name: np.ascontiguousarray(frame).copy()
                    for name in self.cameras
                    if (frame := obs.get(name)) is not None
                }
                self._updated_at = time.monotonic()
        try:
            return self._keys.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3)

    def set_status(self, status: str) -> None:
        """Set a control-loop status string without issuing a robot command."""
        with self._lock:
            self._status = str(status)

    def set_gripper_force_limits(self, limits: dict[str, float]) -> None:
        with self._lock:
            for arm, value in limits.items():
                if arm in self.gripper_force_limits:
                    self.gripper_force_limits[arm] = float(value)

    def set_episode(self, episode_id: int, state: str = "next") -> None:
        """Publish the collector-owned episode ID and its recording state."""
        if (
            isinstance(episode_id, bool)
            or not isinstance(episode_id, Integral)
            or episode_id < 0
        ):
            raise ValueError("Episode ID must be a nonnegative integer")
        if state not in {"next", "queued", "recording"}:
            raise ValueError("Episode state must be next, queued, or recording")
        with self._lock:
            self._episode_id = int(episode_id)
            self._episode_state = state

    def request_camera_reconnect(self) -> None:
        """Queue one RGB-only recovery request for the collection loop."""
        with self._lock:
            self._status = "Camera reconnect requested — followers will disarm"
        self._camera_reconnect_requested.set()

    def set_episode_length(
        self, episode_length: int | None, recorded_frames: int, rate_hz: float
    ) -> None:
        """Publish collector-confirmed limits and progress, never command a robot."""
        episode_length = validate_episode_length(episode_length)
        with self._lock:
            self._episode_length = episode_length
            self._recorded_frames = int(recorded_frames)
            self._recording_rate_hz = float(rate_hz)

    def take_camera_reconnect_request(self) -> bool:
        """Return one explicit browser recovery request to the collection loop."""
        if not self._camera_reconnect_requested.is_set():
            return False
        self._camera_reconnect_requested.clear()
        return True

    def _enqueue_key(self, key: object) -> None:
        if not isinstance(key, str) or key.lower() not in self.keys:
            return
        try:
            self._keys.put_nowait(key.lower())
        except queue.Full:
            pass

    def _enqueue_episode(self, episode: object) -> None:
        if (
            isinstance(episode, bool)
            or not isinstance(episode, Integral)
            or episode < 0
        ):
            return
        try:
            self._keys.put_nowait({"episode": int(episode)})
        except queue.Full:
            pass

    def _enqueue_gripper_force(self, request: object) -> None:
        if not isinstance(request, dict):
            return
        arm = request.get("arm")
        value = request.get("value")
        if arm not in self.gripper_force_limits:
            return
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if not np.isfinite(value) or not 1 <= value <= 50:
            return
        try:
            self._keys.put_nowait({"gripper_force": {"arm": arm, "value": value}})
        except queue.Full:
            pass

    def _enqueue_episode_length(self, value: object) -> None:
        try:
            episode_length = validate_episode_length(value)
        except ValueError:
            return
        try:
            self._keys.put_nowait({"episode_length": episode_length})
        except queue.Full:
            pass

    def _snapshot(self) -> dict:
        with self._lock:
            return {
                "frames": self._frames.copy(),
                "recording": self._recording,
                "status": self._status,
                "episode": self._episode_id,
                "episode_state": self._episode_state,
                "episode_length": self._episode_length,
                "recorded_frames": self._recorded_frames,
                "recording_rate_hz": self._recording_rate_hz,
                "gripper_force_limits": self.gripper_force_limits.copy(),
                "updated_at": self._updated_at,
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

        async def index(_request):
            return web.FileResponse(
                STATIC / "index.html", headers={"Cache-Control": "no-cache"}
            )

        async def asset(request):
            path = STATIC / request.match_info["name"]
            if not path.is_file() or path.parent != STATIC:
                raise web.HTTPNotFound()
            return web.FileResponse(path, headers={"Cache-Control": "no-cache"})

        async def websocket(request):
            ws = web.WebSocketResponse(heartbeat=10, max_msg_size=1024)
            await ws.prepare(request)
            clients.add(ws)
            with self._lock:
                self._clients = len(clients)
            await ws.send_json(
                {
                    "type": "config",
                    "cameras": self.cameras,
                    "keys": self.key_bindings,
                    "gripper_force_limits": self.gripper_force_limits,
                    "episode_length": self._snapshot()["episode_length"],
                }
            )
            try:
                async for message in ws:
                    if message.type.name == "TEXT":
                        try:
                            command = json.loads(message.data)
                            self._enqueue_key(command.get("key"))
                            self._enqueue_episode(command.get("episode"))
                            self._enqueue_gripper_force(command.get("gripper_force"))
                            if "episode_length" in command:
                                self._enqueue_episode_length(command["episode_length"])
                            if command.get("reconnect_cameras") is True:
                                self.request_camera_reconnect()
                        except (AttributeError, json.JSONDecodeError):
                            pass
            finally:
                clients.discard(ws)
                with self._lock:
                    self._clients = len(clients)
            return ws

        async def demos(_request):
            return web.json_response(list_demos(self.recording_directory))

        async def frame(request):
            try:
                return web.json_response(
                    demo_frame(
                        self.recording_directory,
                        int(request.match_info["demo"]),
                        int(request.query.get("index", "0")),
                        self.config["width"],
                        self.config["jpeg_quality"],
                    )
                )
            except (OSError, ValueError, KeyError) as error:
                return web.json_response({"error": str(error)}, status=400)

        async def delete(request):
            with self._lock:
                recording = self._recording
            if recording:
                return web.json_response(
                    {"error": "Stop the current recording before deleting a demo"},
                    status=409,
                )
            try:
                return web.json_response(
                    delete_demo(
                        self.recording_directory, int(request.match_info["demo"])
                    )
                )
            except FileNotFoundError as error:
                return web.json_response({"error": str(error)}, status=404)
            except (OSError, ValueError) as error:
                return web.json_response({"error": str(error)}, status=400)

        @web.middleware
        async def loopback_only(request, handler):
            expected = {"127.0.0.1", "::1"}
            if request.remote not in expected:
                raise web.HTTPForbidden()
            return await handler(request)

        app = web.Application(middlewares=[loopback_only])
        app.router.add_get("/", index)
        app.router.add_get("/{name:app.js|style.css}", asset)
        app.router.add_get("/ws", websocket)
        if self.recording_directory is not None:
            app.router.add_get("/api/demos", demos)
            app.router.add_get("/api/demos/{demo}/frame", frame)
            app.router.add_delete("/api/demos/{demo}", delete)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, self.config["host"], self.config["port"]).start()
        self._ready.set()
        next_frame = 0.0
        try:
            while not self._stop.wait(0.01):
                now = time.monotonic()
                if not clients or now < next_frame:
                    await asyncio.sleep(0)
                    continue
                snapshot = self._snapshot()
                images = {}
                for name, frame in snapshot["frames"].items():
                    try:
                        images[name] = _jpeg_data_url(
                            frame, self.config["width"], self.config["jpeg_quality"]
                        )
                    except (RuntimeError, ValueError):
                        continue
                message = {
                    "type": "frame",
                    "images": images,
                    "recording": snapshot["recording"],
                    "status": snapshot["status"],
                    "episode": snapshot["episode"],
                    "episode_state": snapshot["episode_state"],
                    "episode_length": snapshot["episode_length"],
                    "recorded_frames": snapshot["recorded_frames"],
                    "recording_rate_hz": snapshot["recording_rate_hz"],
                    "gripper_force_limits": snapshot["gripper_force_limits"],
                    "age_ms": round(max(0.0, now - snapshot["updated_at"]) * 1000),
                }
                for client in tuple(clients):
                    if not client.closed:
                        await client.send_json(message)
                next_frame = now + 1 / self.config["camera_hz"]
        finally:
            for client in tuple(clients):
                await client.close()
            await runner.cleanup()
