#!/usr/bin/env python3
"""Websocket inference server for Scale ALOHA HPT checkpoints.

Protocol compatibility:
- Matches OpenPI websocket client behavior used by trainmybot
  (`openpi_client.websocket_client_policy`).
- Sends one metadata frame immediately on connect (msgpack).
- Receives msgpack observations and returns msgpack responses with:
    {"actions": np.ndarray[T, 14], "server_timing": {...}}

Expected incoming observation payload:
{
  "state": np.ndarray shape (14,),  # concatenated joint positions [left7|right7]
  "images": {
      "cam_high": np.ndarray shape (3, 224, 224) or (224, 224, 3),
      "cam_low": ...,
      "cam_left_wrist": ...,
      "cam_right_wrist": ...,
  },
  "prompt": str,  # currently unused by HPT, accepted for compatibility
}
"""

from __future__ import annotations

import argparse
import asyncio
import http
import pickle
import time
import traceback
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np
import torch
import websockets
import websockets.asyncio.server as ws_server
from scipy.spatial.transform import Rotation as R

from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.robot.scale_aloha.scale_aloha_kinematics import (
    ScaleAlohaMinkKinematicsSolver,
)
from egomimic.utils.egomimicUtils import EXTRINSICS, interpolate_arr_euler


# ---------------------------------------------------------------------------
# Safe checkpoint loading — handles checkpoints that contain serialised
# mujoco.MjModel / MjData objects which break across MuJoCo versions.
# ---------------------------------------------------------------------------

class _MjStub:
    """Drop-in placeholder for any mujoco C-extension object."""
    def __init__(self, *a, **kw): pass
    def __setstate__(self, state): pass


class _SafeUnpickler(pickle.Unpickler):
    """Replaces mujoco classes with lightweight stubs during unpickling."""
    def find_class(self, module: str, name: str):
        if "mujoco" in module.lower():
            return _MjStub
        return super().find_class(module, name)


def _reinit_mujoco_solvers(obj, depth: int = 0, visited: set | None = None):
    """Walk object graph; call _init_mujoco_runtime() on any solver with stubs."""
    if visited is None:
        visited = set()
    oid = id(obj)
    if oid in visited or depth > 12:
        return
    visited.add(oid)
    if isinstance(obj, (torch.Tensor, _MjStub)):
        return

    from egomimic.robot.kinematics import MinkKinematicsSolver
    if isinstance(obj, MinkKinematicsSolver):
        needs_reinit = (
            not hasattr(obj, "model")
            or isinstance(getattr(obj, "model", None), _MjStub)
            or getattr(obj, "model", None) is None
        )
        if needs_reinit:
            print(f"  [safe-load] Reinitializing {type(obj).__name__} "
                  f"(model_path={getattr(obj, 'model_path', '?')})")
            obj._init_mujoco_runtime()
        return

    if hasattr(obj, "__dict__"):
        for v in obj.__dict__.values():
            _reinit_mujoco_solvers(v, depth + 1, visited)
    if isinstance(obj, dict):
        for v in obj.values():
            _reinit_mujoco_solvers(v, depth + 1, visited)
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _reinit_mujoco_solvers(v, depth + 1, visited)


def _safe_load_checkpoint(path: str, map_location: str = "cpu") -> ModelWrapper:
    """Load a Lightning checkpoint, transparently stubbing any MuJoCo objects."""
    _orig = torch.serialization._load

    def _patched(zf, ml, pm, pf="data.pkl", **kw):
        class _SP:
            Unpickler = _SafeUnpickler
            @staticmethod
            def load(f):
                return _SafeUnpickler(f).load()
        return _orig(zf, ml, _SP, pf, **kw)

    torch.serialization._load = _patched
    try:
        wrapper = ModelWrapper.load_from_checkpoint(
            path, weights_only=False, map_location=map_location,
        )
    finally:
        torch.serialization._load = _orig

    _reinit_mujoco_solvers(wrapper)
    return wrapper


def _pack_array(obj: Any):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj: dict):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _packb(x: Any) -> bytes:
    return msgpack.packb(x, default=_pack_array)


def _unpackb(x: bytes) -> Any:
    return msgpack.unpackb(x, object_hook=_unpack_array)


def _to_chw_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize image input to CHW uint8."""
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got {arr.ndim}")

    if arr.shape[0] == 3:  # CHW
        chw = arr
    elif arr.shape[2] == 3:  # HWC
        chw = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Expected CHW or HWC with 3 channels, got {arr.shape}")

    if chw.dtype != np.uint8:
        if np.issubdtype(chw.dtype, np.floating):
            if chw.max() <= 1.0:
                chw = np.clip(chw * 255.0, 0, 255).astype(np.uint8)
            else:
                chw = np.clip(chw, 0, 255).astype(np.uint8)
        else:
            chw = np.clip(chw, 0, 255).astype(np.uint8)
    return chw


@dataclass
class ServerConfig:
    checkpoint_path: str
    host: str = "0.0.0.0"
    port: int = 8000
    action_chunk_len: int = 50
    model_action_horizon: int = 100
    embodiment_name: str = "scale_aloha_bimanual"
    extrinsics_key: str = "scale_aloha"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ScaleAlohaHPTPolicyServer:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        print(f"Loading checkpoint: {cfg.checkpoint_path}")
        self.policy = _safe_load_checkpoint(cfg.checkpoint_path, map_location=cfg.device)
        self.policy.to(cfg.device)
        self.policy.eval()
        print("Checkpoint loaded successfully.")

        self.ik_solver = ScaleAlohaMinkKinematicsSolver()

        # Convention in egomimicUtils: T_base_from_cam (aka cam-to-base).
        ex = EXTRINSICS[cfg.extrinsics_key]
        self.T_base_from_cam_left = ex["left"]
        self.T_base_from_cam_right = ex["right"]

        self.last_left_joints = np.zeros(6, dtype=np.float64)
        self.last_right_joints = np.zeros(6, dtype=np.float64)

        self.metadata = {
            "server": "egoverse_hpt_scale_aloha",
            "embodiment": cfg.embodiment_name,
            "action_chunk_len": cfg.action_chunk_len,
            "model_action_horizon": cfg.model_action_horizon,
        }

    @staticmethod
    def _cam_xyz_to_base(xyz_cam: np.ndarray, T_base_from_cam: np.ndarray) -> np.ndarray:
        pts = np.concatenate([xyz_cam, np.ones((xyz_cam.shape[0], 1))], axis=1)
        return (T_base_from_cam @ pts.T).T[:, :3]

    def _hpt_forward(self, obs: dict[str, Any]) -> np.ndarray:
        """Run HPT forward pass and return cartesian chunk (H, 14)."""
        state = np.asarray(obs["state"], dtype=np.float32).reshape(14)
        images = obs["images"]

        front = _to_chw_rgb_uint8(images["cam_high"]).astype(np.float32) / 255.0
        low = _to_chw_rgb_uint8(images["cam_low"]).astype(np.float32) / 255.0
        left = _to_chw_rgb_uint8(images["cam_left_wrist"]).astype(np.float32) / 255.0
        right = _to_chw_rgb_uint8(images["cam_right_wrist"]).astype(np.float32) / 255.0

        # Approximate observation ee_pose from current joints.
        # This matches training representation shape requirements.
        obs_ee_pose = self._joints_to_cartesian(state[None, :]).astype(np.float32)[0]

        # process_batch_for_training expects action key to exist and be 3D after collate.
        dummy_actions = np.zeros((self.cfg.model_action_horizon, 14), dtype=np.float32)

        batch = {
            self.cfg.embodiment_name: {
                # Add batch dimension to mirror DataLoader output.
                "observations.images.front_img_1": torch.from_numpy(front)[None, ...],
                "observations.images.cam_low": torch.from_numpy(low)[None, ...],
                "observations.images.left_wrist_img": torch.from_numpy(left)[None, ...],
                "observations.images.right_wrist_img": torch.from_numpy(right)[None, ...],
                "observations.state.joint_positions": torch.from_numpy(state)[None, ...],
                "observations.state.ee_pose": torch.from_numpy(obs_ee_pose)[None, ...],
                "actions_cartesian": torch.from_numpy(dummy_actions)[None, ...],
            }
        }

        with torch.no_grad():
            processed = self.policy.model.process_batch_for_training(batch)
            pred = self.policy.model.forward_eval(processed)[
                f"{self.cfg.embodiment_name}_actions_cartesian"
            ]
        return pred.detach().cpu().numpy().squeeze(0).astype(np.float32)

    def _joints_to_cartesian(self, joints_14: np.ndarray) -> np.ndarray:
        """Map joint positions (N,14) -> camera-frame cartesian (N,14)."""
        # Lazy import to avoid circular dependency.
        from egomimic.robot.scale_aloha.scale_aloha_kinematics import ScaleAlohaDualArmFK

        if not hasattr(self, "_fk_solver"):
            self._fk_solver = ScaleAlohaDualArmFK()

        cart_base = self._fk_solver.fk_bimanual_batch(joints_14)

        left_xyz = cart_base[:, :3]
        left_ypr = cart_base[:, 3:6]
        left_grip = cart_base[:, 6:7]
        right_xyz = cart_base[:, 7:10]
        right_ypr = cart_base[:, 10:13]
        right_grip = cart_base[:, 13:14]

        T_cam_from_left = np.linalg.inv(self.T_base_from_cam_left)
        T_cam_from_right = np.linalg.inv(self.T_base_from_cam_right)
        left_xyz_cam = self._cam_xyz_to_base(left_xyz, T_cam_from_left)
        right_xyz_cam = self._cam_xyz_to_base(right_xyz, T_cam_from_right)

        return np.concatenate(
            [left_xyz_cam, left_ypr, left_grip, right_xyz_cam, right_ypr, right_grip],
            axis=1,
        )

    def _cartesian_to_joints(self, actions_cam: np.ndarray) -> np.ndarray:
        """Convert predicted cartesian chunk (H,14) to joint chunk (H,14)."""
        H = actions_cam.shape[0]
        out = np.zeros((H, 14), dtype=np.float32)

        for i in range(H):
            row = actions_cam[i]
            left = row[:7]
            right = row[7:14]

            # Convert xyz from camera frame -> base frame. Keep ypr as predicted.
            left_xyz_base = self._cam_xyz_to_base(left[None, :3], self.T_base_from_cam_left)[0]
            right_xyz_base = self._cam_xyz_to_base(right[None, :3], self.T_base_from_cam_right)[0]

            left_rot = R.from_euler("ZYX", left[3:6], degrees=False).as_matrix()
            right_rot = R.from_euler("ZYX", right[3:6], degrees=False).as_matrix()

            l_ik = self.ik_solver.ik(left_xyz_base, left_rot, self.last_left_joints)
            r_ik = self.ik_solver.ik(right_xyz_base, right_rot, self.last_right_joints)

            if l_ik is None or not np.all(np.isfinite(l_ik)):
                l_ik = self.last_left_joints.copy()
            if r_ik is None or not np.all(np.isfinite(r_ik)):
                r_ik = self.last_right_joints.copy()

            self.last_left_joints = l_ik.copy()
            self.last_right_joints = r_ik.copy()

            out[i, :6] = l_ik[:6]
            out[i, 6] = float(left[6])   # gripper passthrough
            out[i, 7:13] = r_ik[:6]
            out[i, 13] = float(right[6])  # gripper passthrough

        return out

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        start = time.monotonic()
        cart_chunk = self._hpt_forward(obs)

        # Model usually predicts 100 steps; client consumes 50-step chunks.
        if cart_chunk.shape[0] != self.cfg.action_chunk_len:
            left = interpolate_arr_euler(cart_chunk[:, :7][None, ...], self.cfg.action_chunk_len)[0]
            right = interpolate_arr_euler(cart_chunk[:, 7:14][None, ...], self.cfg.action_chunk_len)[0]
            cart_chunk = np.hstack([left, right]).astype(np.float32)

        action_chunk = self._cartesian_to_joints(cart_chunk)
        infer_ms = (time.monotonic() - start) * 1000.0

        return {
            "actions": action_chunk,
            "server_timing": {"infer_ms": infer_ms},
        }


async def _health_check(
    connection: ws_server.ServerConnection, request: ws_server.Request
) -> ws_server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def _handler(
    websocket: ws_server.ServerConnection,
    policy_server: ScaleAlohaHPTPolicyServer,
):
    await websocket.send(_packb(policy_server.metadata))
    prev_total_time = None

    while True:
        try:
            start = time.monotonic()
            obs = _unpackb(await websocket.recv())
            result = policy_server.infer(obs)
            if prev_total_time is not None:
                result.setdefault("server_timing", {})["prev_total_ms"] = prev_total_time * 1000
            await websocket.send(_packb(result))
            prev_total_time = time.monotonic() - start
        except websockets.ConnectionClosed:
            break
        except Exception:
            await websocket.send(traceback.format_exc())
            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="Internal server error. Traceback included in previous frame.",
            )
            raise


async def _run_server(cfg: ServerConfig):
    policy_server = ScaleAlohaHPTPolicyServer(cfg)
    async with ws_server.serve(
        lambda ws: _handler(ws, policy_server),
        cfg.host,
        cfg.port,
        compression=None,
        max_size=None,
        process_request=_health_check,
    ) as server:
        await server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Serve EgoVerse HPT as OpenPI-compatible websocket policy")
    parser.add_argument("--checkpoint", required=True, help="Path to EgoVerse lightning checkpoint")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--action-chunk-len", type=int, default=50, help="Returned chunk length for client")
    parser.add_argument("--model-action-horizon", type=int, default=100, help="Dummy horizon for model batch")
    parser.add_argument("--extrinsics-key", default="scale_aloha", help="EXTRINSICS key from egomimicUtils")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    cfg = ServerConfig(
        checkpoint_path=args.checkpoint,
        host=args.host,
        port=args.port,
        action_chunk_len=args.action_chunk_len,
        model_action_horizon=args.model_action_horizon,
        extrinsics_key=args.extrinsics_key,
        device=args.device,
    )
    asyncio.run(_run_server(cfg))


if __name__ == "__main__":
    main()

