#!/usr/bin/env python3
"""
Serve an EgoVerse policy over WebSocket.

Usage:
    python egomimic/scripts/serve_policy.py --checkpoint path/to/last.ckpt --port 8000

Example:
    /coc/flash7/zhenyang/EgoVerse/emimic/bin/python egomimic/scripts/serve_policy.py \
  --checkpoint logs/RBY_test/test_2026-02-27_11-39-37/checkpoints/last.ckpt \
  --port 8000

Clients send observation dicts via msgpack; server returns action dicts.
See egomimic/serving/egoverse_policy.py for observation schema per embodiment.

Checkpoint-specific loading notes (the Lightning checkpoint pickles the WHOLE
model object graph, so every class it references must be importable):
  * Adapt3R (DINOv2 backbone): the `dinov2` package only becomes importable as a
    side effect of `torch.hub.load(...)`; we trigger that once on demand.
  * NVS-3D "snap" encoders: the frozen backbone's class lives in an external
    asset directory (`model.py` next to the `.pt` weights), registered as the
    `nvs3d_model` module. Pass `--nvs3d-dir` (or `export NVS3D_DIR=...`) — the
    directory only needs `model.py` for serving; the weights are in the ckpt.
"""

import argparse
import logging
import os
import signal
import socket
import sys

from egomimic.models.denoising_policy import DenoisingPolicy
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.serving.egoverse_policy import EgoVersePolicy
from egomimic.serving.websocket_policy_server import WebsocketPolicyServer


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Serve an EgoVerse policy over WebSocket.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--nvs3d-dir",
        type=str,
        default=os.environ.get("NVS3D_DIR"),
        help="NVS-3D asset dir (model.py [+ .pt]) for snap/NVS3DEncoder "
             "checkpoints. Defaults to $NVS3D_DIR; ignored by other checkpoints.",
    )
    # --- opt-in input/output recording (egomimic/serving/input_recorder.py) ---
    parser.add_argument(
        "--save-inputs-dir",
        type=str,
        default=None,
        help="Record every request's obs (as received) + returned actions under "
             "DIR/<ckpt_tag>_<port>_<ts>/. Off by default (no serving cost).",
    )
    parser.add_argument(
        "--save-inputs-every",
        type=int,
        default=1,
        help="Record every Nth inference (thinning).",
    )
    parser.add_argument(
        "--save-inputs-max-gb",
        type=float,
        default=None,
        help="Stop recording (keep serving) once the session exceeds this size.",
    )
    return parser.parse_args()


def _register_nvs3d(asset_dir: str) -> None:
    """Pre-register the `nvs3d_model` module so NVS-3D encoders unpickle."""
    model_py = os.path.join(asset_dir, "model.py")
    if not os.path.isfile(model_py):
        raise FileNotFoundError(
            f"--nvs3d-dir {asset_dir!r} has no model.py (need the NVS-3D asset dir)")
    # custom_encoders reads NVS3D_DIR at import time for its fallback path.
    os.environ.setdefault("NVS3D_DIR", asset_dir)
    from egomimic.models.custom_encoders import _load_nvs3d_module
    _load_nvs3d_module(asset_dir)
    logging.info("Registered nvs3d_model from %s", model_py)


def _load_model(args):
    """`ModelWrapper.load_from_checkpoint` with the on-demand import fixes."""
    if args.nvs3d_dir:
        _register_nvs3d(args.nvs3d_dir)
    dinov2_retried = False
    while True:
        try:
            return ModelWrapper.load_from_checkpoint(args.checkpoint, weights_only=False)
        except ModuleNotFoundError as e:
            if e.name == "dinov2" and not dinov2_retried:
                # weights_only=False unpickles the full object graph, including any
                # nested DINOv2 ViT instance (Adapt3R3DEncoder). Its class lives in
                # the `dinov2` module, which torch.hub only makes importable as a
                # side effect of calling torch.hub.load() (it patches sys.path).
                import torch
                logging.info("Checkpoint references DINOv2; loading it once to fix "
                             "sys.path, then retrying")
                torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=False)
                dinov2_retried = True
                continue
            if e.name == "nvs3d_model":
                raise ModuleNotFoundError(
                    "This checkpoint embeds an NVS-3D ('snap') encoder whose class "
                    "lives in an external asset dir. Pass --nvs3d-dir DIR (or export "
                    "NVS3D_DIR=DIR) where DIR contains Dennis's model.py "
                    "(+ 0981at.pt)."
                ) from e
            raise


def _install_shutdown_handlers(recorder) -> None:
    """SIGTERM/SIGINT -> flush the recorder, then exit.

    asyncio's default SIGINT handling can leave the websockets server hanging
    with an in-flight inference, and SIGTERM would skip `finally` entirely, so
    the recorder's last chunk + meta.json would be lost. A second signal exits
    immediately.
    """
    state = {"shutting_down": False}

    def _handler(signum, _frame):
        if state["shutting_down"]:
            os._exit(1)
        state["shutting_down"] = True
        logging.info("Signal %d: shutting down", signum)
        try:
            if recorder is not None:
                recorder.close()
        finally:
            logging.shutdown()
            os._exit(0)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = _parse_args()

    logging.info("Loading policy from %s", args.checkpoint)
    model = _load_model(args)

    if getattr(model.model, "diffusion", False):
        for head in model.model.nets["policy"].heads.values():
            if isinstance(head, DenoisingPolicy):
                head.num_inference_steps = 10
        logging.info("Set diffusion num_inference_steps=10")

    policy = EgoVersePolicy(model)
    metadata = policy.metadata

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Policy server: %s (%s), embodiment=%s", hostname, local_ip, metadata["embodiment"])
    logging.info("Listening on %s:%d", args.host, args.port)

    recorder = None
    if args.save_inputs_dir:
        from egomimic.serving.input_recorder import InputRecorder, make_session_dir
        session_dir = make_session_dir(args.save_inputs_dir, args.checkpoint, args.port)
        recorder = InputRecorder(
            session_dir,
            {"checkpoint": args.checkpoint, "port": args.port, "host": args.host,
             "server_metadata": {k: v for k, v in metadata.items()
                                 if isinstance(v, (str, int, float, list, dict))}},
            every_n=args.save_inputs_every,
            max_gb=args.save_inputs_max_gb,
        )
        logging.info("Recording inputs -> %s (every %d, max_gb=%s)",
                     session_dir, args.save_inputs_every, args.save_inputs_max_gb)

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
        recorder=recorder,
    )
    _install_shutdown_handlers(recorder)
    try:
        server.serve_forever()
    finally:
        if recorder is not None:
            recorder.close()
        sys.stdout.flush()


if __name__ == "__main__":
    main()
