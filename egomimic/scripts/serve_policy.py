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
"""

import argparse
import logging
import socket

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


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = _parse_args()

    logging.info("Loading policy from %s", args.checkpoint)
    try:
        model = ModelWrapper.load_from_checkpoint(args.checkpoint, weights_only=False)
    except ModuleNotFoundError as e:
        if e.name != "dinov2":
            raise
        # weights_only=False unpickles the full object graph, including any nested
        # DINOv2 ViT instance (Adapt3R3DEncoder). Its class lives in the `dinov2`
        # module, which torch.hub only makes importable as a side effect of calling
        # torch.hub.load() (it patches sys.path). Trigger that once, then retry.
        import torch
        logging.info("Checkpoint references DINOv2; loading it once to fix sys.path, then retrying")
        torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=False)
        model = ModelWrapper.load_from_checkpoint(args.checkpoint, weights_only=False)

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
    try:
        server.serve_forever()
    finally:
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    main()
