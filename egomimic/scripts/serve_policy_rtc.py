#!/usr/bin/env python3
"""Serve an EgoVerse FM policy with Real-Time Chunking support.

Identical to serve_policy.py (same args, same recorder, same shutdown), except
the policy wrapper is RTCEgoVersePolicy: requests carrying rtc_prev_actions /
rtc_freeze_steps / rtc_soft_horizon / rtc_soft_power get inpainting-constrained
chunks; plain requests behave exactly like the normal server, so this server is
a drop-in superset.

    emimic/bin/python egomimic/scripts/serve_policy_rtc.py \
        --checkpoint checkpoints/abl0921/V0.ckpt --port 8100
    # or: RTC=1 bash serve_abl_0921.sh V0
"""
import logging
import os
import socket
import sys

from egomimic.models.denoising_policy import DenoisingPolicy
from egomimic.scripts.serve_policy import (
    _install_shutdown_handlers,
    _load_model,
    _parse_args,
)
from egomimic.serving.rtc_policy import RTCEgoVersePolicy
from egomimic.serving.websocket_policy_server import WebsocketPolicyServer


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = _parse_args()

    logging.info("Loading policy (RTC-capable) from %s", args.checkpoint)
    model = _load_model(args)

    if getattr(model.model, "diffusion", False) and args.num_inference_steps > 0:
        for head in model.model.nets["policy"].heads.values():
            if isinstance(head, DenoisingPolicy):
                head.num_inference_steps = int(args.num_inference_steps)
        logging.info("Set num_inference_steps=%d", args.num_inference_steps)

    policy = RTCEgoVersePolicy(model)
    metadata = dict(policy.metadata)
    metadata["checkpoint"] = os.path.abspath(args.checkpoint)
    metadata["num_inference_steps"] = (
        int(args.num_inference_steps) if args.num_inference_steps > 0 else None)
    metadata["rtc"] = True

    hostname = socket.gethostname()
    logging.info("RTC policy server: %s, embodiment=%s, listening on %s:%d",
                 hostname, metadata["embodiment"], args.host, args.port)

    recorder = None
    if args.save_inputs_dir:
        from egomimic.serving.input_recorder import InputRecorder, make_session_dir
        session_dir = make_session_dir(args.save_inputs_dir, args.checkpoint, args.port)
        recorder = InputRecorder(
            session_dir,
            {"checkpoint": args.checkpoint, "port": args.port, "host": args.host,
             "server_metadata": {k: v for k, v in metadata.items()
                                 if isinstance(v, (str, int, float, list, dict, bool))}},
            every_n=args.save_inputs_every,
            max_gb=args.save_inputs_max_gb,
        )
        logging.info("Recording inputs -> %s", session_dir)

    server = WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port,
        metadata=metadata, recorder=recorder,
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
