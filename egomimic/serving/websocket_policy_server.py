"""
WebSocket policy server for EgoVerse.

Mirrors the openpi websocket policy protocol: msgpack serialization,
metadata on connect, infer/infer_batch loop. Health check at /healthz.
"""

import asyncio
import http
import logging
import time
import traceback

import msgpack_numpy
import websockets
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)

# Use msgpack_numpy for numpy array serialization (openpi-compatible)


def _pack(data):
    """Pack data for wire. Handles numpy arrays via msgpack_numpy."""
    return msgpack_numpy.packb(data)


def _unpack(data):
    """Unpack wire data. Handles numpy arrays via msgpack_numpy."""
    return msgpack_numpy.unpackb(data)


class WebsocketPolicyServer:
    """
    Serves a policy over WebSocket.

    Protocol:
    - On connect: server sends metadata (methods, embodiment, action specs)
    - Loop: client sends obs (msgpack) -> server returns action dict (msgpack)
    - Single obs -> infer; list of obs -> infer_batch
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        recorder=None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        # Opt-in input/output recording (serving/input_recorder.py). None =
        # the default = a single `is not None` check per request, nothing else.
        # NOTE: this file is overwritten by apply_skynet_snapshot.py --apply;
        # re-apply the three `_recorder` hooks afterwards (see memory /
        # POLICY_QUICKSTART.md).
        self._recorder = recorder
        self._conn_counter = 0
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            port_info = self._port if self._port is not None else "dynamic"
            logger.info("Policy server listening on %s:%s", self._host, port_info)
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        self._conn_counter += 1
        conn_id = self._conn_counter
        if self._recorder is not None:
            self._recorder.on_connection(conn_id, "open", websocket.remote_address)
        metadata = dict(self._metadata)
        existing_methods = set(metadata.get("methods", []))
        existing_methods.update({"infer", "infer_batch"})
        metadata["methods"] = sorted(existing_methods)
        await websocket.send(_pack(metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = _unpack(await websocket.recv())

                infer_time = time.monotonic()
                if isinstance(obs, list):
                    action = self._policy.infer_batch(obs)
                else:
                    action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time
                if self._recorder is not None:
                    self._recorder.record(obs, action, conn_id=conn_id,
                                          infer_ms=infer_time * 1000)

                if isinstance(action, list):
                    for a in action:
                        if isinstance(a, dict):
                            a.setdefault("server_timing", {})["infer_ms"] = infer_time * 1000
                            if prev_total_time is not None:
                                a["server_timing"]["prev_total_ms"] = prev_total_time * 1000
                else:
                    action["server_timing"] = {"infer_ms": infer_time * 1000}
                    if prev_total_time is not None:
                        action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(_pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                if self._recorder is not None:
                    self._recorder.on_connection(conn_id, "close", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(
    connection: _server.ServerConnection,
    request: _server.Request,
) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
