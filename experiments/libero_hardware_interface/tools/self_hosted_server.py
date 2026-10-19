"""Pinned TP8 DeepSeek service; lifecycle belongs to this OSMO group only."""

import base64
import hashlib
import importlib.metadata
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from astra_reversal.hardware_interface.replication import (
    DEEPSEEK_MODEL,
    DEEPSEEK_REVISION,
    VLLM_IMAGE,
)

STATE = {"phase": "starting", "model": DEEPSEEK_MODEL, "revision": DEEPSEEK_REVISION}
STOP = threading.Event()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


class Control(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path == "/artifacts":
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                for path in sorted(
                    Path(os.environ["HARDWARE_SERVER_ARTIFACTS"]).glob("*")
                ):
                    if path.is_file() and not path.is_symlink():
                        archive.add(path, arcname=path.name)
            raw = buffer.getvalue()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if self.path != "/identity":
            self.send_error(404)
            return
        raw = json.dumps(STATE).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        if (
            self.path != "/finish"
            or int(self.headers.get("Content-Length", "0")) > 1024
        ):
            self.send_error(400)
            return
        value = json.loads(
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
        )
        if value.get("workflow") != os.environ["HARDWARE_WORKFLOW"]:
            self.send_error(403)
            return
        STOP.set()
        self.send_response(200)
        self.end_headers()


def command(snapshot):
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(snapshot),
        "--served-model-name",
        DEEPSEEK_MODEL,
        "--tokenizer-mode",
        "deepseek_v41",
        "--tensor-parallel-size",
        "8",
        "--engram-config",
        '{"cpu_offload":true}',
        "--max-model-len",
        "1048576",
        "--max-num-batched-tokens",
        "4096",
        "--gpu-memory-utilization",
        "0.92",
        "--max-num-seqs",
        "8",
        "--reasoning-parser",
        "deepseek_v41",
        "--tool-call-parser",
        "deepseek_v41",
        "--enable-auto-tool-choice",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--speculative-config",
        '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}',
    ]


def serving_smoke():
    """One image/tool round trip, outside all measured actor trials."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (128, 128), (32, 96, 160)).save(buffer, format="PNG")
    tool = {
        "type": "function",
        "name": "report_frame",
        "description": "Report whether an image was supplied.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"image_supplied": {"type": "boolean"}},
            "required": ["image_supplied"],
            "additionalProperties": False,
        },
    }
    body = {
        "model": DEEPSEEK_MODEL,
        "instructions": "Transport check only. Call report_frame once with image_supplied=true. Do not answer in prose.",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Inspect the supplied image and call the tool.",
                    },
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,"
                        + base64.b64encode(buffer.getvalue()).decode(),
                    },
                ],
            }
        ],
        "tools": [tool],
        "parallel_tool_calls": False,
        "chat_template_kwargs": {
            "thinking": True,
            "reasoning_effort": 75,
            "drop_thinking": False,
        },
        "store": False,
        "truncation": "disabled",
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": 42,
    }
    records = []
    for turn in range(2):
        request = urllib.request.Request(
            "http://127.0.0.1:8000/v1/responses",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=180) as response:
            value = json.load(response)
        calls = [
            item
            for item in value.get("output", [])
            if item.get("type") == "function_call"
        ]
        if (
            len(calls) != 1
            or calls[0]["name"] != "report_frame"
            or json.loads(calls[0]["arguments"]) != {"image_supplied": True}
        ):
            raise RuntimeError("self_hosted_multimodal_tool_smoke_failed")
        records.append(
            {
                "turn": turn,
                "wall_seconds": time.monotonic() - started,
                "usage": value.get("usage"),
                "response_model": value.get("model"),
            }
        )
        body["input"].extend(value["output"])
        body["input"].append(
            {
                "type": "function_call_output",
                "call_id": calls[0]["call_id"],
                "output": "Image received. Repeat report_frame(image_supplied=true) once to verify tool history.",
            }
        )
    return {
        "passed": True,
        "scored_result": False,
        "model": DEEPSEEK_MODEL,
        "revision": DEEPSEEK_REVISION,
        "image_tool_and_history_roundtrip": True,
        "requests": records,
    }


def main():
    root = Path(os.environ["HARDWARE_SERVING_CACHE"])
    root.mkdir(parents=True, exist_ok=True)
    artifacts = Path(
        os.environ.get("HARDWARE_SERVER_ARTIFACTS", "/tmp/serving-artifacts")
    )
    artifacts.mkdir(parents=True, exist_ok=True)
    control = ThreadingHTTPServer(("0.0.0.0", 8001), Control)
    threading.Thread(target=control.serve_forever, daemon=True).start()
    process = None
    try:
        import torch
        from huggingface_hub import HfApi, snapshot_download

        version = importlib.metadata.version("vllm")
        if version.split("+")[0] != "0.31.0":
            raise RuntimeError("vllm_version_mismatch")
        if torch.cuda.device_count() != 8 or any(
            "H100" not in torch.cuda.get_device_name(i) for i in range(8)
        ):
            raise RuntimeError("eight_allocated_H100s_required")
        info = HfApi().model_info(
            DEEPSEEK_MODEL, revision=DEEPSEEK_REVISION, files_metadata=True
        )
        if info.sha != DEEPSEEK_REVISION:
            raise RuntimeError("checkpoint_revision_mismatch")
        expected_bytes = sum(
            f.size or 0 for f in info.siblings if f.rfilename.endswith(".safetensors")
        )
        cached = (
            sum(
                p.stat().st_size
                for p in (root / "huggingface").rglob("*")
                if p.is_file() and not p.is_symlink()
            )
            if (root / "huggingface").exists()
            else 0
        )
        disk = shutil.disk_usage(root)
        write(
            artifacts / "storage.json",
            {
                "cache": str(root),
                "free_bytes": disk.free,
                "expected_weight_bytes": expected_bytes,
                "existing_cache_bytes": cached,
            },
        )
        if disk.free < max(0, expected_bytes - cached) + 20 * 1024**3:
            raise RuntimeError("checkpoint_cache_insufficient_free_space")
        STATE.update(phase="downloading", image=VLLM_IMAGE, vllm_version=version)
        print(json.dumps(STATE), flush=True)
        snapshot = Path(
            snapshot_download(
                DEEPSEEK_MODEL,
                revision=DEEPSEEK_REVISION,
                cache_dir=str(root / "huggingface"),
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.model",
                    "tokenizer*",
                    "*.jinja",
                ],
                max_workers=8,
            )
        )
        index = snapshot / "model.safetensors.index.json"
        value = json.loads(index.read_text())
        expected = {f.rfilename: f for f in info.siblings}
        shards = []
        for name in sorted(set(value["weight_map"].values())):
            path = snapshot / name
            meta = expected[name]
            if path.stat().st_size != meta.size:
                raise RuntimeError("checkpoint_shard_size_mismatch")
            shards.append(
                {
                    "name": name,
                    "bytes": meta.size,
                    "expected_sha256": meta.lfs.sha256 if meta.lfs else None,
                }
            )
        args = command(snapshot)
        identity = {
            "phase": "loading",
            "workflow": os.environ["HARDWARE_WORKFLOW"],
            "model": DEEPSEEK_MODEL,
            "revision": DEEPSEEK_REVISION,
            "image": VLLM_IMAGE,
            "vllm_version": version,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpus": [
                {
                    "name": torch.cuda.get_device_name(i),
                    "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                }
                for i in range(8)
            ],
            "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
            "stored_weight_bytes": value["metadata"]["total_size"],
            "shards": shards,
            "snapshot": str(snapshot),
            "command": args,
            "weight_validation": "Pinned HF commit and content-addressed hub download; all shard sizes checked against pinned API metadata. Expected LFS SHA-256 recorded, not independently rehashed here.",
        }
        STATE.update(identity)
        write(artifacts / "identity.json", identity)
        log = (artifacts / "vllm.log").open("w")
        process = subprocess.Popen(
            args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        started = time.monotonic()
        last_metrics = 0.0
        while not STOP.wait(5):
            if process.poll() is not None:
                # This log contains no mounted API or storage credentials.
                print((artifacts / "vllm.log").read_text()[-16000:], flush=True)
                raise RuntimeError("vllm_server_exited_" + str(process.returncode))
            try:
                with urllib.request.urlopen(
                    "http://127.0.0.1:8000/health", timeout=3
                ) as response:
                    healthy = response.status == 200
                if healthy and STATE["phase"] != "ready":
                    STATE.update(
                        phase="ready", engine_startup_seconds=time.monotonic() - started
                    )
                    write(artifacts / "identity.json", STATE)
                    print(
                        json.dumps(
                            {
                                "phase": "ready",
                                "engine_startup_seconds": STATE[
                                    "engine_startup_seconds"
                                ],
                            }
                        ),
                        flush=True,
                    )
                    if os.environ.get("HARDWARE_SERVING_SMOKE_ONLY") == "1":
                        receipt = serving_smoke()
                        write(artifacts / "serving-smoke.json", receipt)
                        print(
                            json.dumps(
                                {
                                    "serving_smoke_passed": True,
                                    "requests": receipt["requests"],
                                }
                            ),
                            flush=True,
                        )
                        STOP.set()
                if healthy and time.monotonic() - last_metrics >= 60:
                    with urllib.request.urlopen(
                        "http://127.0.0.1:8000/metrics", timeout=5
                    ) as response:
                        with (artifacts / "metrics.jsonl").open("a") as stream:
                            stream.write(
                                json.dumps(
                                    {
                                        "unix": time.time(),
                                        "prometheus": response.read().decode(),
                                    }
                                )
                                + "\n"
                            )
                    last_metrics = time.monotonic()
            except OSError:
                pass
    except BaseException as error:
        STATE.update(
            phase="failed",
            error_class=type(error).__name__,
            reason=str(error)
            if isinstance(error, RuntimeError)
            else type(error).__name__,
        )
        write(artifacts / "failure.json", STATE)
        print(json.dumps(STATE), flush=True)
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        control.shutdown()


if __name__ == "__main__":
    main()
