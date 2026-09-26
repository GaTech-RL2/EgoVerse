"""Selected sealed-archive paper-direction replay; no GPU/model/physics calls."""

import base64
import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import features

from astra_reversal import frs_agent, frs_audit, frs_guide, frs_operators, records
from astra_reversal.action_adapter import ActionSpec

RETRIEVAL_SHA = "600e8a4c14a03b72537b71daaf6ac041471d9be53d493de15971684f1cec3756"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent / "worker_0_direction"
    retrieval_path = root / "retrieval_receipt.json"
    assert sha(retrieval_path) == RETRIEVAL_SHA
    receipt = json.loads(retrieval_path.read_text())
    assert np.__version__ == "1.26.4"
    assert PIL.__version__ == "12.3.0" and features.version("zlib") == "1.3"
    arrays_by_name = {}
    for row in receipt["arrays"]:
        path = root / row["file"]
        assert sha(path) == row["file_sha256"]
        array = np.load(path, allow_pickle=False)
        assert records.digest(array) == row["array_sha256"]
        arrays_by_name[row["file"]] = array

    def arrays(value):
        if isinstance(value, dict):
            if set(value) == {"array", "shape", "dtype", "sha256"}:
                array = arrays_by_name[value["array"]]
                assert list(array.shape) == value["shape"]
                assert str(array.dtype) == value["dtype"]
                assert records.digest(array) == value["sha256"]
                return array
            return {key: arrays(item) for key, item in value.items()}
        if isinstance(value, list):
            return [arrays(item) for item in value]
        return value

    events_path = root / "events_prefix.jsonl"
    providers_path = root / "provider_prefix.jsonl"
    assert sha(events_path) == receipt["event_prefix_sha256"]
    assert sha(providers_path) == receipt["provider_prefix_sha256"]
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    providers = [json.loads(line) for line in providers_path.read_text().splitlines()]
    index = receipt["request_index"]
    assert index == 204
    row = next(row for row in providers if row["request_index"] == index)
    decision = next(
        event
        for event in events
        if event["kind"] == "astra_decision" and event["request_index"] == index
    )
    generation = arrays(
        next(
            event
            for event in events
            if event["kind"] == "generation"
            and (event.get("proposal") or {}).get("request_index") == index
        )
    )
    start = next(
        event
        for event in events
        if event["kind"] == "rollout_start"
        and event["attempt_id"] == generation["attempt_id"]
    )
    entry = next(
        entry
        for entry in events[0]["entries"]
        if entry["episode_id"] == start["episode_id"]
    )
    request = frs_agent.direction_request(
        **{
            key: row[key]
            for key in ("episode_id", "attempt_id", "request_index", "observation_step")
        },
        target_task=entry["instruction"],
        external_image=generation["observation"]["observation/image"],
        guide_image=generation["guide"],
    )
    frs_audit.verify_provider_binding(
        request, decision, row, events[0]["protocol"]["astra"]
    )
    assert generation["proposal"] == decision["response"]
    signs = frs_audit.replay_guide(
        generation["observation"], generation["guide"], generation["guide_receipt"]
    )
    norm_path = (
        Path(frs_agent.__file__).parent / ".deps/reference/pi05_libero/norm_stats.json"
    )
    norm_sha = sha(norm_path)
    assert (
        norm_sha
        == events[0]["checkpoint"]["input_profile_assets"]["norm_stats"]["sha256"]
    )
    inventory = json.loads(
        (
            Path(frs_agent.__file__).parent
            / "checkpoints/openpi_libero_input_assets.json"
        ).read_text()
    )
    assert norm_sha == inventory["norm_stats"]["sha256"]
    stats = json.loads(norm_path.read_text())["norm_stats"]["actions"]
    adapter = frs_audit.normalization_adapter(
        ActionSpec(**start["action_spec"]), stats["q01"], stats["q99"]
    )
    replay = frs_audit.replay_generation(generation, entry=entry, adapter=adapter)
    payload = frs_agent.build_payload(
        request, row["requested_model"], sampling=row["sampling_settings"]
    )
    final_rollouts = [
        event["result"]
        for event in events
        if event["kind"] == "rollout_end"
        and event["result"]["method"] in ("astra_direction_direct", "astra_frs")
    ]
    assert len(final_rollouts) == 2
    panels = [("FRS initial raw observation (step 0)", generation["observation"])]
    for rollout in final_rollouts:
        snapshot = rollout["snapshots"][-1]
        panels.append(
            (
                f"{rollout['method']}: recorded final step {snapshot['step']}",
                arrays(snapshot["observation"]),
            )
        )
    figure, axes = plt.subplots(3, 2, figsize=(8, 12), constrained_layout=True)
    for row_index, (label, observation) in enumerate(panels):
        for column, camera in enumerate(
            ("observation/image", "observation/wrist_image")
        ):
            axes[row_index, column].imshow(observation[camera], interpolation="nearest")
            axes[row_index, column].set_title(label + "\n" + camera, fontsize=9)
            axes[row_index, column].axis("off")
    figure.savefig(root / "direction_final_raw_frames.png", dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    for axis, image, title in zip(
        axes,
        (generation["observation"]["observation/image"], generation["guide"]),
        ("Raw external (policy input)", "Calibrated guide (reasoner only)"),
        strict=True,
    ):
        axis.imshow(image, interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    figure.savefig(root / "direction_guide.png", dpi=150)
    plt.close(figure)
    safe = {
        "schema_version": "frs-selected-direction-verification-1.0",
        "status": "passed",
        "scope": "One existing astra_frs generation, strict provider/raw/guide/axis/action/noise boundary replay, and two final raw pairs. No velocity solve, model, provider, or physics rerun; not a complete task audit.",
        "episode_id": entry["episode_id"],
        "attempt_id": start["attempt_id"],
        "request_index": index,
        "step": generation["step"],
        "request_fingerprint": request["request_fingerprint"],
        "http_payload_sha256": hashlib.sha256(
            json.dumps(payload, allow_nan=False).encode()
        ).hexdigest(),
        "strict_provider_binding_passed": True,
        "applied_decision_matches_provider": True,
        "guide_raster_and_projection_passed": True,
        "generation_replay_passed": True,
        "request_images": {
            name: hashlib.sha256(base64.b64decode(wire["data"])).hexdigest()
            for name, wire in request["inputs"].items()
        },
        "camera_to_controller_signs": signs,
        "proposal": {
            key: generation["proposal"][key]
            for key in ("fine", "coords", "motion_amount", "justification")
        },
        "reference": generation["reference"]["receipt"],
        "noise_transform": generation["noise_transform"]["receipt"],
        "reconstruction_metrics": generation["reconstruction"],
        "replay_counts": replay,
        "archived_final_frames": [
            {
                "method": rollout["method"],
                "episode_id": rollout["episode_id"],
                "final_step": rollout["snapshots"][-1]["step"],
                "recorded_success": rollout["success"],
                "raw_camera_sha256": {
                    camera: rollout["snapshots"][-1]["observation"][camera]["sha256"]
                    for camera in ("observation/image", "observation/wrist_image")
                },
            }
            for rollout in final_rollouts
        ],
        "archive_sha256": receipt["archive_sha256"],
        "source_sha256": {
            Path(module.__file__).name: sha(Path(module.__file__))
            for module in (frs_agent, frs_audit, frs_guide, frs_operators)
        },
        "input_sha256": {
            "retrieval_receipt.json": sha(retrieval_path),
            "events_prefix.jsonl": sha(events_path),
            "provider_prefix.jsonl": sha(providers_path),
            "norm_stats.json": norm_sha,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "pillow_zlib": features.version("zlib"),
        },
        "figure_sha256": {
            name: sha(root / name)
            for name in ("direction_final_raw_frames.png", "direction_guide.png")
        },
        "verification_helper_sha256": sha(Path(__file__)),
    }
    output = root / "selected_direction_verification.json"
    output.write_text(json.dumps(safe, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "request_index": index,
                "guide_signs": signs,
                "receipt_sha256": sha(output),
            }
        )
    )


if __name__ == "__main__":
    main()
