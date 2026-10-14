"""Continuously verify archived trials and publish separate LIBERO pilot reports."""

import base64
import hashlib
import html
import json
import os
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import yaml
from botocore.exceptions import ClientError

from astra_reversal.hardware_interface.analysis import analyze, audit
from astra_reversal.hardware_interface.common import strict_json, write_json
from astra_reversal.hardware_interface.reporting import export, power_worksheet

from .parallel_plan import design_hash

PREFIX = "experiments/libero-hardware-interface-20261008/"
RECORDS = ("outcome.json", "independent-replay.json", "trial.json", "events.jsonl")
INFRASTRUCTURE = {
    "MODEL_ERROR",
    "TOOL_ERROR",
    "PROTOCOL_DEVIATION",
    "RESET_FAILURE",
    "EVALUATOR_ERROR",
}


def source_for_owner(cohort, owner):
    """Pin each worker to either the original source or a declared reset fix."""
    amendment = cohort.get("source_amendments", {}).get(owner)
    if amendment is None:
        return cohort["plan"]["experiment_source_sha256"]
    if (
        amendment["base_source_commit"] != cohort["plan"]["experiment_source_commit"]
        or amendment["base_source_sha256"] != cohort["plan"]["experiment_source_sha256"]
        or sorted(amendment["changed_runtime_files"])
        != ["libero.py", "reset_validation.py"]
        or amendment["native_validation"]["status"] != "passed"
    ):
        raise ValueError("invalid_reset_source_amendment")
    return amendment["source_sha256"]


def verify_record(receipt, contents, expected, source_sha):
    files = {r["path"]: r["sha256"] for r in receipt["files"]}
    if len(files) != len(receipt["files"]):
        raise ValueError("duplicate_archive_path")
    for name in RECORDS:
        if hashlib.sha256(contents[name]).hexdigest() != files[name]:
            raise ValueError("archive_record_hash_mismatch")
    trial = strict_json(contents["trial.json"])
    if trial["source_sha256"] != source_sha:
        raise ValueError("different_experiment_source")
    if any(str(trial["row"].get(k)) != str(v) for k, v in expected.items()):
        raise ValueError("different_frozen_trial")
    row = strict_json(contents["outcome.json"])
    replay = strict_json(contents["independent-replay.json"])
    if any(
        row.get(k) != expected[k]
        for k in (
            "trial_id",
            "condition",
            "task_id",
            "init_state_index",
            "replicate",
            "env_seed",
            "init_state_hash",
        )
    ):
        raise ValueError("outcome_identity_mismatch")
    if (
        replay["status"] != "passed"
        or replay["success"] != row["success"]
        or replay["replayed_steps"] != row["sim_steps"]
        or replay["terminal_state_matched"] is not True
    ):
        raise ValueError("independent_replay_disagrees")
    row["independent_evaluation_passed"] = True
    row["experiment_source_sha256"] = trial["source_sha256"]
    return row


def read_object(client, key):
    try:
        return client.get_object(Bucket="rldb", Key=key)["Body"].read()
    except ClientError as error:
        if error.response["Error"]["Code"] in ("NoSuchKey", "404", "NotFound"):
            return None
        raise


def collect_trial(client, owner, expected, cohort, cache):
    trial_id = expected["trial_id"]
    directory = cache / cohort["name"] / owner / trial_id / "runs" / trial_id
    verified = directory.parent.parent / "verified.json"
    source_sha = source_for_owner(cohort, owner)
    if verified.exists():
        cached = strict_json(verified.read_bytes())
        if cached.get("experiment_source_sha256") != source_sha:
            raise ValueError("cached_experiment_source_changed")
        return cached
    prefix = PREFIX + owner + "/trials/" + trial_id + "/"
    receipt_raw = read_object(client, prefix + "archive-receipt.json")
    if receipt_raw is None:
        return None
    receipt = strict_json(receipt_raw)
    if receipt["workflow"] != owner or receipt["prefix"] != prefix:
        raise ValueError("archive_owner_mismatch")
    contents = {name: read_object(client, prefix + name) for name in RECORDS}
    if any(value is None for value in contents.values()):
        raise ValueError("published_archive_incomplete")
    row = verify_record(receipt, contents, expected, source_sha)
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        (directory / name).write_bytes(content)
    write_json(directory.parent.parent / "archive-receipt.json", receipt, replace=True)
    if audit(directory.parent)["status"] != "passed":
        raise ValueError("event_chain_audit_failed")
    row["archive_prefix"] = prefix
    row["archive_receipt_sha256"] = hashlib.sha256(receipt_raw).hexdigest()
    row["recorded_transport_errors"] = sorted(
        {
            event.get("reason")
            for event in (
                strict_json(line) for line in contents["events.jsonl"].splitlines()
            )
            if event.get("event") == "error"
            and str(event.get("reason", "")).startswith("provider_")
        }
    )
    write_json(verified, row)
    return row


def runtime_receipts(client, cohort):
    pending, differences, verified = [], {}, []
    for owner in cohort["owners"]:
        workflow = owner["workflow"]
        raw = read_object(
            client, PREFIX + workflow + "/final/pilot-preregistration.yaml"
        )
        if raw is None:
            pending.append(workflow)
            continue
        actual = yaml.safe_load(raw)
        errors = []
        if design_hash(actual) != cohort["plan"]["design_sha256"]:
            errors.append("design")
        for key, value in cohort["manifest"]["resolved"].items():
            if key == "adapter_sha256":
                value = source_for_owner(cohort, workflow)
            if actual["resolved"].get(key) != value:
                errors.append(key)
        amendment = cohort.get("source_amendments", {}).get(workflow)
        if amendment:
            source = read_object(
                client, PREFIX + workflow + "/final/source-receipt.json"
            )
            validation = read_object(client, amendment["native_validation"]["key"])
            if source is None or validation is None:
                pending.append(workflow)
                continue
            if strict_json(source)["commit"] != amendment["source_commit"]:
                errors.append("amended_source_commit")
            if (
                hashlib.sha256(validation).hexdigest()
                != amendment["native_validation"]["sha256"]
                or strict_json(validation).get("status") != "passed"
            ):
                errors.append("native_reset_validation")
        if errors:
            differences[workflow] = errors
        else:
            verified.append(workflow)
    return {"verified": verified, "pending": pending, "differences": differences}


def snapshot(client, campaign, cache, destination):
    result = {
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "deadline_utc": campaign["deadline_utc"],
        "cohorts": {},
        "complete": True,
    }
    destination.mkdir(parents=True, exist_ok=False)
    panels = []
    for cohort in campaign["cohorts"]:
        expected = {
            r["trial_id"]: r for r in cohort["schedule"] if r["split"] == "pilot"
        }
        ownership = {}
        for owner in cohort["owners"]:
            for trial_id in owner["trial_ids"]:
                if trial_id in ownership:
                    raise ValueError("duplicate_execution_owner")
                ownership[trial_id] = owner["workflow"]
        if set(ownership) != set(expected):
            raise ValueError("incomplete_execution_ownership")
        outcomes, missing, errors = [], [], {}
        for trial_id, reference in expected.items():
            try:
                row = collect_trial(
                    client, ownership[trial_id], reference, cohort, cache
                )
                if row is None:
                    missing.append(trial_id)
                else:
                    outcomes.append(row)
            except (ValueError, KeyError, TypeError) as error:
                errors[trial_id] = str(error)
        prior_outcomes, prior_missing = [], []
        for attempt in cohort.get("prior_attempts", []):
            try:
                previous = collect_trial(
                    client,
                    attempt["workflow"],
                    expected[attempt["trial_id"]],
                    cohort,
                    cache,
                )
                if previous is None:
                    prior_missing.append(attempt)
                else:
                    prior_outcomes.append(
                        {**previous, "recovery_reason": attempt["reason"]}
                    )
            except (ValueError, KeyError, TypeError) as error:
                errors[attempt["workflow"] + "/" + attempt["trial_id"]] = str(error)
        runtime = runtime_receipts(client, cohort)
        infra = [
            r["trial_id"] for r in outcomes if r["terminal_reason"] in INFRASTRUCTURE
        ]
        complete = not (
            missing
            or errors
            or infra
            or prior_missing
            or runtime["pending"]
            or runtime["differences"]
        )
        result["complete"] &= complete
        details = {
            "planned": len(expected),
            "verified_trials": len(outcomes),
            "missing": missing,
            "record_errors": errors,
            "infrastructure_trials": infra,
            "prior_attempts_retained": len(prior_outcomes),
            "prior_attempts_missing": prior_missing,
            "runtime_parity": runtime,
            "complete": complete,
            "source_amendments": cohort.get("source_amendments", {}),
            "setup_interruptions": cohort.get("setup_interruptions", []),
        }
        result["cohorts"][cohort["name"]] = details
        folder = destination / cohort["name"]
        analysis = export(outcomes, cohort["manifest"], folder, split="pilot")
        write_json(folder / "verified-outcomes.json", outcomes)
        write_json(folder / "prior-attempts.json", prior_outcomes)
        first = {row["trial_id"]: row for row in outcomes}
        for previous in reversed(prior_outcomes):
            first[previous["trial_id"]] = previous
        first_analysis = analyze(
            list(first.values()), cohort["manifest"], split="pilot"
        )
        write_json(folder / "first-attempt-analysis.json", first_analysis)
        write_json(folder / "first-attempt-outcomes.json", list(first.values()))
        write_json(folder / "completion.json", details)
        if complete:
            write_json(
                folder / "power-worksheet.json",
                power_worksheet(outcomes, cohort["manifest"]),
            )
        rows = []
        for arm, summary in analysis["arms"].items():
            n, successes = summary["n"], summary["successes"]
            rows.append(
                "<tr><td>%s</td><td>%s/%s</td><td>%s</td></tr>"
                % (
                    html.escape(arm),
                    successes,
                    n,
                    "%.1f%%" % (100 * successes / n) if n else "pending",
                )
            )
        images = []
        for name in ("success-by-task.png", "resource-distributions.png"):
            encoded = base64.b64encode((folder / name).read_bytes()).decode()
            images.append(
                '<img alt="%s" src="data:image/png;base64,%s">' % (name, encoded)
            )
        recovery_note = ""
        if cohort.get("setup_interruptions"):
            recovery_note += "<p>Previously unstarted trials were continued after a pre-actor reset-check interruption. The original failed workflow evidence is retained in completion.json.</p>"
        if cohort.get("source_amendments"):
            recovery_note += "<p>Reset-validation amendment: the affected starting state accepts only MuJoCo’s exact normalization of an already-unit quaternion; other state must match exactly. Actor code, prompts, controller, task, seed and episode limits are unchanged. Amended source hashes and native reset/replay evidence are recorded per worker in completion.json.</p>"
        if cohort.get("prior_attempts"):
            first_counts = ", ".join(
                "%s %s/%s" % (arm, value["successes"], value["n"])
                for arm, value in first_analysis["arms"].items()
            )
            recovery_note += (
                "<p>Transport recovery: %s earlier attempts retained. The charts use the documented recovery partition; whole matched blocks are repeated only after an infrastructure interruption. First-attempt counts, with interrupted attempts retained: %s. Both analyses and prior archive references are in the ZIP.</p>"
                % (len(prior_outcomes), html.escape(first_counts))
            )
        panels.append(
            "<section><h2>%s</h2><p>%s/%s audited trials · %s</p><table><tr><th>Method</th><th>Successes / trials</th><th>Success rate</th></tr>%s</table>%s</section>"
            % (
                html.escape(cohort["title"]),
                len(outcomes),
                len(expected),
                "complete pilot"
                if complete
                else "partial — completion/parity checks pending",
                "".join(rows),
                recovery_note + "".join(images),
            )
        )
    write_json(destination / "status.json", result)
    write_json(destination / "campaign.json", campaign)
    (destination / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>LIBERO pilot results</title><style>body{font:16px system-ui;margin:40px auto;padding:0 20px;max-width:1150px;color:#172535}section{margin:36px 0}img{width:100%%;height:auto}table{border-collapse:collapse}td,th{padding:10px 22px;border-bottom:1px solid #d2d9e1;text-align:left}</style>"
        "<h1>LIBERO pilot results</h1><p>Updated %s. Separate cohorts; exploratory pilot results, not confirmatory claims. Native 1,000-step episodes; no experiment token/time caps. Wall times include shared inference-service latency under concurrent execution.</p>%s"
        % (html.escape(result["observed_utc"]), "".join(panels))
    )
    return result


def publish(client, destination, workflow, final=False):
    prefix = (
        PREFIX + workflow + "/results/" + ("final" if final else destination.name) + "/"
    )
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            client.upload_file(
                str(path), "rldb", prefix + str(path.relative_to(destination))
            )
    zip_path = destination.parent / (destination.name + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as out:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                out.write(path, str(path.relative_to(destination)))
    client.upload_file(str(zip_path), "rldb", prefix + "libero-results.zip")
    receipt = {
        "prefix": prefix,
        "zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
        "status": strict_json((destination / "status.json").read_bytes()),
    }
    client.put_object(
        Bucket="rldb",
        Key=PREFIX + workflow + "/latest.json",
        Body=json.dumps(receipt).encode(),
    )
    return receipt


def main():
    campaign = strict_json(Path(os.environ["HARDWARE_CAMPAIGN"]).read_bytes())
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    )
    deadline = (
        datetime.fromisoformat(campaign["deadline_utc"]).timestamp()
        if campaign.get("deadline_utc")
        else None
    )
    root = Path("artifacts/parallel-results")
    root.mkdir(parents=True, exist_ok=False)
    while True:
        directory = root / (
            "snapshot-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        )
        try:
            result = snapshot(client, campaign, root / "cache", directory)
            final = result["complete"] or (
                deadline is not None and time.time() >= deadline
            )
            receipt = publish(
                client, directory, os.environ["HARDWARE_WORKFLOW"], final=final
            )
        except Exception as error:
            # An archive/network outage must not silently stop overnight reporting.
            # Preserve the partial attempt and retry without changing actor jobs.
            print(json.dumps({"report_retry": type(error).__name__}), flush=True)
            time.sleep(60)
            continue
        write_json(root / "latest.json", receipt, replace=True)
        print(
            json.dumps(
                {
                    "report": receipt["prefix"],
                    "complete": result["complete"],
                    "counts": {
                        k: v["verified_trials"] for k, v in result["cohorts"].items()
                    },
                }
            ),
            flush=True,
        )
        if final:
            break
        time.sleep(300)


if __name__ == "__main__":
    main()
