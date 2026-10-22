"""One pass of the ingestion loop: convert new deliveries, hand approved ones to the committer.

Each pass:

1. Collect conversion outcomes: a delivery whose job recorded one becomes `ready` (staged,
   for review), `rejected` or `failed`; a job that ended without one fails it.
2. Submit conversions for every complete delivery that is new or re-uploaded, after the
   checks a listing allows (format, file sizes): at most `max_in_flight` running at once,
   and while a vendor has nothing approved at the current converter revision, only
   `canary` of its deliveries at a time (the rest wait as `queued`).
3. Act on staff decisions given for a delivery's current stage: decline drops the stage;
   approve hands it off. The raw files go to <raw_root>/<vendor>/<delivery>/<stage>/ and
   the staged delivery to <lance_staging>/<lab>/<vendor recording id>/ (meta.json last,
   carrying the provenance), both copied server-side; then the delivery and its stage
   are deleted from the ingest buckets.
4. Follow handed-off deliveries: `committed` once the committer took them, or
   `commit_rejected` with its errors (e.g. the recording is already committed, which
   needs lance_commit --replace).

egomimic.ingest.run runs a pass and then the committer every few minutes.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from egomimic.ingest import formats
from egomimic.ingest.launch import ENDED, Launcher
from egomimic.ingest.stage import META, STAGE, StageCorrupt, verify
from egomimic.ingest.store import Listed, Ref, Store, now

HISTORY = 20  # state changes kept in each status record
WAITING = ("converting", "ready")  # count against a vendor's canary
APPROVED = ("approved", "committed")


class Busy(Exception):
    """Another pass holds the lock."""


@contextmanager
def _flock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Busy(f"another pass holds {path}") from None
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@dataclass
class Summary:
    submitted: list[str] = field(default_factory=list)
    queued: list[str] = field(default_factory=list)  # held back by the caps
    staged: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    declined: list[str] = field(default_factory=list)
    approved: list[str] = field(default_factory=list)
    committed: list[str] = field(default_factory=list)
    commit_rejected: list[str] = field(default_factory=list)


def _key(ref: Ref) -> str:
    return f"{ref.vendor}/{ref.delivery_id}"


def _set(store: Store, ref: Ref, old: dict | None, **fields) -> dict:
    """Write the status, keeping a short history of states."""
    history = list((old or {}).get("history", []))
    if fields.get("state") and fields["state"] != (old or {}).get("state"):
        history = (history + [{"state": fields["state"], "at": now()}])[-HISTORY:]
    status = {**(old or {}), **fields, "history": history}
    store.put_status(ref, status)
    return status


def _drop_stage(store: Store, ref: Ref, digest: str) -> None:
    if digest and store.has_staged(ref, digest):
        store.delete_staged(ref, digest)


# ---------------------------------------------------------------- conversion


def _needs_conversion(d: Listed, status: dict | None) -> bool:
    if not d.complete:
        return False
    if status is None or status.get("state") == "queued":
        return True
    return status.get("fingerprint") != d.fingerprint


FRESH = {
    "problems": [],
    "error": "",
    "stage": "",
    "diff": {},
    "report": {},
    "recording_id": "",
    "provenance": {},
}


def _precheck(d: Listed) -> tuple[str, dict] | None:
    """A refusal the listing alone decides (no format, ambiguous, oversized), or None."""
    try:
        fmt = formats.detect(d.files)
    except formats.AmbiguousFormat as e:
        return "failed", {"error": str(e)}
    if fmt is None:
        known = "; ".join(
            f"{f.name}: {', '.join(sorted(f.required))}" for f in formats.formats()
        )
        return "unsupported", {
            "problems": [f"no known format has files {sorted(d.files)} ({known})"]
        }
    problems = fmt.oversize({n: size for n, (size, _) in d.files.items()})
    extra = fmt.unexpected(set(d.files))
    if extra:
        problems.insert(0, f"{fmt.name} deliveries hold only {sorted(fmt.allowed)}")
    return (
        ("rejected", {"problems": problems, "format": fmt.name}) if problems else None
    )


def _collect(store: Store, launcher: Launcher, ref: Ref, status: dict, summary):
    out = store.get_job(ref, status["fingerprint"])
    name = _key(ref)
    if out is None:
        if launcher.state(status.get("job", "")) == ENDED:
            _set(
                store, ref, status, state="failed",
                error="the conversion job ended without an outcome "
                "(killed, out of memory, or interrupted past its retries)",
            )  # fmt: skip
            summary.failed.append(name)
        return
    if out["status"] == "superseded":  # re-uploaded meanwhile: convert the new one
        _set(store, ref, status, state="queued", fingerprint="")
        return
    fields = {k: out.get(k, FRESH.get(k)) for k in FRESH}
    if out["status"] == "staged":
        fields |= {k: out[k] for k in ("lab", "vendor_recording_id")}
        _set(store, ref, status, state="ready", **fields)
        summary.staged.append(name)
    else:
        _set(store, ref, status, state=out["status"], **fields)
        (summary.failed if out["status"] == "failed" else summary.rejected).append(name)
    _drop_stage(store, ref, status.get("previous_stage", ""))


def _canary_room(statuses: dict, vendor: str, revision: str, canary: int) -> int | None:
    """How many more of `vendor`'s deliveries may convert at `revision`: None (no limit)
    once one was approved at it, else what is left of the canary."""
    mine = [
        s
        for r, s in statuses.items()
        if r.vendor == vendor and s.get("revision") == revision
    ]
    if any(s.get("state") in APPROVED for s in mine):
        return None
    return max(0, canary - sum(s.get("state") in WAITING for s in mine))


# ---------------------------------------------------------------- hand-off


def _hand_off(store: Store, ref: Ref, status: dict, raw_root, lance_staging) -> str:
    """Copy an approved stage and its raw delivery out of the ingest buckets; returns
    where the committer will find it."""
    import hashlib

    from egomimic.rldb.lance import commit

    digest = status["stage"]
    pin_bytes = store.get_staged_file(ref, digest, STAGE)
    if pin_bytes is None or hashlib.sha256(pin_bytes).hexdigest() != digest:
        raise StageCorrupt(f"stored stage {digest} does not hash to its own digest")
    pin = json.loads(pin_bytes)
    verify(pin, store.staged_files(ref, digest))
    current = store.get_delivery(ref)
    raw = commit.join(raw_root, ref.vendor, ref.delivery_id, digest)
    for name in current.files:
        store.copy_out(ref.file_key(name), commit.join(raw, name))
    dst = commit.join(lance_staging, pin["lab"], pin["vendor_recording_id"])
    for name in pin["files"]:
        if name != META:
            store.copy_out(ref.staged_prefix(digest) + name, commit.join(dst, name))
    meta = json.loads(store.get_staged_file(ref, digest, META))
    meta["provenance"] = {
        **meta.get("provenance", {}),
        "stage": digest,
        "raw": raw,
        "vendor": ref.vendor,
        "delivery_id": ref.delivery_id,
    }
    commit.put_bytes(json.dumps(meta, indent=1).encode(), commit.join(dst, META))
    return dst


def _decide(store: Store, ref: Ref, status: dict, raw_root, lance_staging, summary):
    stage = status.get("stage", "")
    decisions = [x for x in store.decisions(ref) if x.get("stage") == stage]
    if not stage or not decisions:
        return
    last = decisions[-1]
    name = _key(ref)
    if last["decision"] == "decline":
        _drop_stage(store, ref, stage)
        _set(store, ref, status, state="declined", decision=last)
        summary.declined.append(name)
        return
    # Read the delivery fresh: it may have been re-uploaded, or a correction started,
    # since it was staged. The next pass converts the new version for review.
    current = store.get_delivery(ref)
    if (
        current is None
        or not current.complete
        or current.fingerprint != status.get("fingerprint")
    ):
        return
    try:
        dst = _hand_off(store, ref, status, raw_root, lance_staging)
    except Exception as e:  # stays ready and approved: retried next pass
        _set(
            store, ref, status,
            error=f"hand-off failed: {type(e).__name__}: {e}\n"
            f"{traceback.format_exc(limit=3)}",
        )  # fmt: skip
        summary.failed.append(name)
        return
    store.delete_delivery(ref)
    _drop_stage(store, ref, stage)
    store.delete_jobs(ref)
    _set(store, ref, status, state="approved", decision=last, handed_off=dst, error="")
    summary.approved.append(name)


def _follow(store: Store, ref: Ref, status: dict, summary) -> None:
    from egomimic.rldb.lance import commit
    from egomimic.scripts.lance_commit import REJECTED

    dst = status["handed_off"]
    files = commit.list_files(dst)
    if REJECTED in files:
        note = json.loads(commit.get_bytes(commit.join(dst, REJECTED)))
        _set(store, ref, status, state="commit_rejected", problems=note["errors"])
        summary.commit_rejected.append(_key(ref))
    elif not files:  # the committer removes what it committed
        _set(store, ref, status, state="committed")
        summary.committed.append(_key(ref))


# ---------------------------------------------------------------- pass


def run_pass(
    store: Store,
    launcher: Launcher,
    work: Path,
    *,
    raw_root,
    lance_staging,
    revision: str,
    max_in_flight: int = 56,
    canary: int = 3,
) -> Summary:
    """One pass over the ingest buckets. Raises Busy if another pass is running."""
    summary = Summary()
    with _flock(Path(work) / ".watch.lock"):

        def collect():
            for ref, status in sorted(store.statuses().items()):
                if status.get("state") == "converting":
                    _collect(store, launcher, ref, status, summary)

        collect()
        statuses = store.statuses()
        in_flight = sum(s.get("state") == "converting" for s in statuses.values())
        for d in store.list_deliveries():
            ref, status = d.ref, statuses.get(d.ref)
            if not _needs_conversion(d, status):
                continue
            refused = _precheck(d)
            if refused:
                state, fields = refused
                statuses[ref] = _set(
                    store, ref, status, state=state, fingerprint=d.fingerprint,
                    **{**FRESH, **fields},
                )  # fmt: skip
                (summary.failed if state == "failed" else summary.rejected).append(
                    _key(ref)
                )
                _drop_stage(store, ref, (status or {}).get("stage", ""))
                continue
            room = _canary_room(statuses, ref.vendor, revision, canary)
            if in_flight >= max_in_flight or room == 0:
                if (status or {}).get("state") != "queued":
                    statuses[ref] = _set(
                        store, ref, status, state="queued", fingerprint="", **FRESH
                    )
                summary.queued.append(_key(ref))
                continue
            try:
                job = launcher.submit(ref, d.fingerprint)
            except Exception as e:
                statuses[ref] = _set(
                    store, ref, status, state="queued", fingerprint="",
                    error=f"submit failed: {type(e).__name__}: {e}",
                )  # fmt: skip
                summary.queued.append(_key(ref))
                continue
            statuses[ref] = _set(
                store, ref, status, state="converting", fingerprint=d.fingerprint,
                job=job, revision=revision,
                previous_stage=(status or {}).get("stage", ""), **FRESH,
            )  # fmt: skip
            in_flight += 1
            summary.submitted.append(_key(ref))
        collect()  # jobs that ran in-process are already done
        for ref, status in sorted(store.statuses().items()):
            if status.get("state") == "ready":
                _decide(store, ref, status, raw_root, lance_staging, summary)
        for ref, status in sorted(store.statuses().items()):
            if status.get("state") == "approved":
                _follow(store, ref, status, summary)
    return summary


def mano_problem(model_dir: str | None = None) -> str | None:
    """Why the MANO fit cannot run here, or None. Checked before converting, so a
    missing package or model file stops the job instead of failing every delivery."""
    try:
        import mano  # noqa: F401  (optional, license-gated: not in the lockfile)

        from egomimic.scripts.aria_process.aria_utils import _default_mano_model_dir

        model_dir = Path(model_dir or _default_mano_model_dir())
    except Exception as e:
        return f"MANO is not installed ({type(e).__name__}: {e})"
    missing = [
        n for n in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl") if not (model_dir / n).is_file()
    ]
    if missing:
        return f"MANO model files missing from {model_dir}: {', '.join(missing)}"
    return None


def add_pass_args(ap: argparse.ArgumentParser) -> None:
    from egomimic.ingest.convert_job import add_convert_args, add_store_args

    ap.add_argument("--lance", required=True, help="Lance root (read by the jobs)")
    ap.add_argument("--work", type=Path, required=True, help="local scratch")
    ap.add_argument("--raw-root", default="s3://rldb/raw_v2")
    ap.add_argument("--lance-staging", default="s3://rldb/lance_staging")
    ap.add_argument("--launcher", choices=("local", "batch"), default="batch")
    ap.add_argument("--batch-queue", default="egoverse-convert")
    ap.add_argument("--batch-definition", default="egoverse-convert")
    ap.add_argument("--job-work", default="/scratch", help="the job's local scratch")
    ap.add_argument("--max-in-flight", type=int, default=56)
    ap.add_argument("--canary", type=int, default=3)
    add_store_args(ap)
    add_convert_args(ap)


def launcher_from_args(args, store: Store) -> Launcher:
    from egomimic.ingest.convert_job import convert_settings
    from egomimic.ingest.launch import BatchLauncher, LocalLauncher

    if args.launcher == "local":
        return LocalLauncher(
            store, args.lance, args.work / "jobs", **convert_settings(args)
        )
    job_args = ["--lance", str(args.lance), "--work", args.job_work]
    job_args += ["--bucket", args.bucket, "--vendor-bucket", args.vendor_bucket]
    if args.root:
        job_args += ["--root", args.root]
    job_args += ["--codec", args.codec, "--mano-iters", str(args.mano_iters)]
    job_args += ["--mano" if args.mano else "--no-mano"]
    if args.mano_model_dir:
        job_args += ["--mano-model-dir", args.mano_model_dir]
    return BatchLauncher(args.batch_queue, args.batch_definition, job_args)


def pass_from_args(args, store: Store) -> Summary:
    from egomimic.ingest.stage import converter_revision

    return run_pass(
        store,
        launcher_from_args(args, store),
        args.work,
        raw_root=args.raw_root,
        lance_staging=args.lance_staging,
        revision=converter_revision()["revision"],
        max_in_flight=args.max_in_flight,
        canary=args.canary,
    )


def main(argv: list[str] | None = None) -> int:
    from egomimic.ingest.convert_job import store_from_args

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_pass_args(ap)
    args = ap.parse_args(argv)
    if args.launcher == "local" and args.mano:
        if problem := mano_problem(args.mano_model_dir):
            print(json.dumps({"error": f"{problem}; install it or pass --no-mano"}))
            return 2
    try:
        s = pass_from_args(args, store_from_args(args))
    except Busy as e:
        print(json.dumps({"busy": str(e)}))
        return 0
    print(json.dumps(s.__dict__))
    return 1 if s.failed else 0


if __name__ == "__main__":
    sys.exit(main())
