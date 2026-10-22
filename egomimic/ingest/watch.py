"""One pass of the ingestion loop: stage new deliveries, commit approved ones.

    python -m egomimic.ingest.watch --lance <root> --work <dir> [--bucket rldb | --local-store <dir>]

Each pass, idempotent and safe to repeat (a Slurm job runs it every few minutes):

1. Stage every complete delivery that is new, re-uploaded (its fingerprint moved) or
   whose last stage went stale: check sizes from the listing, download and verify the
   snapshot, prepare it with its format, upload the staged result to the bucket under
   its digest, and record the outcome in its status (what vendors and staff see).
2. Act on staff decisions given for a delivery's current stage: approve -> download the
   stage, verify it against the approved digest and commit it (this pass is the only
   Lance writer); decline -> drop the stage. Decisions for an older stage are ignored.
3. Optionally tag every table's version after commits.

A delivery that fails is recorded and the pass moves on. Before committing, the
delivery is read fresh from the bucket and must still be complete with the fingerprint
its stage was made from (staging other deliveries can take long enough for a vendor to
start a correction). Locks on the work folder and on the Lance root keep two passes from
overlapping, or from writing one Lance root through different work folders; watchers on
different machines writing one Lance root are not covered (run exactly one).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from egomimic.ingest import formats
from egomimic.ingest.stage import StaleStage, read_pin, stage_digest
from egomimic.ingest.store import (
    DeliveryChanged,
    Listed,
    LocalStore,
    R2Store,
    Ref,
    Store,
    now,
)

HISTORY = 20  # state changes kept in each status record


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


@contextmanager
def _lock(work: Path, lance_root: Path):
    """The work folder and the Lance root: one pass per folder, one writer per root."""
    with _flock(Path(work) / ".watch.lock"), _flock(Path(lance_root) / ".ingest.lock"):
        yield


@dataclass
class Summary:
    staged: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    committed: list[str] = field(default_factory=list)
    declined: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # changed mid-download: next pass
    tag: dict | None = None


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


# ---------------------------------------------------------------- staging


def _needs_staging(d: Listed, status: dict | None) -> bool:
    if not d.complete:
        return False
    if status is None or status.get("fingerprint") != d.fingerprint:
        return True
    # "processing" while this pass holds the lock means an earlier pass was killed
    # (timeout, preemption) mid-staging: start it again.
    return status.get("state") in ("stale", "processing")


def _drop_superseded(store: Store, ref: Ref, old_stage: str) -> None:
    """A delivery's previous stage, once a new outcome replaced it: delete it."""
    new = (store.get_status(ref) or {}).get("stage", "")
    if old_stage and old_stage != new and store.has_staged(ref, old_stage):
        store.delete_staged(ref, old_stage)


def _stage(
    store: Store, d: Listed, status: dict | None, lance_root, work, settings, summary
):
    ref = d.ref
    name = _key(ref)
    names = set(d.files)
    fresh = {
        "fingerprint": d.fingerprint,
        "problems": [],
        "error": "",
        "stage": "",
        "diff": {},
        "report": {},
        "format": "",
        "recording_id": "",
        "provenance": {},
    }
    old_stage = (status or {}).get("stage", "")
    try:
        fmt = formats.detect(names)
    except formats.AmbiguousFormat as e:
        _set(store, ref, status, state="failed", **{**fresh, "error": str(e)})
        summary.failed.append(name)
        _drop_superseded(store, ref, old_stage)
        return
    if fmt is not None:
        too_big = fmt.oversize({n: size for n, (size, _) in d.files.items()})
        if too_big:  # refuse before downloading hundreds of GB
            _set(
                store,
                ref,
                status,
                state="rejected",
                **{**fresh, "format": fmt.name, "problems": too_big},
            )
            summary.rejected.append(name)
            _drop_superseded(store, ref, old_stage)
            return
    status = _set(store, ref, status, state="processing", **fresh)
    local = work / "deliveries" / ref.vendor / ref.delivery_id
    staged = work / "staged" / ref.vendor / ref.delivery_id
    try:
        try:
            store.download(d, local)
        except DeliveryChanged:
            # Re-uploaded while we looked: try again next pass.
            _set(store, ref, status, state="queued", fingerprint="")
            summary.skipped.append(name)
            return
        p = formats.prepare_delivery(
            local, staged, lance_root, fingerprint=d.fingerprint, **settings
        )
        fields = {
            "format": fmt.name if fmt else "",
            "recording_id": p.recording_id,
            "problems": p.problems,
            "error": p.error,
            "report": p.report,
            "diff": p.diff,
        }
        if p.status == "staged":
            digest = stage_digest(p.staged)
            store.put_staged(ref, digest, p.staged)
            # Which code and settings made it, kept after the stage is deleted on commit.
            pin = read_pin(p.staged)
            fields["provenance"] = {
                k: pin.get(k) for k in ("converter", "settings", "content_digest")
            }
            _set(store, ref, status, state="ready", stage=digest, **fields)
            summary.staged.append(name)
        else:
            _set(store, ref, status, state=p.status, **fields)
            (summary.failed if p.status == "failed" else summary.rejected).append(name)
    except Exception as e:  # never let one delivery stop the pass
        _set(
            store, ref, status, state="failed",
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}",
        )  # fmt: skip
        summary.failed.append(name)
    finally:
        shutil.rmtree(local, ignore_errors=True)
        shutil.rmtree(staged, ignore_errors=True)
        _drop_superseded(store, ref, old_stage)


# ---------------------------------------------------------------- decisions


def _decide(store: Store, ref: Ref, status: dict, lance_root, work, summary):
    stage = status.get("stage", "")
    decisions = [x for x in store.decisions(ref) if x.get("stage") == stage]
    if not stage or not decisions:
        return
    last = decisions[-1]
    name = _key(ref)
    if last["decision"] == "decline":
        store.delete_staged(ref, stage)
        _set(store, ref, status, state="declined", decision=last)
        summary.declined.append(name)
        return
    # Read the delivery fresh: it may have been re-uploaded, or a correction started
    # (which clears completion), while this pass staged other deliveries.
    current = store.get_delivery(ref)
    if (
        current is None
        or not current.complete
        or current.fingerprint != status.get("fingerprint")
    ):
        return  # the next pass stages the new version for review
    local = work / "commit" / ref.vendor / ref.delivery_id
    try:
        status = _set(store, ref, status, state="committing", decision=last)
        store.get_staged(ref, stage, local)
        if stage_digest(local) != stage:
            raise StaleStage(f"stored stage {stage} does not hash to its own digest")
        out = formats.commit_delivery(
            local,
            lance_root,
            approved_stage=stage,
            fingerprint=current.fingerprint,
        )
        _set(store, ref, status, state="committed", committed=out)
        store.delete_staged(ref, stage)
        summary.committed.append(name)
    except StaleStage as e:
        # Lance or the delivery moved on since the review: re-stage and re-review.
        _set(store, ref, status, state="stale", error=str(e))
        summary.stale.append(name)
    except Exception as e:
        _set(
            store, ref, status, state="ready",  # stays approved; retried next pass
            error=f"commit failed: {type(e).__name__}: {e}",
        )  # fmt: skip
        summary.failed.append(name)
    finally:
        shutil.rmtree(local, ignore_errors=True)


# ---------------------------------------------------------------- pass


def run_pass(
    store: Store,
    lance_root: Path,
    work: Path,
    *,
    tag: str | None = None,
    **settings,
) -> Summary:
    """One pass over the bucket. `settings` go to the format's prepare (e.g. codec,
    mano). Raises Busy if another pass is running."""
    lance_root, work = Path(lance_root), Path(work)
    summary = Summary()
    with _lock(work, lance_root):
        listed = {d.ref: d for d in store.list_deliveries()}
        statuses = store.statuses()
        for ref, d in listed.items():
            if _needs_staging(d, statuses.get(ref)):
                _stage(store, d, statuses.get(ref), lance_root, work, settings, summary)
        statuses = store.statuses()  # what staging just wrote
        for ref, status in sorted(statuses.items()):
            # "committing" under the lock: an earlier pass was killed mid-commit. Its
            # approved stage is still in the bucket and the commit resumes (stage
            # commit_staged finishes a partial multi-table write).
            if status.get("state") in ("ready", "committing"):
                _decide(store, ref, status, lance_root, work, summary)
        if tag and summary.committed:
            from egomimic.rldb.lance import commit

            summary.tag = commit.tag(
                lance_root, tag, lab=_lab_of(store, summary.committed)
            )
    return summary


def _lab_of(store: Store, committed: list[str]) -> str:
    labs = set()
    for name in committed:
        status = store.get_status(Ref(*name.split("/", 1))) or {}
        versions = (status.get("committed") or {}).get("versions", {})
        labs |= {Path(k).stem for k in versions if k.startswith("steps/")}
    if len(labs) != 1:
        raise ValueError(
            f"--tag covers one steps table; this pass committed {sorted(labs)}"
        )
    return labs.pop()


def mano_problem(model_dir: str | None = None) -> str | None:
    """Why the MANO fit cannot run here, or None. Checked before a pass, so a missing
    package or model file stops the job instead of failing every Aria delivery."""
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lance", type=Path, required=True, help="Lance root")
    ap.add_argument("--work", type=Path, required=True, help="scratch for downloads")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--bucket", default="rldb", help="R2 bucket (default rldb)")
    src.add_argument("--local-store", type=Path, help="a directory instead of R2")
    ap.add_argument("--root", default="", help="R2 key prefix, e.g. _test/run1/")
    ap.add_argument("--tag", help="tag every table's version after commits")
    ap.add_argument("--codec", default="libsvtav1")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--mano", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--mano-model-dir", help="MANO_LEFT/RIGHT.pkl (default: external_ckpts/mano)"
    )
    args = ap.parse_args(argv)
    if args.mano and (problem := mano_problem(args.mano_model_dir)):
        print(json.dumps({"error": f"{problem}; install it or pass --no-mano"}))
        return 2
    store = (
        LocalStore(args.local_store)
        if args.local_store
        else R2Store(args.bucket, root=args.root)
    )
    try:
        s = run_pass(
            store, args.lance, args.work, tag=args.tag,
            codec=args.codec, ffmpeg=args.ffmpeg, mano=args.mano,
            mano_model_dir=args.mano_model_dir,
        )  # fmt: skip
    except Busy as e:
        print(json.dumps({"busy": str(e)}))
        return 0
    print(json.dumps(s.__dict__))
    return 1 if s.failed else 0


if __name__ == "__main__":
    sys.exit(main())
