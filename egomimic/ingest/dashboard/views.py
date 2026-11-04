"""What the dashboard shows about a delivery: the bucket's listing and the watcher's
status merged into one row, and what of it a vendor may see."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from egomimic.ingest.store import Listed, Ref, Store

# What each state means, in the words a vendor reads.
STATES = {
    "uploading": "Upload in progress",
    "queued": "Waiting to be checked",
    "converting": "Checking and converting",
    "rejected": "Rejected: fix the problems and upload again",
    "unsupported": "Not a format we accept",
    "failed": "Failed on our side; we are looking into it",
    "ready": "Converted; waiting for EgoVerse review",
    "approved": "Approved; added to the dataset at the next pass",
    "committed": "In the dataset",
    "commit_rejected": "Approved, but not added yet; EgoVerse is looking into it",
    "declined": "Declined by EgoVerse",
}


@dataclass
class Row:
    ref: Ref
    state: str
    files: dict = field(default_factory=dict)  # name: (size, etag)
    status: dict = field(default_factory=dict)
    decision: dict = field(default_factory=dict)  # the newest, if for the current stage

    @property
    def decidable(self) -> bool:
        """Staff can approve or decline it: the watcher has it ready (whatever staff
        decided since, so a decline can still become an approval before the watcher
        acts) and the bucket has not moved on (re-uploaded, a correction under way)."""
        return (
            self.status.get("state") == "ready"
            and bool(self.status.get("stage"))
            and self.state in ("ready", "approved", "declined")
        )

    @property
    def key(self) -> str:
        return f"{self.ref.vendor}/{self.ref.delivery_id}"

    @property
    def size(self) -> int:
        return sum(size for size, _ in self.files.values())

    @property
    def updated(self) -> str:
        """The status's last write (a retry or an error leaves the state unchanged)."""
        history = self.status.get("history") or []
        return self.status.get("updated_at") or (history[-1]["at"] if history else "")

    @property
    def frames(self):
        return (self.status.get("report") or {}).get("num_frames")

    @property
    def problems(self) -> list:
        return self.status.get("problems") or []


def state_of(
    listed: Listed | None, status: dict | None, decision: dict | None = None
) -> str:
    """The watcher's state, unless the bucket has moved on since it wrote it, or staff
    decided on its current stage and the watcher has not acted yet."""
    if listed is not None and not listed.complete:
        return "uploading"  # a first upload, or a correction under way
    if status is None:
        return "queued"
    if listed is not None and status.get("fingerprint") != listed.fingerprint:
        return "queued"  # re-uploaded: the next pass checks the new files
    state = status.get("state", "queued")
    if state == "ready" and decision and decision.get("stage") == status.get("stage"):
        return "approved" if decision["decision"] == "approve" else "declined"
    return state


def rows(store: Store, vendor: str | None = None) -> list[Row]:
    """Every delivery of one vendor (None: all), newest activity first."""
    listed = {d.ref: d for d in store.list_deliveries(vendor)}
    statuses = store.statuses(vendor)
    decided = store.latest_decisions(vendor)
    out = []
    for ref in listed.keys() | statuses.keys():
        out.append(_row(ref, listed.get(ref), statuses.get(ref), decided.get(ref)))
    return sorted(out, key=lambda r: (r.updated, r.key), reverse=True)


def row(store: Store, ref: Ref) -> Row | None:
    d = store.get_delivery(ref)
    st = store.get_status(ref)
    if d is None and st is None:
        return None
    decisions = store.decisions(ref)
    return _row(ref, d, st, decisions[-1] if decisions else None)


def _row(ref, listed, status, decision) -> Row:
    status = status or {}
    current = (
        decision if decision and decision.get("stage") == status.get("stage") else None
    )
    return Row(
        ref,
        state_of(listed, status or None, current),
        listed.files if listed else {},
        status,
        current or {},
    )


_PATH = re.compile(r"(?<![\w.])/(?:[\w.@+-]+/)+[\w.@+-]*")


def vendor_error(status: dict) -> str:
    """The error as a vendor sees it: the first line of what the converter said about
    their files, with our paths blanked; nothing of our own crashes (a traceback) or
    of commit errors, which are ours to fix."""
    error = (status.get("error") or "").strip()
    if not error or status.get("state") not in ("failed", "rejected", "unsupported"):
        return ""
    if "Traceback" in error:
        return ""
    return _PATH.sub("<path>", error.splitlines()[0])[:300]


def vendor_decision(decision: dict) -> dict:
    """The decision without who made it."""
    d = decision or {}
    return {k: d[k] for k in ("decision", "comment", "at") if d.get(k)}
