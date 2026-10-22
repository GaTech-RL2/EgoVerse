"""The bucket the ingestion pipeline runs through: vendor deliveries, their status, staff
decisions and dashboard accounts.

The dashboard (on AWS) and the watcher (on SkyNet) share nothing but this bucket, so every
kind of object has exactly one writer:

    deliveries/<vendor>/<delivery_id>/<file>             vendor files (presigned uploads)
    deliveries/<vendor>/<delivery_id>/_complete.json     dashboard, when an upload finishes:
                                                         the exact files (size, etag) it holds
    pipeline/status/<vendor>/<delivery_id>.json          watcher
    pipeline/decisions/<vendor>/<delivery_id>/<t>-<who>-<decision>-<stage>.json
                                                         dashboard (staff)
    pipeline/uploads/<vendor>/<delivery_id>/<id>.json    dashboard: uploads in progress
    pipeline/staged/<vendor>/<delivery_id>/<digest>/...  watcher: staged results, by digest
    pipeline/accounts.json                               dashboard (staff)

A delivery is complete only while its files match the manifest in _complete.json, and
starting any new upload to it removes that marker; it cannot be marked complete while an
upload to it is unfinished. So the watcher never sees a mix of old and new files during a
correction. The layout is format-neutral: what a delivery holds is
decided later, from its files.
Vendors upload from the browser through presigned multipart parts (start_upload ...);
trusted lab machines with R2 credentials use upload_file / upload_delivery (boto3's
managed uploader). Both end in the same manifest and completion marker, so everything
downstream is identical. Two backends share the API: R2Store (the `rldb` bucket, via boto3) and LocalStore (a
directory, for tests and local runs).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import shutil
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

DELIVERIES = "deliveries"
PIPELINE = "pipeline"
COMPLETE = "_complete.json"
ACCOUNTS = f"{PIPELINE}/accounts.json"

# Vendors are lowercase slugs (as the spec's `vendor` key); delivery ids may also hold dots
# and upper case, e.g. an episode id or a VRS stem.
VENDOR = re.compile(r"[a-z0-9]+(?:[_-][a-z0-9]+)*")
DELIVERY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


@dataclass(frozen=True, order=True)
class Ref:
    """One delivery: a vendor's folder under deliveries/."""

    vendor: str
    delivery_id: str

    def __post_init__(self):
        if not VENDOR.fullmatch(self.vendor):
            raise ValueError(f"vendor {self.vendor!r} is not a lowercase slug")
        if not DELIVERY_ID.fullmatch(self.delivery_id):
            raise ValueError(f"delivery id {self.delivery_id!r} has unsafe characters")

    @property
    def prefix(self) -> str:
        return f"{DELIVERIES}/{self.vendor}/{self.delivery_id}/"

    def file_key(self, name: str) -> str:
        return self.prefix + _file_name(name)

    @property
    def status_key(self) -> str:
        return f"{PIPELINE}/status/{self.vendor}/{self.delivery_id}.json"

    @property
    def decisions_prefix(self) -> str:
        return f"{PIPELINE}/decisions/{self.vendor}/{self.delivery_id}/"

    def staged_prefix(self, digest: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{digest!r} is not a stage digest")
        return f"{PIPELINE}/staged/{self.vendor}/{self.delivery_id}/{digest}/"

    @property
    def uploads_prefix(self) -> str:
        return f"{PIPELINE}/uploads/{self.vendor}/{self.delivery_id}/"

    def upload_key(self, upload_id: str) -> str:
        # R2 upload ids are long and not path-safe: key the record by their hash.
        return self.record_key(hashlib.sha256(upload_id.encode()).hexdigest())

    def record_key(self, record: str) -> str:
        if not re.fullmatch(r"(pending-)?[0-9a-f]{32,64}", record):
            raise ValueError(f"{record!r} is not an upload record id")
        return f"{self.uploads_prefix}{record}.json"


_DECISION = re.compile(
    r"(?P<at>\d{8}T\d{6}\.\d{6}Z)-(?P<who>[a-z0-9_-]+)-(?P<decision>approve|decline)"
    r"-(?P<stage>[0-9a-f]{64}|)\.json"
)


def _file_name(name: str) -> str:
    """A delivery-relative path (subfolders allowed), never escaping the delivery."""
    p = PurePosixPath(name)
    if p.is_absolute() or not p.parts or any(x in ("", ".", "..") for x in p.parts):
        raise ValueError(f"file name {name!r} is not a plain relative path")
    if p.name == COMPLETE:
        raise ValueError(f"{COMPLETE} is written by the dashboard, not uploaded")
    return str(p)


@dataclass
class Listed:
    """A delivery as found in the bucket."""

    ref: Ref
    files: dict[str, tuple[int, str]] = field(default_factory=dict)  # name: size, etag
    complete: bool = False  # marked complete, and the files are exactly that snapshot
    marked: bool = False  # a _complete.json exists (it may no longer match the files)

    @property
    def fingerprint(self) -> str:
        """Changes when any file is added, removed or re-uploaded."""
        h = hashlib.sha256()
        for name, (size, etag) in sorted(self.files.items()):
            h.update(f"{name}\0{size}\0{etag}\n".encode())
        return h.hexdigest()


class DeliveryChanged(Exception):
    """A delivery's files changed while it was being read; try again later."""


class UploadRejected(ValueError):
    """An upload broke what it declared when it started (parts, part sizes, file size);
    whatever it wrote has been removed."""


def part_length(rec: dict, n: int) -> int | None:
    """The exact length part n of a size-bound upload must have, or None when the upload
    declared no size (internal uploads). Raises UploadRejected for a part outside it."""
    if rec.get("size") is None:
        return None
    if not 1 <= n <= rec["parts"]:
        raise UploadRejected(f"part {n} is outside 1..{rec['parts']}")
    if n < rec["parts"]:
        return rec["part_size"]
    return rec["size"] - (rec["parts"] - 1) * rec["part_size"]


class UploadsInProgress(Exception):
    """The delivery still has unfinished uploads, so it cannot be marked complete. A
    record left by a crashed or abandoned upload is removed with Store.clear_upload."""

    def __init__(self, ref: "Ref", records: list[dict]):
        ids = ", ".join(r["record"] for r in records) or "(just started)"
        super().__init__(f"{ref.prefix} has unfinished uploads: {ids}")
        self.records = records


def now() -> str:
    """UTC timestamp used in records and decision keys; sorts in time order."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


class Store(ABC):
    """The pipeline's view of the bucket. Keys are '/'-separated, relative to it."""

    # ---- primitives each backend provides

    @abstractmethod
    def _list(self, prefix: str) -> dict[str, tuple[int, str]]:
        """key -> (size, etag) for every object under `prefix`."""

    @abstractmethod
    def _get(self, key: str) -> bytes | None:
        """The object's bytes, or None if it does not exist."""

    @abstractmethod
    def _put(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def _download(self, key: str, dest: Path) -> None: ...

    @abstractmethod
    def _delete(self, key: str) -> None:
        """Remove the object if it exists."""

    @abstractmethod
    def _start(self, key: str) -> str:
        """Begin a multipart upload of `key`; returns its upload id."""

    @abstractmethod
    def _finish(self, key: str, upload_id: str, parts: list[tuple[int, str]]) -> None:
        """Join the parts into the object at `key`."""

    @abstractmethod
    def _abort(self, key: str, upload_id: str) -> None:
        """Abort a multipart upload; one that no longer exists is not an error."""

    @abstractmethod
    def _upload_file(self, key: str, path: Path) -> None:
        """Upload a local file to `key` in one managed operation (multipart for big
        files, aborted on failure)."""

    # ---- uploads from a trusted machine (lab tooling with R2 credentials)

    def upload_file(self, ref: Ref, path: Path, name: str | None = None) -> None:
        """Upload one local file into the delivery, under the same rules as vendor
        uploads: tracked as in progress, and the delivery stops being complete."""
        name = name or Path(path).name
        key = ref.file_key(name)
        record = ref.record_key(uuid.uuid4().hex)
        self._put(
            record, _json({"name": name, "state": "uploading", "started_at": now()})
        )
        try:
            self._delete(ref.prefix + COMPLETE)
            self._upload_file(key, Path(path))
        finally:
            self._delete(record)

    def upload_delivery(self, ref: Ref, folder: Path, info: dict) -> dict:
        """Upload every file under `folder` (subfolders kept) and mark the delivery
        complete as exactly those files; returns its manifest. The internal twin of a
        vendor upload session."""
        folder = Path(folder)
        files = sorted(p for p in folder.rglob("*") if p.is_file())
        if not files:
            raise ValueError(f"{folder} has no files")
        for p in files:
            self.upload_file(ref, p, p.relative_to(folder).as_posix())
        # The folder is the whole delivery: files from an earlier version not in it go.
        names = [p.relative_to(folder).as_posix() for p in files]
        return self.mark_complete(ref, info, names)

    # ---- multipart uploads from a vendor's browser (presigned parts)

    def start_upload(
        self,
        ref: Ref,
        name: str,
        size: int | None = None,
        part_size: int | None = None,
    ) -> str:
        """Begin a multipart upload of one file; returns its upload id. The delivery
        stops being complete, and cannot be marked complete again, until the upload is
        finished or aborted. With `size` (and `part_size`), the upload is bound to them:
        its record pins size, part size and part count, part links are signed for each
        part's exact length, and finishing needs exactly those parts and that size."""
        key = ref.file_key(name)
        bound = {}
        if size is not None:
            if size < 0 or not part_size or part_size <= 0:
                raise ValueError("a size-bound upload needs size >= 0 and a part size")
            parts = max(1, -(-size // part_size))
            bound = {"size": size, "part_size": part_size, "parts": parts}
        # In this order: once the pending record exists, a concurrent mark_complete
        # either sees it and refuses, or loses its marker to the delete below.
        pending = ref.record_key(f"pending-{uuid.uuid4().hex}")
        self._put(
            pending, _json({"name": name, "state": "pending", "started_at": now()})
        )
        try:
            self._delete(ref.prefix + COMPLETE)
            upload_id = self._start(key)
        except BaseException:
            self._delete(pending)  # nothing was started: nothing to track
            raise
        rec = {
            "name": name,
            "state": "uploading",
            "upload_id": upload_id,
            "started_at": now(),
            **bound,
        }
        try:
            self._put(ref.upload_key(upload_id), _json(rec))
        except BaseException:
            try:  # an untracked upload would be invisible: undo it
                self._abort(key, upload_id)
            finally:
                self._delete(pending)
            raise
        self._delete(pending)
        return upload_id

    def finish_upload(
        self, ref: Ref, name: str, upload_id: str, parts: list[tuple[int, str]]
    ) -> None:
        """Join the parts [(part number, etag), ...] into the file. A size-bound upload
        needs exactly parts 1..N, and the joined file must have the declared size: if
        it does not, the file and the upload are removed (UploadRejected)."""
        key = ref.file_key(name)
        rec = self.upload_record(ref, upload_id) or {}
        if rec.get("size") is not None:
            numbers = sorted(n for n, _ in parts)
            if numbers != list(range(1, rec["parts"] + 1)):
                raise UploadRejected(
                    f"{name}: needs exactly parts 1..{rec['parts']}, got {len(numbers)}"
                )
        self._finish(key, upload_id, parts)
        if rec.get("size") is not None:
            got = self._files(ref).get(_file_name(name), (None,))[0]
            if got != rec["size"]:
                self._delete(key)
                self._delete(ref.upload_key(upload_id))
                raise UploadRejected(
                    f"{name}: {got} bytes uploaded, {rec['size']} declared; removed"
                )
        self._delete(ref.upload_key(upload_id))

    def abort_upload(self, ref: Ref, name: str, upload_id: str) -> None:
        self._abort(ref.file_key(name), upload_id)
        self._delete(ref.upload_key(upload_id))

    def upload_record(self, ref: Ref, upload_id: str) -> dict | None:
        """The record of an unfinished upload to this delivery, or None: how the
        dashboard checks an upload id a client sends belongs to that delivery."""
        return _load(self._get(ref.upload_key(upload_id)))

    def uploads_in_progress(self, ref: Ref) -> list[dict]:
        """Unfinished uploads to the delivery, each with its `record` id and `state`
        ("pending": being started, "uploading"), oldest first."""
        out = []
        for key in sorted(self._list(ref.uploads_prefix)):
            rec = _load(self._get(key))
            if rec is not None:
                out.append(
                    {**rec, "record": key.rsplit("/", 1)[1].removesuffix(".json")}
                )
        return sorted(out, key=lambda r: r.get("started_at", ""))

    def clear_upload(self, ref: Ref, record: str) -> None:
        """Recover from a crashed, abandoned or half-bookkept upload: abort its multipart
        upload if it still exists (already finished or gone is fine) and remove the
        record, so the delivery can be marked complete again."""
        key = ref.record_key(record)
        rec = _load(self._get(key)) or {}
        if rec.get("upload_id"):
            # Gone (finished / expired) counts as aborted; any other failure propagates and
            # keeps the record, so the cleanup can be retried.
            self._abort(ref.file_key(rec["name"]), rec["upload_id"])
        self._delete(key)

    def remove_file(self, ref: Ref, name: str) -> None:
        """Delete one file from the delivery; it stops being complete."""
        self._delete(ref.prefix + COMPLETE)
        self._delete(ref.file_key(name))

    @abstractmethod
    def part_url(
        self,
        ref: Ref,
        name: str,
        upload_id: str,
        part: int,
        expires_s: int = 3600,
        length: int | None = None,
    ) -> str | None:
        """A presigned PUT URL for one part (signed for exactly `length` bytes when
        given, so the bucket refuses any other size), or None when parts must go
        through upload_part (the local backend)."""

    @abstractmethod
    def upload_part(
        self, ref: Ref, name: str, upload_id: str, part: int, data: bytes
    ) -> str:
        """Upload one part directly; returns its etag, to pass to finish_upload as is."""

    # ---- deliveries

    def list_deliveries(self, vendor: str | None = None) -> list[Listed]:
        """Every delivery folder (complete or still uploading), sorted."""
        prefix = f"{DELIVERIES}/{vendor}/" if vendor else f"{DELIVERIES}/"
        found: dict[Ref, Listed] = {}
        for key, meta in self._list(prefix).items():
            parts = key.split("/", 3)
            if len(parts) < 4:
                continue
            try:
                ref = Ref(parts[1], parts[2])
            except ValueError:
                continue  # not something the dashboard could have created
            d = found.setdefault(ref, Listed(ref))
            if parts[3] == COMPLETE:
                d.marked = True
            else:
                d.files[parts[3]] = meta
        for d in found.values():
            if d.marked:
                manifest = (self.completion(d.ref) or {}).get("files", {})
                d.complete = d.files == {k: tuple(v) for k, v in manifest.items()}
        return [found[r] for r in sorted(found)]

    def _files(self, ref: Ref) -> dict[str, tuple[int, str]]:
        n = len(ref.prefix)
        items = self._list(ref.prefix).items()
        return {k[n:]: v for k, v in items if k[n:] != COMPLETE}

    def mark_complete(
        self, ref: Ref, info: dict, files: list[str]
    ) -> dict[str, tuple[int, str]]:
        """The upload is finished and `files` is the delivery, exactly: any other object
        left in it (e.g. by the version a correction replaces) is deleted, and a missing
        one is an error. Then snapshot the files (size, etag) into _complete.json; the
        watcher processes the delivery only while they match."""
        if self.uploads_in_progress(ref):
            raise UploadsInProgress(ref, self.uploads_in_progress(ref))
        wanted = {_file_name(f) for f in files}
        if not wanted:
            raise ValueError(f"{ref.prefix}: a delivery needs at least one file")
        missing = wanted - set(self._files(ref))
        if missing:
            raise ValueError(f"{ref.prefix} lacks {sorted(missing)}")
        for name in sorted(set(self._files(ref)) - wanted):
            self._delete(ref.file_key(name))
        files = self._files(ref)
        self._put(
            ref.prefix + COMPLETE,
            _json({"completed_at": now(), **info, "files": files}),
        )
        if self.uploads_in_progress(ref):  # one started meanwhile: withdraw the marker
            self._delete(ref.prefix + COMPLETE)
            raise UploadsInProgress(ref, self.uploads_in_progress(ref))
        return files

    def get_delivery(self, ref: Ref) -> Listed | None:
        """One delivery as it is now (a fresh listing of just its folder), or None."""
        found = [d for d in self.list_deliveries(ref.vendor) if d.ref == ref]
        return found[0] if found else None

    def completion(self, ref: Ref) -> dict | None:
        return _load(self._get(ref.prefix + COMPLETE))

    def download(self, listed: Listed, dest: Path) -> Path:
        """Exactly the delivery's files under dest/ (same relative names); returns dest.
        Downloads into a fresh sibling folder and swaps it in only once the snapshot is
        verified, so dest never holds a partial or stale mix. Raises DeliveryChanged
        unless the delivery is complete and unchanged throughout."""
        if not listed.complete:
            raise DeliveryChanged(f"{listed.ref.prefix} is not complete")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}")
        try:
            for name in listed.files:
                target = tmp / _file_name(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                self._download(listed.ref.file_key(name), target)
            if self._files(listed.ref) != listed.files or not self.completion(
                listed.ref
            ):
                raise DeliveryChanged(f"{listed.ref.prefix} changed while downloading")
            if dest.exists():
                shutil.rmtree(dest)
            tmp.rename(dest)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return dest

    # ---- status (watcher writes, dashboard reads)

    def put_status(self, ref: Ref, status: dict) -> None:
        self._put(ref.status_key, _json({**status, "updated_at": now()}))

    def get_status(self, ref: Ref) -> dict | None:
        return _load(self._get(ref.status_key))

    def statuses(self, vendor: str | None = None) -> dict[Ref, dict]:
        prefix = f"{PIPELINE}/status/{vendor}/" if vendor else f"{PIPELINE}/status/"
        out = {}
        for key in self._list(prefix):
            v, name = key.split("/")[-2:]
            if name.endswith(".json"):
                ref = Ref(v, name[: -len(".json")])
                out[ref] = _load(self._get(key)) or {}
        return out

    # ---- staff decisions (dashboard writes, watcher reads)

    def add_decision(
        self, ref: Ref, decision: str, who: str, comment: str = "", stage: str = ""
    ) -> dict:
        """Record a staff decision; `stage` is the stage_digest that was reviewed, so an
        approval applies to exactly that staged result."""
        if decision not in ("approve", "decline"):
            raise ValueError(f"decision {decision!r} is not approve or decline")
        if not VENDOR.fullmatch(who):
            raise ValueError(f"reviewer {who!r} is not a username")
        if stage and not re.fullmatch(r"[0-9a-f]{64}", stage):
            raise ValueError(f"stage {stage!r} is not a stage digest")
        rec = {"decision": decision, "who": who, "comment": comment, "at": now()}
        rec["stage"] = stage
        # The name carries the decision and stage, so one listing tells the dashboard
        # what is decided (latest_decisions) without reading every record.
        name = f"{rec['at']}-{who}-{decision}-{stage}.json"
        self._put(ref.decisions_prefix + name, _json(rec))
        return rec

    def latest_decisions(self, vendor: str | None = None) -> dict[Ref, dict]:
        """The newest decision per delivery ({at, who, decision, stage}, comment left
        out), from one listing of the decision names."""
        prefix = (
            f"{PIPELINE}/decisions/{vendor}/" if vendor else f"{PIPELINE}/decisions/"
        )
        out: dict[Ref, dict] = {}
        for key in sorted(self._list(prefix)):
            parts = key[len(f"{PIPELINE}/decisions/") :].split("/")
            m = _DECISION.fullmatch(parts[-1]) if len(parts) == 3 else None
            if m is None:
                continue
            try:
                ref = Ref(parts[0], parts[1])
            except ValueError:
                continue
            out[ref] = m.groupdict()  # sorted by time: the last one wins
        return out

    def decisions(self, ref: Ref) -> list[dict]:
        """Oldest first; the last one is in force."""
        keys = sorted(self._list(ref.decisions_prefix))
        return [d for d in (_load(self._get(k)) for k in keys) if d]

    # ---- staged results (watcher writes and reads; immutable, keyed by stage digest)

    def put_staged(self, ref: Ref, digest: str, folder: Path) -> None:
        """Upload a staged folder; its manifest (file -> size) goes last, so a stage
        counts as stored only once every file is in."""
        prefix = ref.staged_prefix(digest)
        folder = Path(folder)
        manifest = {}
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                rel = _file_name(p.relative_to(folder).as_posix())
                self._upload_file(prefix + rel, p)
                manifest[rel] = p.stat().st_size
        self._put(prefix + "_manifest.json", _json(manifest))

    def get_staged(self, ref: Ref, digest: str, dest: Path) -> Path:
        """Exactly the stored stage under dest/ (replacing it only once complete). The
        caller verifies it (stage.load checks every file against the pin, and the pin's
        own digest must equal `digest`)."""
        prefix = ref.staged_prefix(digest)
        manifest = _load(self._get(prefix + "_manifest.json"))
        if manifest is None:
            raise FileNotFoundError(f"no stored stage {prefix}")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}")
        try:
            for rel, size in manifest.items():
                target = tmp / _file_name(rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                self._download(prefix + rel, target)
                if target.stat().st_size != size:
                    raise DeliveryChanged(f"{prefix}{rel} does not match its manifest")
            if dest.exists():
                shutil.rmtree(dest)
            tmp.rename(dest)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return dest

    def has_staged(self, ref: Ref, digest: str) -> bool:
        return self._get(ref.staged_prefix(digest) + "_manifest.json") is not None

    def delete_staged(self, ref: Ref, digest: str) -> None:
        prefix = ref.staged_prefix(digest)
        self._delete(prefix + "_manifest.json")  # first: it stops counting as stored
        for key in self._list(prefix):
            self._delete(key)

    # ---- dashboard accounts

    def get_accounts(self) -> dict:
        return _load(self._get(ACCOUNTS)) or {}

    def put_accounts(self, accounts: dict) -> None:
        self._put(ACCOUNTS, _json(accounts))


def _json(obj) -> bytes:
    return json.dumps(obj, indent=1, sort_keys=True).encode()


def _load(data: bytes | None):
    return None if data is None else json.loads(data)


# ---------------------------------------------------------------- local directory


class LocalStore(Store):
    """The bucket as a directory: keys are paths under `root`. Multipart parts are kept
    under root/.uploads/ until finished."""

    upload_origins: list[str] = []  # parts go through the dashboard itself

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / PurePosixPath(key)

    def _list(self, prefix: str) -> dict[str, tuple[int, str]]:
        base = self._path(prefix.rstrip("/"))
        if not base.exists():
            return {}
        out = {}
        for p in sorted(base.rglob("*")):
            if p.is_file():
                key = p.relative_to(self.root).as_posix()
                out[key] = (p.stat().st_size, _md5(p, cached=True))
        return out

    def _get(self, key: str) -> bytes | None:
        p = self._path(key)
        return p.read_bytes() if p.is_file() else None

    def _put(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)  # readers never see a half-written record

    def _download(self, key: str, dest: Path) -> None:
        shutil.copyfile(self._path(key), dest)

    def _delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def _upload_file(self, key: str, path: Path) -> None:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        shutil.copyfile(path, tmp)
        tmp.replace(target)

    def _parts_dir(self, upload_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
            raise ValueError(f"unknown upload id {upload_id!r}")
        return self.root / ".uploads" / upload_id

    def _start(self, key: str) -> str:
        upload_id = hashlib.md5(f"{key}{now()}".encode()).hexdigest()
        d = self._parts_dir(upload_id)
        d.mkdir(parents=True)
        (d / "key").write_text(key)
        return upload_id

    def part_url(self, ref, name, upload_id, part, expires_s=3600, length=None) -> None:
        return None  # no presigning on disk: parts go through upload_part

    def upload_part(self, ref, name, upload_id, part, data) -> str:
        d = self._parts_dir(upload_id)
        if (d / "key").read_text() != ref.file_key(name):
            raise ValueError("upload id belongs to another file")
        (d / f"{part:05d}").write_bytes(data)
        return hashlib.md5(data).hexdigest()

    def _finish(self, key, upload_id, parts) -> None:
        d = self._parts_dir(upload_id)
        if (d / "key").read_text() != key:
            raise ValueError("upload id belongs to another file")
        for n, etag in sorted(parts):  # check every part before touching the target
            part = d / f"{n:05d}"
            if not part.is_file() or _md5(part) != etag:
                raise ValueError(f"part {n} is missing or does not match its etag")
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        with open(tmp, "wb") as out:
            for n, _ in sorted(parts):
                out.write((d / f"{n:05d}").read_bytes())
        tmp.replace(target)  # the old file stays whole until the new one is
        shutil.rmtree(d)

    def _abort(self, key, upload_id) -> None:
        shutil.rmtree(self._parts_dir(upload_id), ignore_errors=True)


_MD5_CACHE: dict[tuple, str] = {}


def _md5(p: Path, cached: bool = False) -> str:
    """A local file's etag. With `cached` (listings), a file of 8 MB or more is
    remembered while its identity, size and mtime stay the same, so listing a big local
    bucket does not re-read every byte each time; small files, and checks of parts
    rewritten in place, always read the bytes."""
    st = p.stat()
    key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    cached = cached and st.st_size >= 8 * 1024**2
    if cached and key in _MD5_CACHE:
        return _MD5_CACHE[key]
    h = hashlib.md5()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if cached:
        if len(_MD5_CACHE) > 100_000:
            _MD5_CACHE.clear()
        _MD5_CACHE[key] = h.hexdigest()
    return h.hexdigest()


# ---------------------------------------------------------------- R2


class R2Store(Store):
    """The `rldb` bucket on Cloudflare R2 (S3 API). Credentials as the rest of the repo:
    R2_ENDPOINT_URL / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY (~/.egoverse_env).

    `root` puts the whole layout under one prefix (e.g. "_test/<run>/" for a disposable
    namespace); the default is the bucket root."""

    def __init__(self, bucket: str = "rldb", client=None, root: str = ""):
        if client is None:
            from egomimic.utils.aws.aws_data_utils import get_boto3_s3_client

            client = get_boto3_s3_client()
        if root and not re.fullmatch(r"[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*/", root):
            raise ValueError(f"root {root!r} must be a relative prefix ending in '/'")
        self.s3 = client
        self.bucket = bucket
        self.root = root

    @property
    def upload_origins(self) -> list[str]:
        """Where a browser sends presigned parts (the page's CSP must allow them)."""
        from urllib.parse import urlsplit

        u = urlsplit(self.s3.meta.endpoint_url or "")
        if not u.hostname:
            return []
        return [f"{u.scheme}://{u.netloc}", f"{u.scheme}://*.{u.netloc}"]

    def _k(self, key: str) -> str:
        return self.root + key

    def _list(self, prefix: str) -> dict[str, tuple[int, str]]:
        out = {}
        n = len(self.root)
        for page in self.s3.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=self._k(prefix)
        ):
            for obj in page.get("Contents", []):
                out[obj["Key"][n:]] = (obj["Size"], obj["ETag"].strip('"'))
        return out

    def _get(self, key: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=self._k(key))[
                "Body"
            ].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def _put(self, key: str, data: bytes) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self._k(key),
            Body=data,
            ContentType="application/json",
        )

    def _download(self, key: str, dest: Path) -> None:
        self.s3.download_file(self.bucket, self._k(key), str(dest))

    def _delete(self, key: str) -> None:
        self.s3.delete_object(Bucket=self.bucket, Key=self._k(key))

    def _upload_file(self, key: str, path: Path) -> None:
        from boto3.s3.transfer import TransferConfig

        # As egomimic/utils/aws/aws_data_utils.py: 64 MiB parts, 16 at a time.
        config = TransferConfig(
            max_concurrency=16,
            multipart_threshold=64 * 1024 * 1024,
            multipart_chunksize=64 * 1024 * 1024,
        )
        self.s3.upload_file(str(path), self.bucket, self._k(key), Config=config)

    def _start(self, key: str) -> str:
        r = self.s3.create_multipart_upload(Bucket=self.bucket, Key=self._k(key))
        return r["UploadId"]

    def part_url(self, ref, name, upload_id, part, expires_s=3600, length=None) -> str:
        params = {
            "Bucket": self.bucket,
            "Key": self._k(ref.file_key(name)),
            "UploadId": upload_id,
            "PartNumber": part,
        }
        if length is not None:  # content-length becomes a signed header
            params["ContentLength"] = length
        return self.s3.generate_presigned_url(
            "upload_part", Params=params, ExpiresIn=expires_s
        )

    def upload_part(self, ref, name, upload_id, part, data) -> str:
        r = self.s3.upload_part(
            Bucket=self.bucket,
            Key=self._k(ref.file_key(name)),
            UploadId=upload_id,
            PartNumber=part,
            Body=data,
        )
        return r["ETag"]  # exactly as returned: completion must send it back unchanged

    def _finish(self, key, upload_id, parts) -> None:
        self.s3.complete_multipart_upload(
            Bucket=self.bucket,
            Key=self._k(key),
            UploadId=upload_id,
            MultipartUpload={
                "Parts": [{"PartNumber": n, "ETag": e} for n, e in sorted(parts)]
            },
        )

    def _abort(self, key, upload_id) -> None:
        from botocore.exceptions import ClientError

        try:
            self.s3.abort_multipart_upload(
                Bucket=self.bucket, Key=self._k(key), UploadId=upload_id
            )
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") != "NoSuchUpload":
                raise  # network / auth trouble: the caller must be able to retry
