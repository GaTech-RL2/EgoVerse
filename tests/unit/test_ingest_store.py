"""The ingestion bucket API (egomimic/ingest/store.py) on both backends: a directory, and
R2 through an in-memory stand-in for the boto3 client (no network)."""

import hashlib
import io

import pytest

from egomimic.ingest.store import (
    COMPLETE,
    DeliveryChanged,
    LocalStore,
    R2Store,
    Ref,
    UploadRejected,
    UploadsInProgress,
    part_length,
)


class _NoSuchKey(Exception):
    pass


class FakeS3:
    """The boto3 S3 calls R2Store makes, against a dict."""

    class exceptions:
        NoSuchKey = _NoSuchKey

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.uploads: dict[str, dict] = {}
        self.fail_abort: Exception | None = None

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        s3 = self

        class P:
            def paginate(self, Bucket, Prefix):
                keys = sorted(k for k in s3.objects if k.startswith(Prefix))
                for i in range(0, len(keys), 2):  # two per page: exercises paging
                    yield {
                        "Contents": [
                            {
                                "Key": k,
                                "Size": len(s3.objects[k]),
                                "ETag": f'"{hashlib.md5(s3.objects[k]).hexdigest()}"',
                            }
                            for k in keys[i : i + 2]
                        ]
                    }

        return P()

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise _NoSuchKey(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[Key] = Body

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)

    def upload_file(self, Filename, Bucket, Key, Config=None):
        assert Config.multipart_chunksize == 64 * 1024 * 1024
        with open(Filename, "rb") as f:
            self.objects[Key] = f.read()

    def download_file(self, bucket, key, dest):
        with open(dest, "wb") as f:
            f.write(self.objects[key])

    def create_multipart_upload(self, Bucket, Key):
        uid = f"u{len(self.uploads)}"
        self.uploads[uid] = {"key": Key, "parts": {}}
        return {"UploadId": uid}

    def generate_presigned_url(self, op, Params, ExpiresIn):
        length = Params.get("ContentLength")
        return (
            f"https://r2.example/{Params['Key']}?part={Params['PartNumber']}"
            f"&ttl={ExpiresIn}" + (f"&length={length}" if length is not None else "")
        )

    def upload_part(self, Bucket, Key, UploadId, PartNumber, Body):
        assert self.uploads[UploadId]["key"] == Key
        etag = f'"{hashlib.md5(Body).hexdigest()}"'  # quoted, as S3 / R2 return it
        self.uploads[UploadId]["parts"][PartNumber] = (Body, etag)
        return {"ETag": etag}

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):
        up = self.uploads[UploadId]
        for p in MultipartUpload[
            "Parts"
        ]:  # R2 wants each part's ETag exactly as returned
            if up["parts"][p["PartNumber"]][1] != p["ETag"]:
                raise ValueError(
                    f"part {p['PartNumber']}: ETag {p['ETag']!r} does not match"
                )
        del self.uploads[UploadId]
        nums = [p["PartNumber"] for p in MultipartUpload["Parts"]]
        self.objects[Key] = b"".join(up["parts"][n][0] for n in nums)

    def abort_multipart_upload(self, Bucket, Key, UploadId):
        from botocore.exceptions import ClientError

        if self.fail_abort:
            raise self.fail_abort
        if UploadId not in self.uploads:
            raise ClientError(
                {"Error": {"Code": "NoSuchUpload"}}, "AbortMultipartUpload"
            )
        del self.uploads[UploadId]


@pytest.fixture(params=["local", "r2"])
def store(request, tmp_path):
    if request.param == "local":
        return LocalStore(tmp_path / "bucket")
    return R2Store(bucket="rldb", client=FakeS3())


def upload(store, ref, name, data: bytes, part_size: int = 4) -> None:
    uid = store.start_upload(ref, name)
    parts = []
    for i in range(0, len(data), part_size):
        n = i // part_size + 1
        parts.append((n, store.upload_part(ref, name, uid, n, data[i : i + part_size])))
    store.finish_upload(ref, name, uid, parts)


def complete_as_is(store, ref, info=None):
    """Mark complete as whatever the delivery holds now (for tests about other things)."""
    return store.mark_complete(ref, info or {}, list(store._files(ref)))


def test_a_delivery_is_listed_complete_only_after_mark_complete(store, tmp_path):
    ref = Ref("acme", "episode_0001")
    upload(store, ref, "episode.mcap", b"0123456789")
    upload(store, ref, "sub/annotation.mcap", b"labels")
    (d,) = store.list_deliveries()
    assert d.ref == ref and not d.complete
    assert set(d.files) == {"episode.mcap", "sub/annotation.mcap"}
    assert d.files["episode.mcap"][0] == 10

    complete_as_is(store, ref, {"by": "acme"})
    (d,) = store.list_deliveries()
    assert d.complete and store.completion(ref)["by"] == "acme"
    assert "_complete.json" not in d.files  # not part of the delivery's content

    out = store.download(d, tmp_path / "dl")
    assert (out / "episode.mcap").read_bytes() == b"0123456789"
    assert (out / "sub" / "annotation.mcap").read_bytes() == b"labels"


def test_fingerprint_changes_on_any_re_upload(store):
    ref = Ref("acme", "e1")
    upload(store, ref, "episode.mcap", b"first")
    before = store.list_deliveries()[0].fingerprint
    assert store.list_deliveries()[0].fingerprint == before  # stable
    upload(store, ref, "episode.mcap", b"second")
    assert store.list_deliveries()[0].fingerprint != before
    after = store.list_deliveries()[0].fingerprint
    upload(store, ref, "annotation.mcap", b"labels")  # a new file counts too
    assert store.list_deliveries()[0].fingerprint != after


def test_deliveries_filter_by_vendor(store):
    for v, e in (("acme", "a"), ("acme", "b"), ("other", "c")):
        upload(store, Ref(v, e), "episode.mcap", b"x")
    assert [d.ref.delivery_id for d in store.list_deliveries("acme")] == ["a", "b"]
    assert [d.ref.vendor for d in store.list_deliveries()] == ["acme", "acme", "other"]


def test_status_records(store):
    a, b = Ref("acme", "a"), Ref("other", "b")
    assert store.get_status(a) is None
    store.put_status(a, {"state": "ready", "frames": 60})
    store.put_status(b, {"state": "rejected"})
    got = store.get_status(a)
    assert got["state"] == "ready" and got["frames"] == 60 and got["updated_at"]
    assert set(store.statuses()) == {a, b}
    assert set(store.statuses("acme")) == {a}


def test_decisions_are_kept_in_order_and_validated(store):
    ref = Ref("acme", "a")
    store.add_decision(ref, "decline", "simar", "blurry")
    store.add_decision(ref, "approve", "ryan")
    got = store.decisions(ref)
    assert [d["decision"] for d in got] == ["decline", "approve"]
    assert got[0]["comment"] == "blurry" and got[-1]["who"] == "ryan"
    store.add_decision(ref, "approve", "ryan", stage="ab" * 32)
    assert store.decisions(ref)[-1]["stage"] == "ab" * 32
    with pytest.raises(ValueError):
        store.add_decision(ref, "approve", "ryan", stage="not-a-digest")
    with pytest.raises(ValueError):
        store.add_decision(ref, "maybe", "ryan")
    with pytest.raises(ValueError):
        store.add_decision(ref, "approve", "../ryan")


def test_accounts_round_trip(store):
    assert store.get_accounts() == {}
    store.put_accounts({"acme": {"role": "vendor", "vendor": "acme", "hash": "x"}})
    assert store.get_accounts()["acme"]["role"] == "vendor"


@pytest.mark.parametrize(
    "vendor, delivery_id",
    [("Acme", "a"), ("acme/x", "a"), ("acme", "../b"), ("acme", ""), ("acme", "a/b")],
)
def test_refs_cannot_escape_their_folder(vendor, delivery_id):
    with pytest.raises(ValueError):
        Ref(vendor, delivery_id)


@pytest.mark.parametrize("name", ["../x", "/abs", "a/../b", "", "_complete.json"])
def test_file_names_cannot_escape_or_fake_completion(store, name):
    with pytest.raises(ValueError):
        store.start_upload(Ref("acme", "a"), name)


def test_part_urls_presign_on_r2_and_go_direct_locally(tmp_path):
    ref = Ref("acme", "a")
    r2 = R2Store(client=FakeS3())
    url = r2.part_url(ref, "episode.mcap", r2.start_upload(ref, "episode.mcap"), 3)
    assert url.startswith("https://") and "deliveries/acme/a/episode.mcap" in url
    local = LocalStore(tmp_path / "b")
    assert local.part_url(ref, "x", local.start_upload(ref, "x"), 1) is None


def test_local_uploads_check_their_file_and_parts(tmp_path):
    s = LocalStore(tmp_path / "b")
    a, b = Ref("acme", "a"), Ref("acme", "b")
    uid = s.start_upload(a, "episode.mcap")
    with pytest.raises(ValueError):
        s.upload_part(b, "episode.mcap", uid, 1, b"x")  # someone else's upload id
    etag = s.upload_part(a, "episode.mcap", uid, 1, b"data")
    with pytest.raises(ValueError):
        s.finish_upload(a, "episode.mcap", uid, [(1, "not-the-etag")])
    s.finish_upload(a, "episode.mcap", uid, [(1, etag)])
    s.abort_upload(a, "x", s.start_upload(a, "x"))
    assert [d.files for d in s.list_deliveries()] == [{"episode.mcap": (4, etag)}]


def test_completion_is_a_snapshot_new_uploads_clear_it(store, tmp_path):
    ref = Ref("acme", "e1")
    upload(store, ref, "episode.mcap", b"old episode")
    upload(store, ref, "annotation.mcap", b"old labels")
    complete_as_is(store, ref)
    assert store.list_deliveries()[0].complete

    # A correction starts: the delivery is no longer complete, even mid-upload.
    uid = store.start_upload(ref, "episode.mcap")
    (d,) = store.list_deliveries()
    assert not d.complete and not d.marked
    with pytest.raises(DeliveryChanged):
        store.download(d, tmp_path / "dl")
    store.finish_upload(
        ref,
        "episode.mcap",
        uid,
        [(1, store.upload_part(ref, "episode.mcap", uid, 1, b"new"))],
    )
    assert not store.list_deliveries()[0].complete  # not until marked again
    complete_as_is(store, ref)
    (d,) = store.list_deliveries()
    assert d.complete and d.files["episode.mcap"][0] == 3


def test_files_that_drift_from_the_snapshot_are_not_complete(store):
    ref = Ref("acme", "e1")
    upload(store, ref, "episode.mcap", b"episode")
    complete_as_is(store, ref)
    store._put(ref.file_key("episode.mcap"), b"changed behind the dashboard's back")
    (d,) = store.list_deliveries()
    assert d.marked and not d.complete


def test_download_catches_a_change_while_downloading(store, tmp_path):
    ref = Ref("acme", "e1")
    upload(store, ref, "a.mcap", b"aaaa")
    upload(store, ref, "b.mcap", b"bbbb")
    complete_as_is(store, ref)
    (d,) = store.list_deliveries()
    real = store._download

    def racing(key, dest):
        real(key, dest)
        store._put(ref.file_key("b.mcap"), b"replaced mid-download")

    store._download = racing
    with pytest.raises(DeliveryChanged):
        store.download(d, tmp_path / "dl")


def test_marking_an_empty_delivery_complete_is_refused(store):
    with pytest.raises(ValueError):
        store.mark_complete(Ref("acme", "nothing"), {}, [])


def test_r2_completion_sends_part_etags_back_unchanged():
    s3 = FakeS3()
    store = R2Store(client=s3)
    ref = Ref("acme", "a")
    upload(store, ref, "episode.mcap", b"0123456789")  # FakeS3 rejects altered ETags
    assert s3.objects["deliveries/acme/a/episode.mcap"] == b"0123456789"
    uid = store.start_upload(ref, "x")
    etag = store.upload_part(ref, "x", uid, 1, b"data")
    assert etag.startswith('"')  # quoted, exactly as boto3 returned it
    with pytest.raises(ValueError):
        store.finish_upload(ref, "x", uid, [(1, etag.strip('"'))])


def test_failed_local_completion_keeps_the_previous_file(tmp_path):
    s = LocalStore(tmp_path / "b")
    ref = Ref("acme", "a")
    upload(s, ref, "episode.mcap", b"the good original")
    uid = s.start_upload(ref, "episode.mcap")
    e1 = s.upload_part(ref, "episode.mcap", uid, 1, b"new start")
    s.upload_part(ref, "episode.mcap", uid, 2, b"new end")
    with pytest.raises(ValueError):  # part 2's etag is wrong, part 3 is missing
        s.finish_upload(ref, "episode.mcap", uid, [(1, e1), (2, "bad"), (3, "x")])
    assert (
        tmp_path / "b/deliveries/acme/a/episode.mcap"
    ).read_bytes() == b"the good original"


def test_completion_waits_for_unfinished_uploads(store):
    ref = Ref("acme", "e1")
    upload(store, ref, "episode.mcap", b"episode")
    uid = store.start_upload(ref, "annotation.mcap")
    assert [u["name"] for u in store.uploads_in_progress(ref)] == ["annotation.mcap"]
    with pytest.raises(UploadsInProgress):
        complete_as_is(store, ref)
    store.abort_upload(ref, "annotation.mcap", uid)
    assert store.uploads_in_progress(ref) == []
    complete_as_is(store, ref)  # nothing unfinished any more
    assert store.list_deliveries()[0].complete


def test_an_upload_starting_during_completion_withdraws_the_marker(store):
    ref = Ref("acme", "e1")
    upload(store, ref, "episode.mcap", b"episode")
    real_put = store._put

    def put_then_race(key, data):
        real_put(key, data)
        if key.endswith(COMPLETE):  # another request starts an upload right now
            store._put = real_put
            store.start_upload(ref, "episode.mcap")

    store._put = put_then_race
    with pytest.raises(UploadsInProgress):
        complete_as_is(store, ref)
    (d,) = store.list_deliveries()
    assert not d.marked and not d.complete


def test_downloads_replace_dest_with_exactly_the_snapshot(store, tmp_path):
    ref = Ref("acme", "e1")
    upload(store, ref, "episode.mcap", b"episode")
    upload(store, ref, "old.mcap", b"dropped later")
    complete_as_is(store, ref)
    dest = tmp_path / "work" / "e1"
    store.download(store.list_deliveries()[0], dest)
    assert sorted(p.name for p in dest.iterdir()) == ["episode.mcap", "old.mcap"]

    store._delete(ref.file_key("old.mcap"))  # the delivery no longer has this file
    complete_as_is(store, ref)
    store.download(store.list_deliveries()[0], dest)  # same folder, retried
    assert sorted(p.name for p in dest.iterdir()) == ["episode.mcap"]
    assert sorted(p.name for p in dest.parent.iterdir()) == ["e1"]  # no temp left


def test_a_failed_download_leaves_dest_untouched(store, tmp_path):
    ref = Ref("acme", "e1")
    upload(store, ref, "a.mcap", b"aaaa")
    complete_as_is(store, ref)
    dest = tmp_path / "work" / "e1"
    store.download(store.list_deliveries()[0], dest)
    (d,) = store.list_deliveries()
    real = store._download

    def racing(key, target):
        real(key, target)
        store._put(ref.file_key("a.mcap"), b"replaced mid-download")

    store._download = racing
    with pytest.raises(DeliveryChanged):
        store.download(d, dest)
    assert (dest / "a.mcap").read_bytes() == b"aaaa"  # the last good copy
    assert sorted(p.name for p in dest.parent.iterdir()) == ["e1"]


def test_a_failed_start_leaves_nothing_behind(store):
    ref = Ref("acme", "e1")
    upload(store, ref, "episode.mcap", b"episode")

    def boom(key):
        raise ConnectionError("R2 is down")

    real, store._start = store._start, boom
    with pytest.raises(ConnectionError):
        store.start_upload(ref, "annotation.mcap")
    store._start = real
    assert store.uploads_in_progress(ref) == []
    complete_as_is(store, ref)  # not stranded


def test_a_failed_tracking_write_aborts_the_multipart_upload(store):
    ref = Ref("acme", "e1")
    aborted = []
    real_put, real_abort = store._put, store._abort

    def put(key, data):
        if "/uploads/" in key and "pending-" not in key:
            raise ConnectionError("lost the record write")
        real_put(key, data)

    def abort(key, upload_id):
        aborted.append(key)
        real_abort(key, upload_id)

    store._put, store._abort = put, abort
    with pytest.raises(ConnectionError):
        store.start_upload(ref, "episode.mcap")
    assert aborted == ["deliveries/acme/e1/episode.mcap"]
    assert store.uploads_in_progress(ref) == []


def test_a_stranded_record_is_listed_and_can_be_cleared(store):
    ref = Ref("acme", "e1")
    upload(store, ref, "episode.mcap", b"episode")
    # A dashboard process died between writing its pending record and starting.
    store._put(
        ref.record_key("pending-" + "a" * 32), b'{"name": "x.mcap", "state": "pending"}'
    )
    (rec,) = store.uploads_in_progress(ref)
    assert (rec["record"], rec["state"]) == ("pending-" + "a" * 32, "pending")
    with pytest.raises(UploadsInProgress) as e:
        complete_as_is(store, ref)
    assert rec["record"] in str(e.value)
    store.clear_upload(ref, rec["record"])
    complete_as_is(store, ref)


def test_finished_upload_whose_record_was_not_deleted_can_be_cleared(store):
    ref = Ref("acme", "e1")
    uid = store.start_upload(ref, "episode.mcap")
    etag = store.upload_part(ref, "episode.mcap", uid, 1, b"episode")
    real_delete = store._delete

    def delete(key):
        if "/uploads/" in key:
            raise ConnectionError("lost the record delete")
        real_delete(key)

    store._delete = delete
    with pytest.raises(ConnectionError):
        store.finish_upload(ref, "episode.mcap", uid, [(1, etag)])
    store._delete = real_delete
    (rec,) = store.uploads_in_progress(ref)  # the file is in, the record is stale
    assert rec["state"] == "uploading" and rec["name"] == "episode.mcap"
    store.clear_upload(ref, rec["record"])  # abort finds nothing: that is fine
    complete_as_is(store, ref)
    (d,) = store.list_deliveries()
    assert d.complete and d.files["episode.mcap"][0] == len(b"episode")


def test_record_ids_cannot_reach_outside_the_uploads_folder():
    with pytest.raises(ValueError):
        Ref("acme", "e1").record_key("../../status/acme/e1")


def test_upload_delivery_from_a_lab_machine_matches_a_vendor_upload(store, tmp_path):
    folder = tmp_path / "episode_0001"
    (folder / "sub").mkdir(parents=True)
    (folder / "episode.mcap").write_bytes(b"episode bytes")
    (folder / "sub" / "annotation.mcap").write_bytes(b"labels")
    ref = Ref("acme", "episode_0001")
    manifest = store.upload_delivery(ref, folder, {"by": "lab-cli"})
    assert set(manifest) == {"episode.mcap", "sub/annotation.mcap"}
    (d,) = store.list_deliveries()
    assert d.complete and store.uploads_in_progress(ref) == []
    assert store.completion(ref)["by"] == "lab-cli"
    out = store.download(d, tmp_path / "dl")
    assert (out / "sub" / "annotation.mcap").read_bytes() == b"labels"


def test_upload_file_follows_the_same_rules_as_vendor_uploads(store, tmp_path):
    ref = Ref("acme", "e1")
    f = tmp_path / "episode.mcap"
    f.write_bytes(b"v1")
    store.upload_file(ref, f)
    complete_as_is(store, ref)
    f.write_bytes(b"v2, a correction")
    store.upload_file(ref, f)  # clears completion, like any upload
    (d,) = store.list_deliveries()
    assert not d.marked and d.files["episode.mcap"][0] == len(b"v2, a correction")

    def boom(key, path):
        raise ConnectionError("network")

    store._upload_file = boom
    with pytest.raises(ConnectionError):
        store.upload_file(ref, f)
    assert store.uploads_in_progress(ref) == []  # a failure leaves nothing blocking


def test_upload_delivery_refuses_an_empty_folder(store, tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        store.upload_delivery(Ref("acme", "e1"), tmp_path / "empty", {})


def test_a_correction_that_drops_a_file_really_drops_it(store, tmp_path):
    v1 = tmp_path / "v1"
    v1.mkdir()
    (v1 / "episode.mcap").write_bytes(b"episode v1")
    (v1 / "annotation.mcap").write_bytes(b"labels that turn out wrong")
    ref = Ref("acme", "e1")
    store.upload_delivery(ref, v1, {})
    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / "episode.mcap").write_bytes(
        b"episode v2"
    )  # annotation.mcap removed on purpose
    manifest = store.upload_delivery(ref, v2, {})
    assert set(manifest) == {"episode.mcap"}
    (d,) = store.list_deliveries()
    assert d.complete and set(d.files) == {"episode.mcap"}


def test_finalizing_declares_the_exact_files(store):
    ref = Ref("acme", "e1")
    upload(store, ref, "episode.mcap", b"episode")
    upload(store, ref, "stray.bin", b"left from an old version")
    with pytest.raises(ValueError):  # a declared file was never uploaded
        store.mark_complete(ref, {}, ["episode.mcap", "annotation.mcap"])
    assert not store.list_deliveries()[0].marked
    store.mark_complete(ref, {}, ["episode.mcap"])
    (d,) = store.list_deliveries()
    assert d.complete and set(d.files) == {"episode.mcap"}  # the stray one is gone


def test_remove_file_uncompletes_the_delivery(store):
    ref = Ref("acme", "e1")
    upload(store, ref, "episode.mcap", b"episode")
    upload(store, ref, "annotation.mcap", b"labels")
    complete_as_is(store, ref)
    store.remove_file(ref, "annotation.mcap")
    (d,) = store.list_deliveries()
    assert not d.marked and set(d.files) == {"episode.mcap"}


def test_clear_upload_keeps_the_record_when_the_abort_really_fails():
    from botocore.exceptions import ClientError

    s3 = FakeS3()
    store = R2Store(client=s3)
    ref = Ref("acme", "e1")
    store.start_upload(ref, "episode.mcap")
    (rec,) = store.uploads_in_progress(ref)
    s3.fail_abort = ClientError(
        {"Error": {"Code": "AccessDenied"}}, "AbortMultipartUpload"
    )
    with pytest.raises(ClientError):
        store.clear_upload(ref, rec["record"])
    assert store.uploads_in_progress(ref)  # kept: retry later
    s3.fail_abort = ConnectionError("network")
    with pytest.raises(ConnectionError):
        store.clear_upload(ref, rec["record"])
    s3.fail_abort = None
    store.clear_upload(ref, rec["record"])  # now it goes through
    assert store.uploads_in_progress(ref) == []


def test_r2_root_namespaces_every_key():
    s3 = FakeS3()
    store = R2Store(client=s3, root="_test/run1/")
    ref = Ref("acme", "e1")
    upload(store, ref, "episode.mcap", b"episode")
    store.mark_complete(ref, {}, ["episode.mcap"])
    store.put_status(ref, {"state": "ready"})
    assert all(k.startswith("_test/run1/") for k in s3.objects), sorted(s3.objects)
    (d,) = store.list_deliveries()  # keys come back relative to the root
    assert d.ref == ref and d.complete and store.get_status(ref)["state"] == "ready"
    url = store.part_url(ref, "x", store.start_upload(ref, "x"), 1)
    assert "_test/run1/deliveries/acme/e1/x" in url
    with pytest.raises(ValueError):
        R2Store(client=s3, root="/abs/")


def test_staged_results_round_trip_by_digest(store, tmp_path):
    ref = Ref("acme", "e1")
    digest = "ab" * 32
    folder = tmp_path / "staged"
    (folder / "videos").mkdir(parents=True)
    (folder / "stage.json").write_text("{}")
    (folder / "videos" / "front_1.mp4").write_bytes(b"video bytes")
    assert not store.has_staged(ref, digest)
    store.put_staged(ref, digest, folder)
    assert store.has_staged(ref, digest)
    out = store.get_staged(ref, digest, tmp_path / "back")
    assert (out / "videos" / "front_1.mp4").read_bytes() == b"video bytes"
    store.delete_staged(ref, digest)
    assert not store.has_staged(ref, digest)
    assert not store._list(ref.staged_prefix(digest))
    with pytest.raises(FileNotFoundError):
        store.get_staged(ref, digest, tmp_path / "again")
    with pytest.raises(ValueError):
        ref.staged_prefix("../../status")


def test_a_stage_without_its_manifest_is_not_stored(store, tmp_path):
    ref = Ref("acme", "e1")
    digest = "cd" * 32
    folder = tmp_path / "staged"
    folder.mkdir()
    (folder / "steps.parquet").write_bytes(b"steps")
    real = store._put

    def no_manifest(key, data):  # the upload stopped before its manifest
        if not key.endswith("_manifest.json"):
            real(key, data)

    store._put = no_manifest
    store.put_staged(ref, digest, folder)
    store._put = real
    assert not store.has_staged(ref, digest)


def test_a_staged_download_that_does_not_match_its_manifest_fails(store, tmp_path):
    ref = Ref("acme", "e1")
    digest = "ef" * 32
    folder = tmp_path / "staged"
    folder.mkdir()
    (folder / "steps.parquet").write_bytes(b"steps")
    store.put_staged(ref, digest, folder)
    store._put(ref.staged_prefix(digest) + "steps.parquet", b"truncated")
    dest = tmp_path / "dest"
    with pytest.raises(DeliveryChanged):
        store.get_staged(ref, digest, dest)
    assert not dest.exists()


# ---- size-bound uploads (the dashboard's): declared size, part size and part count


def bound(store, ref, data: bytes, part_size: int = 4):
    uid = store.start_upload(ref, "episode.mcap", size=len(data), part_size=part_size)
    return uid, store.upload_record(ref, uid)


def test_a_size_bound_upload_pins_its_size_and_parts(store):
    ref = Ref("acme", "a")
    uid, rec = bound(store, ref, b"0123456789")
    assert (rec["size"], rec["part_size"], rec["parts"]) == (10, 4, 3)
    assert [part_length(rec, n) for n in (1, 2, 3)] == [4, 4, 2]
    for n in (0, 4):
        with pytest.raises(UploadRejected):
            part_length(rec, n)
    parts = [
        (n, store.upload_part(ref, "episode.mcap", uid, n, b"0123456789"[i : i + 4]))
        for n, i in ((1, 0), (2, 4), (3, 8))
    ]
    store.finish_upload(ref, "episode.mcap", uid, parts)
    assert store.uploads_in_progress(ref) == []


def test_a_size_bound_upload_needs_exactly_its_parts(store):
    ref = Ref("acme", "a")
    uid, _ = bound(store, ref, b"0123456789")
    e1 = store.upload_part(ref, "episode.mcap", uid, 1, b"0123")
    e2 = store.upload_part(ref, "episode.mcap", uid, 2, b"4567")
    e3 = store.upload_part(ref, "episode.mcap", uid, 3, b"89")
    e4 = store.upload_part(ref, "episode.mcap", uid, 4, b"xxxx")
    with pytest.raises(UploadRejected, match="exactly parts 1..3"):
        store.finish_upload(ref, "episode.mcap", uid, [(1, e1), (2, e2)])  # missing
    with pytest.raises(UploadRejected, match="exactly parts 1..3"):
        store.finish_upload(  # an extra part
            ref, "episode.mcap", uid, [(1, e1), (2, e2), (3, e3), (4, e4)]
        )
    assert len(store.uploads_in_progress(ref)) == 1  # still open: the client can fix it
    store.finish_upload(ref, "episode.mcap", uid, [(1, e1), (2, e2), (3, e3)])


def test_a_size_bound_upload_of_the_wrong_size_is_removed(store):
    ref = Ref("acme", "a")
    uid, _ = bound(store, ref, b"0123456789")
    parts = [
        (1, store.upload_part(ref, "episode.mcap", uid, 1, b"0123")),
        (
            2,
            store.upload_part(ref, "episode.mcap", uid, 2, b"4567" * 1000),
        ),  # oversized
        (3, store.upload_part(ref, "episode.mcap", uid, 3, b"89")),
    ]
    with pytest.raises(UploadRejected, match="10 declared; removed"):
        store.finish_upload(ref, "episode.mcap", uid, parts)
    assert "episode.mcap" not in store._files(ref)
    assert store.uploads_in_progress(ref) == []


def test_part_links_are_signed_for_their_exact_length():
    ref = Ref("acme", "a")
    r2 = R2Store(client=FakeS3())
    uid = r2.start_upload(ref, "episode.mcap", size=10, part_size=4)
    assert r2.part_url(ref, "episode.mcap", uid, 3, length=2).endswith("&length=2")


def test_real_presigned_links_sign_the_content_length():
    """With botocore itself: content-length is a signed header, so R2 refuses a part
    of any other size."""
    import urllib.parse

    import boto3

    s3 = boto3.client(
        "s3", endpoint_url="https://acct.r2.cloudflarestorage.com",
        aws_access_key_id="x", aws_secret_access_key="y", region_name="auto",
    )  # fmt: skip
    r2 = R2Store(client=s3)
    url = r2.part_url(Ref("acme", "a"), "episode.mcap", "uid", 1, length=1234)
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    assert q["X-Amz-SignedHeaders"] == ["content-length;host"]
