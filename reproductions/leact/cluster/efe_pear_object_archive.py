"""Private, streamed checkpoint archives with a manifest committed last.

An archive is usable only after every object has been downloaded and SHA256
verified. Multipart uploads do not require a second checkpoint-sized staging
directory. Credentials are read from a private node-local file, never logged.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from uuid import uuid4

MANIFEST = 'rp_jhh_archive_manifest.json'
PART_BYTES = 64 << 20


def configured_client(credential_file=None):
    import boto3
    from botocore.config import Config
    path = Path(credential_file or os.environ['PEAR_OBJECTSTORE_CREDENTIAL_FILE'])
    if path.stat().st_mode & 0o077:
        raise ValueError('Object-store credentials must be in a private node-local file')
    value = json.loads(path.read_text())
    return boto3.client('s3', endpoint_url=value['endpoint_url'],
        aws_access_key_id=value['access_key_id'], aws_secret_access_key=value['access_key'],
        region_name=value.get('region', 'us-east-1'), config=Config(
            s3={'addressing_style': 'path'}, request_checksum_calculation='when_required',
            response_checksum_validation='when_required', connect_timeout=15, read_timeout=180,
            retries={'mode': 'standard', 'max_attempts': 5}))


def split_uri(uri):
    from urllib.parse import urlsplit
    parsed = urlsplit(uri)
    if parsed.scheme != 's3' or not parsed.netloc or not parsed.path.strip('/'):
        raise ValueError('An archive requires s3://bucket/private-prefix')
    prefix = parsed.path.strip('/')
    if any(p in {'.', '..'} for p in prefix.split('/')):
        raise ValueError('Unsafe checkpoint prefix')
    return parsed.netloc, prefix


def _read_hash(client, bucket, key):
    response = client.get_object(Bucket=bucket, Key=key)
    body = response['Body']; digest = hashlib.sha256(); length = 0
    try:
        for block in iter(lambda: body.read(8 << 20), b''):
            digest.update(block); length += len(block)
    finally:
        body.close()
    return length, digest.hexdigest()


def _upload_file(client, bucket, key, path, *, part_bytes):
    before = path.stat(); digest = hashlib.sha256(); upload_id = None
    try:
        with path.open('rb') as stream:
            if before.st_size <= part_bytes:
                raw = stream.read(); digest.update(raw)
                client.put_object(Bucket=bucket, Key=key, Body=raw)
            else:
                upload_id = client.create_multipart_upload(Bucket=bucket, Key=key)['UploadId']
                parts = []
                while block := stream.read(part_bytes):
                    digest.update(block); number = len(parts) + 1
                    response = client.upload_part(Bucket=bucket, Key=key, UploadId=upload_id,
                        PartNumber=number, Body=block)
                    parts.append({'PartNumber': number, 'ETag': response['ETag']})
                client.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id,
                    MultipartUpload={'Parts': parts})
                upload_id = None
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Checkpoint changed during object-store publication')
        value = {'bytes': before.st_size, 'sha256': digest.hexdigest(), 'object_key': key}
        if _read_hash(client, bucket, key) != (value['bytes'], value['sha256']):
            raise ValueError('Checkpoint failed full object-store readback')
        return value
    finally:
        if upload_id is not None:
            client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)


def archive(source, destination, *, client=None, workers=4, part_bytes=PART_BYTES):
    source = Path(source).resolve(); bucket, prefix = split_uri(destination)
    if not source.is_dir():
        raise ValueError('Checkpoint source is not a directory')
    paths = []
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError('A checkpoint archive cannot contain symbolic links')
        if path.is_file():
            paths.append(path)
    if not paths:
        raise ValueError('Refusing to publish an empty checkpoint')
    client = client or configured_client()
    # This immutable per-attempt prefix is intentionally never reused. A
    # failed transfer cannot overwrite objects of a committed checkpoint.
    object_prefix = prefix + '/objects/' + uuid4().hex
    def publish(path):
        name = str(path.relative_to(source))
        value = _upload_file(client, bucket, object_prefix + '/' + name, path, part_bytes=part_bytes)
        print('PEAR_CHECKPOINT_OBJECT_VERIFIED ' + json.dumps({'file': name, 'bytes': value['bytes']}), flush=True)
        return name, value
    with ThreadPoolExecutor(max_workers=workers) as executor:
        files = dict(executor.map(publish, paths))
    manifest = {'files': files, 'bytes': sum(v['bytes'] for v in files.values()),
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'readback_verified': True,
        'storage': 'private_object_store', 'object_prefix': object_prefix,
        'commit_protocol': 'all_objects_uploaded_and_fully_read_back_before_manifest'}
    raw = (json.dumps(manifest, sort_keys=True) + '\n').encode()
    # No model file, optimizer or pointer is considered committed before this
    # final small object. Upstream saves each step to a distinct checkpoint URI.
    client.put_object(Bucket=bucket, Key=prefix + '/' + MANIFEST, Body=raw,
        ContentType='application/json', IfNoneMatch='*')
    if _read_hash(client, bucket, prefix + '/' + MANIFEST) != (len(raw), hashlib.sha256(raw).hexdigest()):
        raise ValueError('Checkpoint manifest failed object-store readback')
    return manifest


def restore(source, destination, *, client=None):
    bucket, prefix = split_uri(source); client = client or configured_client()
    result = client.get_object(Bucket=bucket, Key=prefix + '/' + MANIFEST)
    try:
        manifest = json.loads(result['Body'].read())
    finally:
        result['Body'].close()
    if manifest.get('readback_verified') is not True or not manifest.get('files'):
        raise ValueError('Checkpoint archive has no verified commit manifest')
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Restore target already exists')
    temporary = destination.with_name('.' + destination.name + '.restore-' + uuid4().hex)
    temporary.mkdir(parents=True)
    try:
        for name, value in manifest['files'].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or any(p in {'.', '..'} for p in relative.parts):
                raise ValueError('Unsafe checkpoint member')
            expected_key = manifest['object_prefix'] + '/' + name
            if value['object_key'] != expected_key or not expected_key.startswith(prefix + '/objects/'):
                raise ValueError('Checkpoint object is outside its committed prefix')
            target = temporary / name; target.parent.mkdir(parents=True, exist_ok=True)
            response = client.get_object(Bucket=bucket, Key=value['object_key'])
            body = response['Body']; digest = hashlib.sha256(); length = 0
            try:
                with target.open('xb') as stream:
                    for block in iter(lambda: body.read(8 << 20), b''):
                        stream.write(block); digest.update(block); length += len(block)
                    stream.flush(); os.fsync(stream.fileno())
            finally:
                body.close()
            if (length, digest.hexdigest()) != (value['bytes'], value['sha256']):
                raise ValueError('Restored checkpoint checksum mismatch')
        (temporary / MANIFEST).write_text(json.dumps(manifest, sort_keys=True) + '\n')
        temporary.replace(destination)
    except BaseException:
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest
