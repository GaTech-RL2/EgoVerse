"""Lossless per-image storage shared across complete multimodal histories."""
import hashlib
import json
from pathlib import Path
import re

FORMAT = 'shared-visual-tensors-v1'
_checked = {}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(batch, path):
    import torch
    path = Path(path)
    batch = dict(batch)
    if not all(isinstance(v, torch.Tensor) for v in batch.values()):
        raise ValueError('Only tensor payloads are supported')
    if 'pixel_values_videos' in batch:
        raise ValueError('Video tensor sharing has not been validated')
    pixels = batch.pop('pixel_values', None)
    blocks = []
    if pixels is not None:
        grids = batch['image_grid_thw']
        lengths = [int(row.prod()) for row in grids]
        if sum(lengths) != pixels.shape[0]:
            raise ValueError('Per-image patch boundaries differ from pixel rows')
        folder = path.parent.parent / 'tensor-blobs'
        folder.mkdir(parents=True, exist_ok=True)
        for value in pixels.split(lengths):
            # Clone: serializing a view otherwise retains the whole history's storage.
            value = value.detach().cpu().contiguous().clone()
            raw = value.view(torch.uint8).numpy().tobytes()
            header = json.dumps({'shape': list(value.shape), 'dtype': str(value.dtype)}, sort_keys=True).encode()
            key = hashlib.sha256(header + b'\0' + raw).hexdigest()
            blob = folder / (key + '.pt')
            if not blob.exists():
                temporary = blob.with_suffix('.pt.tmp')
                torch.save(value, temporary)
                temporary.replace(blob)
            metadata = {'file': blob.name, 'sha256': digest(blob), 'bytes': blob.stat().st_size,
                        'shape': list(value.shape), 'dtype': str(value.dtype)}
            blocks.append(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'tensor_storage': FORMAT, 'batch': batch, 'pixel_blocks': blocks}, path)


def expand(payload, path):
    import torch
    if payload.get('tensor_storage') != FORMAT:
        raise ValueError('Unknown shared tensor representation')
    batch = dict(payload['batch'])
    if not all(isinstance(v, torch.Tensor) for v in batch.values()) or 'pixel_values' in batch:
        raise ValueError('Invalid small tensor payload')
    blocks = []
    folder = Path(path).parent.parent / 'tensor-blobs'
    for record in payload['pixel_blocks']:
        if not re.fullmatch(r'[a-f0-9]{64}\.pt', record['file']):
            raise ValueError('Invalid visual blob name')
        blob = folder / record['file']
        stat = blob.stat()
        fingerprint = (stat.st_size, stat.st_mtime_ns, record['sha256'])
        if stat.st_size != record['bytes']:
            raise ValueError('Visual blob size changed')
        if _checked.get(str(blob)) != fingerprint:
            if digest(blob) != record['sha256']:
                raise ValueError('Visual blob checksum changed')
            _checked[str(blob)] = fingerprint
        value = torch.load(blob, map_location='cpu', weights_only=True)
        if not isinstance(value, torch.Tensor) or list(value.shape) != record['shape'] or str(value.dtype) != record['dtype']:
            raise ValueError('Visual blob tensor layout changed')
        blocks.append(value)
    if blocks:
        if len(blocks) != len(batch['image_grid_thw']):
            raise ValueError('Visual blob count differs from image grid')
        if any(value.shape[0] != int(grid.prod()) for value, grid in zip(blocks, batch['image_grid_thw'])):
            raise ValueError('Visual blob patch boundary changed')
        batch['pixel_values'] = torch.cat(blocks, dim=0)
    elif 'image_grid_thw' in batch and batch['image_grid_thw'].numel():
        raise ValueError('Image grid without pixel blocks')
    return batch
