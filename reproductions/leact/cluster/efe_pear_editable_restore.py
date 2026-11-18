"""Preserve native editable installation artifacts while rebuilding its venv."""
import hashlib
import os
from pathlib import Path
import shutil


def capture_editables(prefix, distributions, destination):
    prefix=Path(prefix).absolute();destination=Path(destination);saved={}
    for distribution in distributions:
        import json
        raw=distribution.read_text('direct_url.json')
        if not raw or not json.loads(raw).get('dir_info',{}).get('editable'):continue
        if distribution.files is None:raise ValueError('An editable installation has no file manifest')
        for entry in distribution.files:
            source=Path(os.path.abspath(distribution.locate_file(entry)))
            if not source.is_relative_to(prefix):raise ValueError('Editable installation artifact escapes its venv')
            if source.is_symlink():raise ValueError('Unexpected editable artifact symlink')
            if not source.is_file():raise ValueError('Editable installation artifact is missing')
            relative=source.relative_to(prefix);target=destination/relative
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            saved[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    return saved


def restore_editables(prefix, destination, saved):
    prefix=Path(prefix);destination=Path(destination)
    for relative,expected in saved.items():
        if Path(relative).is_absolute()or'..'in Path(relative).parts:raise ValueError('Unsafe editable artifact path')
        source=destination/relative
        if source.is_symlink()or hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
            raise ValueError('Saved editable artifact changed')
        target=prefix/relative
        if target.exists()and hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
            raise ValueError('A locked package conflicts with an editable installation artifact')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:raise ValueError('Editable restore readback failed')
