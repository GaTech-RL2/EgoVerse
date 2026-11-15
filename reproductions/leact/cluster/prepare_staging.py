"""Map this task's legacy checkpoint path to its durable study storage."""
from __future__ import annotations

import argparse
import base64
import json
import re
import shlex
import zlib

import pexpect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow")
    parser.add_argument("role", choices=["self", "frontier", "astra"])
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.workflow):
        raise ValueError("Invalid workflow name")
    settings = json.dumps({"workflow": args.workflow, "role": args.role})
    code = f"""
import json, os, shutil
from pathlib import Path
settings = json.loads({settings!r})
assert os.environ.get('LEACT_WORKFLOW_ID') == settings['workflow']
assert os.environ.get('LEACT_ROLE') == settings['role']
home = Path('/mnt/amlfs-03/home/rpunamiya')
assert os.path.ismount(home)
assert shutil.disk_usage(home).free > 100 * 1024**3
study = home / 'leact/studies' / ('overnight-20261022-' + settings['role'])
target = study / 'checkpoint-staging'
target.mkdir(parents=True, exist_ok=True)
legacy = Path('/osmo/run/workspace') / ('astra-checkpoints' if settings['role'] == 'astra' else 'coldstart-checkpoints-' + settings['role'])
if legacy.is_symlink():
    assert legacy.resolve() == target.resolve(), 'Unexpected existing checkpoint link'
else:
    if legacy.exists():
        assert legacy.is_dir() and not any(legacy.iterdir()), 'Existing task checkpoints require inspection'
        legacy.rmdir()
    legacy.symlink_to(target, target_is_directory=True)
record = dict(settings, durable_staging=str(target), legacy_path=str(legacy),
              free_bytes=shutil.disk_usage(home).free)
(study / ('staging-' + settings['workflow'] + '.json')).write_text(json.dumps(record, indent=2) + '\\n')
print('CHECKPOINT_STAGING_READY ' + json.dumps(record), flush=True)
"""
    encoded = base64.b64encode(zlib.compress(code.encode())).decode()
    entry = "import base64,zlib;exec(zlib.decompress(base64.b64decode(" + repr(encoded) + ")))"
    remote = ("source /mnt/amlfs-07/shared/egoverse/rpunamiya/leact/runs/"
              "leact-leduc-qwen3-8b-7/source/emimic/bin/activate\npython -c " + shlex.quote(entry))
    child = pexpect.spawn("osmo", ["workflow", "exec", args.workflow, "train", "--entry",
                                  "bash -c " + shlex.quote(remote)],
                          encoding="utf-8", timeout=60, echo=False)
    try:
        child.expect(pexpect.EOF, timeout=60)
        output = child.before
        child.close()
        for line in output.splitlines():
            if line.startswith("CHECKPOINT_STAGING_READY "):
                record = json.loads(line.removeprefix("CHECKPOINT_STAGING_READY "))
                if record["workflow"] != args.workflow or record["role"] != args.role:
                    raise ValueError("Checkpoint staging receipt belongs to another task")
                print(json.dumps(record), flush=True)
                return
        raise RuntimeError("Checkpoint staging was not confirmed: " + output[-1800:])
    finally:
        if child.isalive():
            child.close(force=True)


if __name__ == "__main__":
    main()
