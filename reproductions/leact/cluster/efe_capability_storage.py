"""Storage-only launcher for the frozen capability evaluator.

Keep generation and scorer source bytes unchanged. Redirect only the scorer's
output mount to local scratch, then archive it using the workflow's owning UID.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil


def install(evaluator):
    original_sandbox = evaluator.sandbox
    original_run = evaluator.run
    destinations = []

    def sandbox(image, *, mounts=()):
        redirected = []
        for source, target, mode in mounts:
            if target == '/output':
                scratch = evaluator.ROOT / 'grader-output-local'
                scratch.mkdir(exist_ok=False)
                scratch.chmod(0o777)
                destinations.append((scratch, Path(source)))
                source = scratch
            redirected.append((source, target, mode))
        return original_sandbox(image, mounts=redirected)

    def run(name, command, **kwargs):
        result = original_run(name, command, **kwargs)
        if name == 'grading':
            if len(destinations) != 1:
                raise ValueError('Expected exactly one redirected grading output')
            scratch, archive = destinations[0]
            shutil.copytree(scratch, archive, dirs_exist_ok=True)
            evaluator.write('grading-storage.json', {
                'scratch': str(scratch), 'archive': str(archive),
                'change': 'Only output storage changed. Generation, scorer source, image, inputs and scoring protocol are unchanged.'})
        return result

    evaluator.sandbox = sandbox
    evaluator.run = run


def main():
    spec = importlib.util.spec_from_file_location('efe_frozen_evaluator', '/tmp/efe-capability-runner.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    install(evaluator)
    evaluator.main()


if __name__ == '__main__':
    main()
