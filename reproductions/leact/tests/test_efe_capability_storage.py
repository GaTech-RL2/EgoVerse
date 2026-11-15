from pathlib import Path
from types import SimpleNamespace

from reproductions.leact.cluster.efe_capability_storage import install


def test_storage_adapter_preserves_sandbox_and_archives_exact_grades(tmp_path):
    root = tmp_path / 'local'; root.mkdir()
    archive = tmp_path / 'shared'; archive.mkdir()
    calls = []; receipts = []

    def sandbox(image, *, mounts=()):
        calls.append((image, mounts))
        return ['podman', 'run', '--network=none', '--user=65534:65534', image]

    def run(name, command, **kwargs):
        output = next(Path(source) for source, target, mode in calls[-1][1] if target == '/output')
        (output / 'results.json').write_bytes(b'{"complete":true}\n')

    evaluator = SimpleNamespace(ROOT=root, sandbox=sandbox, run=run,
                                write=lambda name, value: receipts.append((name, value)))
    install(evaluator)
    inputs = tmp_path / 'inputs'
    command = evaluator.sandbox('frozen-image', mounts=[(inputs, '/inputs', 'ro'), (archive, '/output', 'rw')])
    assert command == ['podman', 'run', '--network=none', '--user=65534:65534', 'frozen-image']
    assert calls[-1][1][0] == (inputs, '/inputs', 'ro')
    assert calls[-1][1][1][0].parent == root
    assert not (archive / 'results.json').exists()
    evaluator.run('grading', command)
    assert (archive / 'results.json').read_bytes() == b'{"complete":true}\n'
    assert receipts[0][0] == 'grading-storage.json'
