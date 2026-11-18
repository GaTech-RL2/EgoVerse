from pathlib import Path

import pytest

from reproductions.leact.cluster.efe_pear_asset_closure import task_owned_files


def test_task_manifest_and_sidecars_survive_without_copying_unselected_tasks(tmp_path):
    parent=Path('scenes/YAM/eval_flywheel/T_gen')
    expected=set()
    for task in ['selected','held_out']:
        for name in ['asset_manifest.json','runtime/material.json','objects/cup.usda']:
            relative=parent/task/name;path=tmp_path/relative
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
            if task=='selected':expected.add(str(relative))
    assert task_owned_files(tmp_path,[str(parent/'selected/objects/cup.usda'),
        'isaac/Environments/Grid/default_environment.usd'])==expected


def test_task_asset_copy_rejects_links_outside_the_pinned_release(tmp_path):
    root=tmp_path/'assets';task=root/'scenes/YAM/eval_flywheel/T_gen/selected'
    task.mkdir(parents=True);outside=tmp_path/'outside.json';outside.write_text('{}')
    (task/'asset_manifest.json').symlink_to(outside)
    with pytest.raises(ValueError,match='regular files'):
        task_owned_files(root,['scenes/YAM/eval_flywheel/T_gen/selected/object.usda'])
