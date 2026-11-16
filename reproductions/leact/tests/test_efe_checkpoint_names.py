import pytest

from reproductions.leact.cluster.efe_checkpoint_names import sync


def test_prefixed_aliases_leave_resume_state_and_model_weights_untouched(tmp_path):
    coverage=tmp_path/'coverage';pear=tmp_path/'pear';registry=tmp_path/'names'
    checkpoint=coverage/'warmstart/restart-states/checkpoint-688';checkpoint.mkdir(parents=True)
    manifest=checkpoint/'checkpoint-manifest.json';manifest.write_text('{"step":688}')
    weights=checkpoint/'adapter.bin';weights.write_bytes(b'unchanged weights')
    parent=coverage/'intermediate/warmstart-step496';(parent/'model').mkdir(parents=True)
    (parent/'checkpoint-pin.json').write_text('{}')
    result=sync(registry,coverage,pear)
    assert result['checkpoint_count']==2
    assert all(name.startswith('rp_jhh_') for name in result['checkpoints'])
    alias=registry/'rp_jhh_qwen36_moe_warmstart_step000688_adapter'
    assert alias.is_symlink() and alias.resolve()==checkpoint
    assert manifest.read_text()=='{"step":688}' and weights.read_bytes()==b'unchanged weights'
    assert sync(registry,coverage,pear)['checkpoint_count']==2


def test_partial_pear_checkpoints_and_alias_collisions_are_not_published(tmp_path):
    coverage=tmp_path/'coverage';pear=tmp_path/'pear';registry=tmp_path/'names'
    root=pear/'runs/qwen38-27b-base-pearplusplus/ckpt'
    (root/'global_step_1').mkdir(parents=True);(root/'global_step_2').mkdir()
    (root/'latest_checkpointed_iteration.txt').write_text('1')
    result=sync(registry,coverage,pear)
    assert list(result['checkpoints'])==['rp_jhh_qwen38_27b_base_pearplusplus_step000001']
    alias=registry/next(iter(result['checkpoints']));alias.unlink();alias.mkdir()
    with pytest.raises(ValueError):sync(registry,coverage,pear)
