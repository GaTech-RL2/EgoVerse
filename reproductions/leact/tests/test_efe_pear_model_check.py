import pytest

from reproductions.leact.cluster.efe_pear_model_check import unused_checkpoint_auxiliaries


def test_only_explicitly_unused_mtp_checkpoint_weights_can_be_omitted():
    active={'model.visual.weight','model.language_model.layers.0.mlp.experts.gate_up_proj'}
    auxiliary={'mtp.fc.weight','mtp.layers.0.self_attn.q_proj.weight'}
    assert unused_checkpoint_auxiliaries(active|auxiliary,active,[r'^mtp.*'])==sorted(auxiliary)
    with pytest.raises(ValueError,match='Parent weights omitted'):
        unused_checkpoint_auxiliaries(active|auxiliary,active,[])
    # Even another ignored-key pattern may not hide missing vision weights.
    with pytest.raises(ValueError,match='model.visual.weight'):
        unused_checkpoint_auxiliaries(active|auxiliary,active-{'model.visual.weight'},
                                      [r'^mtp.*',r'^model.visual.*'])
    with pytest.raises(ValueError,match='experts'):
        unused_checkpoint_auxiliaries(active|auxiliary,{'model.visual.weight'},[r'^mtp.*'])
