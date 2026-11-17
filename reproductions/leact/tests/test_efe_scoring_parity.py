import pytest

from reproductions.leact.cluster.efe_train_qwen36_lora import selection_reference_required


def test_parent_selection_reference_is_required_before_first_update():
    for mode in ['smoke','train','scoring-check']:
        assert selection_reference_required(mode,0,True)


def test_resumed_adapter_uses_its_committed_replay_instead_of_untouched_parent_score():
    assert not selection_reference_required('train',96,True)
    with pytest.raises(ValueError,match='committed full-model loss'):
        selection_reference_required('train',96,False)
