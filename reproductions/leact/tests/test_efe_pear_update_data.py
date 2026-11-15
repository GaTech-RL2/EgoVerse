"""Guard whole-group token accounting before an expensive real policy update."""
from pathlib import Path
import pytest

from reproductions.leact.cluster.efe_pear_update_data import schedule_rows,captured_path


def rows(count):
    return [{'id':str(i),'tokens':100+13*i,'generated_tokens':1+i} for i in range(count)]


@pytest.mark.parametrize('count',[1,8,9,511,512])
def test_every_native_call_once_and_padding_has_no_gradient_mass(count):
    source=rows(count);scheduled,total=schedule_rows(source)
    assert len(scheduled)%8==0
    assert sorted(r['id'] for r in scheduled if r['weight'])==sorted(r['id'] for r in source)
    assert total==sum(r['generated_tokens'] for r in source)
    assert sum(r['generated_tokens']*r['weight'] for r in scheduled)==total
    # Simulate FSDP's rank-mean for nonuniform per-call gradients. Scaling by
    # world/global_tokens must equal a direct global token mean, including pads.
    rank_sums=[sum((int(r['id'])-3)*r['generated_tokens']*r['weight']*8/total
        for r in scheduled[rank::8]) for rank in range(8)]
    expected=sum((int(r['id'])-3)*r['generated_tokens'] for r in source)/total
    assert sum(rank_sums)/8==pytest.approx(expected)


def test_duplicate_call_rejected():
    with pytest.raises(ValueError,match='duplicate'):schedule_rows(rows(8)+rows(1))


@pytest.mark.parametrize('count',[0,-1])
def test_missing_target_tokens_rejected(count):
    with pytest.raises(ValueError,match='generated tokens'):
        schedule_rows([{'id':'broken','tokens':20,'generated_tokens':count}])


def test_sampler_paths_cannot_escape_group(tmp_path):
    source=tmp_path/'group';source.mkdir()
    assert captured_path(source,{'file':'sampler/a.json.gz'})==source/'sampler/a.json.gz'
    with pytest.raises(ValueError,match='escaped'):
        captured_path(source,{'file':'../other-group/a.json.gz'})
    (source/'link').symlink_to(tmp_path)
    with pytest.raises(ValueError,match='escaped'):
        captured_path(source,{'file':'link/a.json.gz'})
