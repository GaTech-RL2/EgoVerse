import json
from pathlib import Path
import time

import pytest


@pytest.fixture
def native(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'cluster'))
    import efe_pear_primary_native
    return efe_pear_primary_native,{'finished':str(tmp_path/'finished.json'),
        'head_heartbeat':str(tmp_path/'head-heartbeat.json'),'owner_model_workflow':'new-model'}


def test_new_native_allocation_ignores_the_previous_model_completion_marker(native):
    module,spec=native
    Path(spec['finished']).write_text('{"completed":false}')
    Path(spec['head_heartbeat']).write_text(json.dumps({'workflow':'old-model','updated_unix':time.time()}))
    with pytest.raises(TimeoutError):module.wait_for_model_head(spec,timeout=0)


def test_native_allocation_starts_only_after_its_current_model_has_cleared_completion(native):
    module,spec=native
    Path(spec['head_heartbeat']).write_text(json.dumps({'workflow':'new-model','updated_unix':time.time()}))
    module.wait_for_model_head(spec,timeout=0)
    Path(spec['finished']).write_text('{"completed":false}')
    with pytest.raises(TimeoutError):module.wait_for_model_head(spec,timeout=0)
