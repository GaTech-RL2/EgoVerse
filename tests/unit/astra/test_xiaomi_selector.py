import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from astra_reversal.complex_manipulation import xiaomi_teacher as teacher
from astra_reversal.complex_manipulation.xiaomi_interventions import (
    add_at_slots, blend_images, flow_integrate, perturb_images, text_slots,
)


def request():
    frame = {"origin": "current", "step": 0, "state": [0.]*14,
             "images": {k: teacher.wire.png_wire(np.zeros((16,16,3),np.uint8)) for k in ("left","right","wrist")}}
    return teacher.build_request(episode_id="case0-astra1", request_index=0, step=0,
        snapshots=[frame], context={"original_instruction":"Pack lunch.","remaining_calls_including_this":16})


def proposal(**kwargs):
    return {**copy.deepcopy(teacher.DEFAULTS), "method":"native", "observed_evidence":"Object on counter.",
            "completion_signal":"Object placed in container.","next_review_controls":128, **kwargs}


def test_selector_keeps_actual_images_and_14d_state_on_wire_and_binds_identity():
    r = request()
    before = copy.deepcopy(r)
    teacher._validate_request(r)
    result = teacher.parse_proposal(proposal(),r)
    assert result["request_fingerprint"] == r["request_fingerprint"]
    assert r == before and len(r["snapshots"][0]["state"]) == 14
    payload = teacher.build_payload(r,"gpt-6-astra")
    assert sum(p["type"]=="image_url" for p in payload["messages"][1]["content"]) == 3
    r["observation_step"] = 1
    with pytest.raises(ValueError,match="fingerprint"):
        teacher._validate_request(r)


@pytest.mark.parametrize("changes",[
    {"method":"native","noise_sigma":.1},
    {"method":"tei","subgoal":"Lift the object.","alpha":float("nan")},
    {"method":"image","alpha":.5,"roi":[.8,0,.4,1]},
    {"method":"frs","alpha":1,"translation_bias":[0,0,.5]},
    {"method":"frs","alpha":1},
    {"layer":True},
])
def test_selector_rejects_hidden_edits_nonfinite_and_invalid_bounds(changes):
    with pytest.raises(ValueError):
        teacher.parse_proposal(proposal(**changes),request())


@pytest.mark.parametrize("method",teacher.METHODS)
def test_all_declared_interventions_have_valid_explicit_contracts(method):
    extra = {"method":method}
    if method in ("phase_prompt","tei","tli"):extra["subgoal"]="Lift the object."
    if method in ("tei","tli","image","vei","vli"):extra["alpha"]=.5
    if method=="frs":extra.update(alpha=1,noise_sigma=.1)
    assert teacher.parse_proposal(proposal(**extra),request())["method"] == method


def test_image_edits_change_real_pixels_only_in_requested_camera_without_mutating_input():
    images={"video.robot0_"+name:np.full((4,16,16,3),40,np.uint8)
            for name in ("agentview_left","agentview_right","eye_in_hand")}
    before=copy.deepcopy(images)
    settings=proposal(method="image",alpha=.5,roi=[.25,.25,.75,.75])
    source=perturb_images(images,settings)
    mixed=blend_images(images,source,.5)
    key="video.robot0_agentview_left"
    assert np.all(mixed[key][:,4:12,4:12]==84)
    assert np.all(mixed[key][:,:4]==40)
    assert np.array_equal(mixed["video.robot0_eye_in_hand"],images["video.robot0_eye_in_hand"])
    assert all(np.array_equal(images[k],before[k]) for k in images)
    assert blend_images(images,source,0) is images


def test_latent_writes_leave_every_unselected_position_exactly_unchanged():
    hidden=torch.randn(1,12,5)
    slots=torch.tensor([4,7])
    result=add_at_slots(hidden,slots,torch.ones(2,5),.5)
    protected=torch.tensor([0,1,2,3,5,6,8,9,10,11])
    assert torch.equal(result[:,protected],hidden[:,protected])
    torch.testing.assert_close(result[:,slots],hidden[:,slots]+.5)


def test_full_flow_reversal_calls_correct_time_direction_and_resynthesizes_target():
    calls=[]
    def velocity(x,t):
        calls.append(float(t[0,0,0]))
        return torch.ones_like(x)*2
    target=torch.ones(1,16,12)*3
    noise=flow_integrate(velocity,target,reverse=True)
    result=flow_integrate(velocity,noise)
    assert calls == pytest.approx([1,.8,.6,.4,.2,0,.2,.4,.6,.8])
    torch.testing.assert_close(noise,torch.ones_like(target))
    torch.testing.assert_close(result,target)


def test_text_slots_exclude_camera_labels_delimiters_and_assistant_tokens():
    tokenizer=SimpleNamespace(encode=lambda s,**kw:list(s.encode()),
                              decode=lambda ids,**kw:bytes(ids).decode())
    instruction="Put corn in the container."
    prefix="Left camera: <video>\nGenerate robot actions for the task:\n"
    suffix=" /no_cot<assistant><cot></cot>"
    ids=torch.tensor([list((prefix+instruction+suffix).encode())])
    slots=text_slots(tokenizer,ids,instruction)
    assert bytes(ids[0,slots].tolist()).decode()==instruction
    assert slots[0]==len(prefix) and slots[-1]==len(prefix)+len(instruction)-1
