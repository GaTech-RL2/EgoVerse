import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def student(monkeypatch):
    # Isolate the policy decision from GPU serving and image encoding dependencies.
    monkeypatch.setitem(sys.modules,'efe_native_pilot',SimpleNamespace(now=lambda:'test-time'))
    class Image:
        @staticmethod
        def new(mode,size,rgb):
            return SimpleNamespace(save=lambda buffer,**kwargs:buffer.write(bytes(rgb)))
    monkeypatch.setitem(sys.modules,'PIL',SimpleNamespace(Image=Image))
    source=Path(__file__).parents[1]/'cluster/efe_qwen_baseline.py'
    spec=importlib.util.spec_from_file_location('student_preflight_test',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def responses(student,monkeypatch,color='blue',first_color='green'):
    call={'type':'function_call','name':'report_color','namespace':'probe','call_id':'call-1',
          'arguments':json.dumps({'color':first_color})}
    values=[{'output':[call]},
            {'output':[{'type':'message','content':[{'type':'output_text','text':'green'}]}]},
            {'output':[{'type':'message','content':[{'type':'output_text','text':color}]}]}]
    bodies=[]
    def response(url,body,token):
        bodies.append(body);return values[len(bodies)-1]
    monkeypatch.setattr(student,'response',response)
    return bodies,values


def test_default_still_stops_on_wrong_new_image(student,monkeypatch):
    responses(student,monkeypatch,color='green')
    with pytest.raises(RuntimeError,match='image inside tool-result'):
        student.vision_tool_preflight({'request_model':'fixture','upstream':'fixture'},'token')


def test_communication_workaround_is_opt_in_and_cannot_override_model_options(student):
    assert student.serving_transport_options({})==([],{})
    args,env=student.serving_transport_options({'serving_transport':{'mode':'nccl_no_p2p_v1'}})
    assert args==['--disable-custom-all-reduce'] and env=={'NCCL_P2P_DISABLE':'1'}
    with pytest.raises(ValueError,match='Unknown serving transport'):
        student.serving_transport_options({'serving_transport':{'mode':'unknown'}})


def test_opt_in_records_wrong_answer_without_changing_task_inputs(student,monkeypatch):
    bodies,_=responses(student,monkeypatch,color='green')
    result=student.vision_tool_preflight({'request_model':'fixture','upstream':'fixture',
        'preflight_semantics_policy':'record_and_evaluate'},'token')
    assert result['status']=='semantic_failure' and result['benchmark_execution_allowed']
    assert [row['passed'] for row in result['semantic_checks']]==[True,True,False]
    assert result['tool_result_image_continuation']=='green'
    original=bodies[2]['input'][0]['content'][1]['image_url']
    returned=bodies[2]['input'][3]['output'][1]['image_url']
    assert original!=returned and bodies[2]['tool_choice']=='none'


def test_opt_in_does_not_hide_function_transport_failure(student,monkeypatch):
    _,values=responses(student,monkeypatch)
    values[0]['output']=[]
    with pytest.raises(RuntimeError,match='function-call preflight failed'):
        student.vision_tool_preflight({'request_model':'fixture','upstream':'fixture',
            'preflight_semantics_policy':'record_and_evaluate'},'token')


def test_opt_in_does_not_swallow_http_or_stream_errors(student,monkeypatch):
    def failed(*args):raise RuntimeError('Student Responses preflight did not complete: response.incomplete')
    monkeypatch.setattr(student,'response',failed)
    with pytest.raises(RuntimeError,match='response.incomplete'):
        student.vision_tool_preflight({'request_model':'fixture','upstream':'fixture',
            'preflight_semantics_policy':'record_and_evaluate'},'token')


def test_correct_answers_keep_existing_pass_behavior(student,monkeypatch):
    responses(student,monkeypatch)
    result=student.vision_tool_preflight({'request_model':'fixture','upstream':'fixture'},'token')
    assert result['status']=='passed' and all(row['passed'] for row in result['semantic_checks'])


def test_wrong_color_is_distinct_from_malformed_function_arguments(student,monkeypatch):
    _,values=responses(student,monkeypatch,first_color='red')
    spec={'request_model':'fixture','upstream':'fixture','preflight_semantics_policy':'record_and_evaluate'}
    result=student.vision_tool_preflight(spec,'token')
    assert not result['semantic_checks'][0]['passed']
    _,values=responses(student,monkeypatch)
    values[0]['output'][0]['arguments']='{"color":"not-a-valid-enum"}'
    with pytest.raises(RuntimeError,match='invalid function arguments'):
        student.vision_tool_preflight(spec,'token')
