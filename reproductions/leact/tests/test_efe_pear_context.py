from copy import deepcopy
import base64
import io
import json
from pathlib import Path

from PIL import Image
import pytest

from reproductions.leact.cluster.efe_action_loop import native_contract
from reproductions.leact.cluster.efe_pear_bridge import image_blocks
from reproductions.leact.cluster.efe_pear_context import VERSION, interface_context


FIXTURES=Path(__file__).parent/'fixtures'


def observation(family='metaworld'):
    spec=json.loads((FIXTURES/'efe-native-action-specs.json').read_text())[family]['spec']
    layout=spec['proprio_layout']
    return {'mode':'task','objective':'pick the requested object','spec':spec,
            'action_contract':native_contract(spec),
            'observation':{'proprio':{'layout':layout,'now':[0.]*len(layout)},
                           'control_steps':981234,'time_s':983271},
            'recent_actions_and_feedback':[]}


def content(visible,size=(32,24)):
    image=Image.new('RGB',size,(255,0,0));out=io.BytesIO();image.save(out,format='PNG')
    url='data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()
    result=[{'type':'input_text','text':json.dumps(visible)}]
    for name in visible['action_contract']['cameras']:
        result.extend([{'type':'input_text','text':'Current camera: '+name},
                       {'type':'input_image','image_url':url}])
    return result


def test_published_world_directions_native_scale_and_gripper_signs_are_preserved():
    visible=observation();before=deepcopy(visible)
    guide=interface_context(visible)['text']
    assert "x to the robot's right, y forward (away from the robot base), z up"in guide
    assert '1.0 = 1 cm'in guide and '-1 open .. +1 close'in guide
    assert '0.0125 seconds'in guide and 'opposite direction'in guide
    assert json.dumps(['dx','dy','dz','grip'])in guide
    assert visible==before


def test_world_up_does_not_invent_base_frame_directions():
    guide=interface_context(observation('robocasa'))['text']
    assert 'world frame: metres, z up'in guide
    assert 'arm channels: end-effector deltas in the robot base frame'in guide
    assert 'not automatically world or robot-base directions'in guide
    assert "x to the robot's right"not in guide
    assert '1.0 = 5 cm / 0.5 rad'in guide


def test_proprio_sensor_template_aligns_values_without_copying_current_state():
    visible=observation();guide=interface_context(visible)['text']
    assert 'observation.proprio.now[i]'in guide and 'observation.proprio.layout[i]'in guide
    assert 'tcp_x'in guide and 'right_j0_vel'in guide
    assert '<number>'in guide and '<array length=27;'in guide
    assert '981234'not in guide and '983271'not in guide
    visible['observation']['proprio']['now'].pop()
    with pytest.raises(ValueError,match='proprioception'):interface_context(visible)


def test_only_selected_delivered_sensors_are_present_and_depth_is_not_claimed():
    visible=observation();visible['action_contract']['cameras']=['right','front']
    visible['spec']['cameras']['hidden_camera']={'rgb':True,'depth':True}
    guide=interface_context(visible,image_sizes=[(640,480),(320,240)])['text']
    assert guide.index('- RGB sensor right')<guide.index('- RGB sensor front')
    assert '- RGB sensor left'not in guide and 'hidden_camera'not in guide
    assert '640 pixels wide by 480 high'in guide
    assert 'Depth is not attached'in guide
    assert 'increasing columns moves right and increasing rows moves down'in guide
    with pytest.raises(ValueError,match='Delivered RGB'):interface_context(visible,image_sizes=[(640,480)])


def test_bimanual_finger_order_and_absolute_target_semantics_are_not_flattened():
    spec=json.loads((FIXTURES/'dexjoco-native-action-spec.json').read_text())['native_spec']
    contract=native_contract(spec,allowed_action_types=['joint_position'])
    guide=interface_context({'spec':spec,'action_contract':contract,'observation':{}})['text']
    assert json.dumps(contract['action_types']['joint_position']['layout'])in guide
    assert 'joint_position: 46 components'in guide and 'units: rad'in guide
    assert 'measured joints may differ from q'in guide
    assert 'Not declared.'in guide
    assert 'ee_pose:'not in guide


def test_runtime_bridge_delivers_guide_exact_json_and_unchanged_rgb_in_order():
    visible=observation();source=content(visible)
    blocks=image_blocks(source)
    assert blocks[0]['type']=='text'and blocks[0]['text'].startswith('Action and observation guide')
    assert blocks[1]=={'type':'text','text':source[0]['text']}
    images=[block['image']for block in blocks if block['type']=='image']
    assert len(images)==4 and all(image.getpixel((0,0))==(255,0,0)for image in images)
    assert '32 pixels wide by 24 high'in blocks[0]['text']
    bad=deepcopy(source);bad[1]['text']='Current camera: wrong_name'
    with pytest.raises(ValueError,match='sensor labels'):image_blocks(bad)


def test_context_version_and_digest_are_model_independent_and_public_metadata_only():
    visible=observation();a=interface_context(visible)
    visible['model']='another student';visible['objective']='PRIVATE_ANSWER_DO_NOT_COPY'
    visible['spec']['task_text']='PRIVATE_ANSWER_DO_NOT_COPY'
    visible['spec']['private_simulator_state']={'answer':'PRIVATE_ANSWER_DO_NOT_COPY'}
    visible['observation']['time_s']=1.
    b=interface_context(visible)
    assert a==b and b['version']==VERSION and len(b['sha256'])==64
    assert 'PRIVATE_ANSWER_DO_NOT_COPY'not in b['text']


def test_missing_proprio_sensor_values_are_distinguished_from_schema():
    visible=observation();visible['observation']['proprio'].pop('now')
    guide=interface_context(visible)['text']
    assert 'No current proprio.now sensor vector is supplied'in guide
    assert 'tcp_x'in guide


def test_resumed_plan_keeps_existing_recipe_results_and_catalog_metadata():
    from reproductions.leact.cluster.launch_efe_pear_scaled import update_interface_plan
    plan={'recipe':{'score_centering':True,'retained_groups_per_update':128},
          'primary_rl_updates':7,'arms':[{'id':'a'},{'id':'b'}],
          'proposal':{'interface':{'contract':'existing native control'},
                      'optimizer':{'lr':1e-6},'catalog_access':'verified source'}}
    update_interface_plan(plan)
    assert plan['primary_rl_updates']==7 and plan['proposal']['catalog_access']=='verified source'
    assert plan['recipe']['score_centering']and plan['recipe']['retained_groups_per_update']==128
    assert plan['interface_context']['same_for_both_arms']and plan['interface_context']['enabled']
    assert plan['proposal']['interface']['contract']=='existing native control'
    assert plan['recipe']['prompt_budget']==plan['proposal']['optimizer']['prompt_budget_initial']==16384
