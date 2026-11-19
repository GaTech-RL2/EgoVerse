"""Contract and controller-budget checks for the opt-in Cartesian branch."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from reproductions.leact.cluster.efe_action_loop import parse_action, system_prompt
from reproductions.leact.cluster.efe_pear_eef import VERSION, contract, native_pose, public_spec
from reproductions.leact.cluster.efe_pear_eef_service import bound_service
from reproductions.leact.cluster.efe_pear_eef import AbcCartesianRobot
from reproductions.leact.cluster.efe_pear_vlabench_eef import install as install_vlabench, stage


def spec(suite):
    count = 1 if suite == 'robolab' else 2
    entry = {'dim': 8 * count, 'layout': [f'{arm}_{k}' for arm in range(count)
             for k in ('x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'gripper')],
             'frame': 'world: x forward, y left, z up', 'semantics': 'unbounded native waypoint'}
    return {'control_hz': 30, 'max_chunk_s': 1., 'cameras': {'rgb': {'rgb': True}},
            'world': {'frame': 'native world', 'up': '+z'},
            'action_types': {'ee_pose': entry, 'joint_position': {}, 'not_yet': ['ee_delta']}}


@pytest.mark.parametrize('suite,opened,closed', [
    ('robodojo', 1., 0.), ('yamlab', -.0475, 0.), ('robolab', 0., 1.)])
def test_common_gripper_mapping_and_native_frame_are_explicit(suite, opened, closed):
    native = spec(suite); frozen = copy.deepcopy(native)
    result = contract(native, suite)
    entry = result['action_types']['ee_pose']
    action = [0., 0., .3, 1., 0., 0., 0., 0.] * (entry['dim'] // 8)
    command = {'action_type': 'ee_pose', 'action': action, 'repeat': 5, 'summary': 'Approach.'}
    assert parse_action(json.dumps(command), contract=result, allow_stop=True) == command
    converted = native_pose(action, entry)
    assert converted[7] == opened
    action[7] = 1.
    assert native_pose(action, entry)[7] == closed
    assert entry['frame'] == native['action_types']['ee_pose']['frame']
    assert native == frozen  # no mutation of pinned/native or joint-run metadata
    advertised = public_spec(native, result)
    assert list(advertised['action_types']) == ['ee_pose']
    assert advertised['frames'] == native['world']
    assert result['protocol'] == VERSION
    assert 'does not guarantee arrival' in system_prompt(result)


@pytest.mark.parametrize('quaternion', [[0., 0., 0., 0.], [1., 1., 0., 0.], [.5, 0., 0., 0.]])
def test_invalid_pose_quaternion_is_rejected_before_physics(quaternion):
    result = contract(spec('robolab'), 'robolab')
    action = [0., 0., .3, *quaternion, 0.]
    with pytest.raises(ValueError, match='unit norm'):
        parse_action(json.dumps({'action_type': 'ee_pose', 'action': action,
            'repeat': 5, 'summary': 'Approach.'}), contract=result, allow_stop=True)


def test_dexterous_hand_targets_and_right_first_order_survive_cartesian_contract():
    native = spec('robolab')
    layout = ['right_' + k for k in ('x', 'y', 'z', 'qw', 'qx', 'qy', 'qz')]
    hand = [f'allegro_{i}' for i in range(16)]
    native['action_types']['ee_pose'].update(dim=23, layout=layout + hand,
        hand_limits={name: [-.1 * i, .1 * i + 1.] for i, name in enumerate(hand)})
    result = contract(native, 'dexjoco')
    entry = result['action_types']['ee_pose']
    assert entry['layout'] == layout + hand
    assert entry['lower'][7:] == [-.1 * i for i in range(16)]
    vector = [0., 0., .4, 1., 0., 0., 0.] + [.1] * 16
    assert native_pose(vector, entry) == vector
    # The real SDK names action columns <joint>_target, but its hand_limits
    # mapping uses the unsuffixed joint names.
    native['action_types']['ee_pose']['layout']=layout+[name+'_target'for name in hand]
    assert contract(native,'dexjoco')['action_types']['ee_pose']['lower'][7:]==entry['lower'][7:]


def test_unimplemented_joint_only_simulator_cannot_silently_enter_eef_training():
    with pytest.raises(ValueError, match='has not been qualified'):
        contract(spec('robolab'), 'unreviewed-simulator')


def fake_service(*, native_steps=100, terminal_step=None, raises=False):
    history = []
    namespace = {'run_ee_pose': object(), 'history': history, 'native_steps': native_steps,
                 'terminal_step': terminal_step, 'raises': raises}
    exec('''
def ctrl_step(action, recorder):
    history.append(action)
    return len(history) == terminal_step
def submit(meta, actions):
    before = len(history)
    if raises:
        raise RuntimeError('native failure')
    for step in range(native_steps):
        if ctrl_step(step, None):
            break
    return {'control_steps_chunk': len(history) - before, 'success': False,
            'plan': [{'position_error_m': .2}]}, ['native recording']
''', namespace)
    original_step = namespace['ctrl_step']
    return SimpleNamespace(submit=namespace['submit']), history, namespace, original_step


@pytest.mark.parametrize('budget', [1, 7, 25])
def test_waypoint_loop_stops_at_requested_native_steps_and_preserves_verdict(budget):
    service, history, namespace, original_step = fake_service()
    bound_service(service)
    output, parts = service.submit({'type': 'ee_pose'}, np.zeros((budget, 16)))
    assert len(history) == budget
    assert output['eef_execution']['executed_control_steps'] == budget
    assert output['eef_execution']['requested_control_steps'] == budget
    assert output['success'] is False and parts == ['native recording']
    assert namespace['ctrl_step'] is original_step


def test_native_termination_stops_early_and_restores_controller():
    service, history, namespace, original_step = fake_service(terminal_step=3)
    bound_service(service)
    output, _ = service.submit({'type': 'ee_pose'}, np.zeros((20, 16)))
    assert len(history) == 3
    assert not output['eef_execution']['step_budget_exhausted']
    assert namespace['ctrl_step'] is original_step


def test_controller_exception_restores_native_step_function():
    service, _, namespace, original_step = fake_service(raises=True)
    bound_service(service)
    with pytest.raises(RuntimeError, match='native failure'):
        service.submit({'type': 'ee_pose'}, np.zeros((2, 16)))
    assert namespace['ctrl_step'] is original_step


def test_bounded_service_rejects_ambiguous_waypoints_without_execution():
    service, history, _, _ = fake_service()
    bound_service(service)
    targets = np.zeros((2, 16)); targets[1, 0] = .1
    with pytest.raises(ValueError, match='identical rows'):
        service.submit({'type': 'ee_pose'}, targets)
    assert history == []


def test_abc_public_ik_is_translated_to_bounded_native_joint_targets():
    native = {'control_hz':30., 'max_chunk_s':1.,'cameras':{'rgb':{'rgb':True}},
        'world':{'frame':'native world, x forward y left z up'},
        'kinematics':{'frame':'world','quaternion_order':'wxyz',
            'ik_targets':['left_tcp','right_tcp'],'joint_names':[f'q_{i}'for i in range(14)]},
        'action_types':{'joint_position':{'limits':{'low':[-2.]*14,'high':[2.]*14}}}}
    state = SimpleNamespace(meta={'control_steps':0,'proprio':{'layout':[f'q_{i}'for i in range(14)]+[
        side+'_tcp_'+axis for side in ['left','right']for axis in 'xyz'],'now':[0.]*20}})
    calls=[]
    class Robot:
        def spec(self):return copy.deepcopy(native)
        def get_obs(self,**kwargs):return state
        def ik(self,targets,seed):
            assert set(targets)=={'left_tcp','right_tcp'} and len(seed)==14
            return {'q':[10.]*14,'error':{}}
        def submit(self,kind,rows,**kwargs):
            calls.append((kind,np.array(rows)));state.meta['control_steps']+=len(rows);return state
    robot=AbcCartesianRobot(Robot());robot.get_obs()
    target=[.1,.2,.3,1.,0.,0.,0.,0.,.4,.5,.6,1.,0.,0.,0.,1.]
    result=robot.submit('ee_pose',[target]*5)
    assert calls[0][0]=='joint_position'
    commands=calls[0][1]
    assert commands.shape==(5,14)
    assert np.max(np.abs(np.diff(np.vstack([np.zeros((1,14)),commands]),axis=0)))<=.050000001
    assert commands[-1,6]>0 and commands[-1,13]==0  # public open/close -> native opening
    assert result.meta['eef_execution']['executed_control_steps']==5
    assert robot.spec()['action_types']['ee_pose']['frame']==native['world']['frame']


def vla_sim():
    class Sim:
        def __init__(self):
            self.position=np.array([0.,0.,.4]);self.changed=False
            self.lower=np.array([-3.]*7+[0.,0.]);self.upper=np.array([3.]*7+[.04,.04])
            env=SimpleNamespace(get_ee_pos=lambda:self.position,get_ee_quat=lambda:np.array([1.,0.,0.,0.]),
                robot=SimpleNamespace(get_qpos=lambda _:np.zeros(7)))
            self.s=SimpleNamespace(env=env,physics=None,targets=np.array([0.]*7+[.04,.04]),
                state_sha=lambda:str(self.changed),ik=lambda p,q:{'success':True,'joint_positions':np.ones(7)})
        def spec(self):return {'frames':'native world','action_types':{'joint_position':{
            'panda_joint_limits_rad':[[-2.,2.]]*7},'not_yet':['ee_pose']}}
        def clip(self,kind,rows):return np.asarray(rows),[]
        def command(self,kind):return self.s.targets.copy()
        def to_sim(self,kind,row):return np.asarray(row)
        def step(self,row):return False,False,{}
    install_vlabench(Sim)
    return Sim()


def test_vlabench_native_ik_keeps_grader_and_bounds_joint_and_gripper_commands():
    sim=vla_sim();row=np.array([.02,0.,.4,1.,0.,0.,0.,1.])
    native=sim.to_sim('ee_pose',row)
    assert np.allclose(native[:7],.05) and np.allclose(native[7:],0.)
    done,success,info=sim.step(native)
    assert not done and not success
    assert info['eef_execution']['native_ik_success']
    json.dumps(info)  # Native RCB receipts require plain JSON scalars.
    assert sim.spec()['action_types']['ee_pose']['frame']=='native world'


def test_vlabench_unsolved_ik_holds_arm_and_reports_failure():
    sim=vla_sim();sim.s.ik=lambda p,q:{'success':False,'joint_positions':np.ones(7)}
    native=sim.to_sim('ee_pose',np.array([2.,2.,2.,1.,0.,0.,0.,0.]))
    assert np.allclose(native[:7],0.)
    assert not sim.step(native)[2]['eef_execution']['native_ik_success']


def test_vlabench_ik_cannot_modify_physical_state():
    sim=vla_sim()
    def mutate(p,q):
        sim.changed=True
        return {'success':True,'joint_positions':np.ones(7)}
    sim.s.ik=mutate
    with pytest.raises(RuntimeError,match='modified physical simulator state'):
        sim.to_sim('ee_pose',np.array([0.,0.,.4,1.,0.,0.,0.,0.]))


def test_vlabench_source_overlay_breaks_task_symlinks_without_touching_original(tmp_path):
    original=tmp_path/'pinned-server.py';raw='class VLABench: pass\nif __name__ == "__main__":\n    main(build)\n'
    original.write_text(raw);adapter=tmp_path/'adapter.py';adapter.write_text('def install(cls):pass\n')
    rows=[]
    for name in ['train-a','train-b']:
        folder=tmp_path/name/'environment/sim';folder.mkdir(parents=True)
        (folder/'server.py').symlink_to(original)
        (folder/'Dockerfile').write_text('FROM pinned\nCOPY --from=pinned-files /server.py /srv/server.py\n')
        rows.append({'suite':'vlabench','path':name,'task':name})
    receipts=stage(tmp_path,rows,adapter)
    assert len(receipts)==2 and original.read_text()==raw
    assert all(not(tmp_path/name/'environment/sim/server.py').is_symlink()for name in ['train-a','train-b'])


def test_sparse_vlabench_tasks_use_one_owned_inline_template_overlay(tmp_path):
    central=tmp_path/'embodied_first_exam/benchmarks/vlabench/sim';central.mkdir(parents=True)
    native=central/'server.py';native.write_text('class VLABench:pass\nif __name__ == "__main__":\n    main(build)\n')
    adapter=tmp_path/'adapter.py';adapter.write_text('def install(cls):pass\n')
    rows=[{'suite':'vlabench','path':name,'task':name}for name in ['train-a','train-b']]
    for row in rows:
        folder=tmp_path/row['path']/'environment/sim';folder.mkdir(parents=True)
        (folder/'Dockerfile').write_text('FROM pinned\nCOPY --from=release /sim/server.py /srv/\n')
    receipts=stage(tmp_path,rows,adapter)
    assert len(receipts)==2 and receipts[0]['adapted_server_sha256']==receipts[1]['adapted_server_sha256']
    text=native.read_text()
    assert 'def install_pear_eef(cls)'in text and 'install_pear_eef(VLABench)'in text
    assert 'from efe_pear_vlabench_eef import'not in text
    compile(text,str(native),'exec')
    for row in rows:
        folder=tmp_path/row['path']/'environment/sim'
        assert(folder/'Dockerfile').read_text().endswith('COPY server.pear-eef.py /srv/server.py\n')
        assert(folder/'server.pear-eef.py').read_text()==text
