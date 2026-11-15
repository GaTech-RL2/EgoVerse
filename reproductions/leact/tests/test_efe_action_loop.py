import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from reproductions.leact.cluster.efe_action_loop import native_contract, parse_action, movement_verdict, tcp, main


FIXTURES = json.loads((Path(__file__).parent/'fixtures/efe-native-action-specs.json').read_text())


class ActionContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = native_contract(FIXTURES['metaworld']['spec'])

    def test_exact_actions_survive_without_coercion(self):
        action = {'action_type':'ee_delta','action':[0.3,-0.2,0,-1], 'repeat':5, 'summary':'Small forward correction.'}
        self.assertEqual(parse_action(json.dumps(action),contract=self.contract,allow_stop=False),action)
        for bad in [dict(action,action=['0.3',0,0,-1]),dict(action,repeat=True),
                    dict(action,repeat=26),dict(action,action=[float('nan'),0,0,-1]),
                    dict(action,action=[1.01,0,0,-1]),dict(action,cmd='python'),
                    dict(action,action_type='joint_position'),dict(action,action=[0]*12)]:
            with self.assertRaises(ValueError):
                parse_action(json.dumps(bad),contract=self.contract,allow_stop=False)

    def test_robot_specific_dimensions_timing_and_cameras(self):
        contract = native_contract(FIXTURES['robocasa']['spec'])
        action = {'action_type':'ee_delta','action':[0]*11+[-1],'repeat':3,'summary':'Small correction.'}
        self.assertEqual(parse_action(json.dumps(action),contract=contract,allow_stop=False),action)
        self.assertEqual(contract['control_dt'],0.05)
        self.assertEqual(self.contract['control_dt'],0.0125)
        self.assertEqual(contract['action_types']['ee_delta']['dim'],12)
        self.assertIn('robot0_eye_in_hand',contract['cameras'])
        with self.assertRaises(ValueError):
            parse_action(json.dumps(action),contract=self.contract,allow_stop=False)

    def test_native_types_and_asymmetric_component_bounds(self):
        spec = {'control_dt':0.1,'max_chunk_s':0.3,'cameras':{'rgb':{'rgb':True}},
            'action_types':{'joint_position':{'dim':2,'layout':['j0','j1'],
                'lower':[-3,0.2],'upper':[3,2.8],'units':'radians','dtype':'float64'}}}
        contract = native_contract(spec)
        self.assertEqual(contract['max_repeat'],3)
        action = {'action_type':'joint_position','action':[-2,2],'repeat':3,'summary':'Joint targets.'}
        self.assertEqual(parse_action(json.dumps(action),contract=contract,allow_stop=False),action)
        for bad in [dict(action,action=[0,0]),dict(action,repeat=4),dict(action,action_type='ee_delta')]:
            with self.assertRaises(ValueError):
                parse_action(json.dumps(bad),contract=contract,allow_stop=False)
        for dt in [0,-1,float('nan'),True,None]:
            with self.assertRaises(ValueError):native_contract(dict(spec,control_dt=dt))

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_action('{"stop":false,"stop":true,"summary":"done"}',contract=self.contract,allow_stop=True)

    def test_single_json_fence_preserves_commands_but_does_not_extract_from_prose(self):
        action={'action_type':'ee_delta','action':[0.1,0,0,-1],'repeat':2,'summary':'Move right.'}
        raw=json.dumps(action)
        for text in ['```json\n'+raw+'\n```', '\n```\n'+raw+'\n```\n']:
            self.assertEqual(parse_action(text,contract=self.contract,allow_stop=False),action)
        for text in ['Here is the action:\n```json\n'+raw+'\n```',
                     '```json\n'+raw+'\n```\nMore text',
                     '```json\n'+raw+'\n```\n```json\n'+raw+'\n```',
                     '```python\n'+raw+'\n```',
                     '```json\n'+json.dumps(dict(action,action=[2,0,0,-1]))+'\n```']:
            with self.assertRaises(ValueError):parse_action(text,contract=self.contract,allow_stop=False)

    def test_stop_is_a_task_decision_not_a_movement_pass(self):
        stop='{"stop":true,"summary":"Task appears complete."}'
        self.assertTrue(parse_action(stop,contract=self.contract,allow_stop=True)['stop'])
        with self.assertRaises(ValueError):
            parse_action(stop,contract=self.contract,allow_stop=False)

    def test_measured_displacement_gate(self):
        start=[0.1,0.6,0.2]
        self.assertTrue(movement_verdict(start,[0.12,0.6,0.2],0)['passed'])
        for end in [[0.1,0.6,0.2],[0.08,0.6,0.2],[0.16,0.6,0.2],[0.12,0.63,0.2]]:
            self.assertFalse(movement_verdict(start,end,0)['passed'])
        self.assertEqual(tcp({'proprio':{'layout':['tcp_z','tcp_x','tcp_y'],
                                       'now':[0.2,0.1,0.6]}}),start)

    def test_bridge_submits_exact_actions_and_measures_feedback(self):
        import numpy as np
        position=[0.1,0.6,0.2]; submitted=[]; bodies=[]
        class Result:
            done=False
            @property
            def meta(self):return {'proprio':{'layout':['tcp_x','tcp_y','tcp_z'],'now':list(position)}}
            last={c:np.zeros((16,16,3),dtype=np.uint8) for c in ['front','left','right','topview']}
        class Robot:
            def spec(self):return FIXTURES['metaworld']['spec']
            def get_obs(self,**kwargs):return Result()
            def submit(self,kind,actions,**kwargs):
                submitted.append((kind,actions,kwargs))
                for action in actions:
                    for i in range(3):position[i]+=action[i]*0.01
                return Result()
        module=types.ModuleType('rcb_client');module.Robot=Robot
        module.EpisodeOver=type('EpisodeOver',(RuntimeError,),{})
        def response(body):
            bodies.append(body);axis=len(bodies)-1
            action=[0,0,0,-1];action[axis]=0.5
            return json.dumps({'action_type':'ee_delta','action':action,'repeat':4,'summary':'Directed movement.'}),{'input_tokens':10,'output_tokens':20},'test-model'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'config.json';instruction=root/'instruction.txt';logs=root/'logs'
            config.write_text(json.dumps({'mode':'movement','model':'test-model','max_calls_per_goal':4,
                'max_task_calls':96,'wall_seconds':600,'max_output_tokens':32768}))
            instruction.write_text('Test task')
            with patch.dict(sys.modules,{'rcb_client':module}), patch('sys.argv',['bridge','--config',str(config),'--instruction',str(instruction),'--logs',str(logs)]), patch('reproductions.leact.cluster.efe_action_loop.response',response):
                main()
            stats=json.loads((logs/'action-metrics.json').read_text())
            self.assertEqual(stats['movement_passes'],3)
            self.assertEqual(stats['control_rows_requested'],12)
            self.assertEqual(stats['output_tokens'],60)
            for axis,(kind,actions,kwargs) in enumerate(submitted):
                expected=[0,0,0,-1];expected[axis]=0.5
                self.assertEqual(kind,'ee_delta');self.assertEqual(actions,[expected]*4)
                self.assertEqual(kwargs['dt'],0.0125)
                content=bodies[axis]['input'][0]['content']
                self.assertEqual(sum(p['type']=='input_image' for p in content),4)
                self.assertEqual(bodies[axis]['reasoning'],{'effort':'high'})
                visible=json.loads(content[0]['text'])
                self.assertEqual(visible['action_contract']['control_dt'],kwargs['dt'])
            trajectory=json.loads((logs/'trajectory.json').read_text())
            import hashlib
            for step,body in zip(trajectory['steps'][1:],bodies):
                self.assertEqual(step['request_sha256'],hashlib.sha256(json.dumps(body,allow_nan=False).encode()).hexdigest())

    def test_task_bridge_accepts_twelve_dimensions_without_tcp_fields(self):
        import numpy as np
        submitted=[]
        spec=FIXTURES['robocasa']['spec']
        class Result:
            done=True
            meta={'proprio':{'layout':['eef_right_x','eef_right_y','eef_right_z'],'now':[1,2,3]},'control_steps':1}
            last={c:np.zeros((16,16,3),dtype=np.uint8) for c in spec['cameras']}
        class Robot:
            def spec(self):return spec
            def get_obs(self,**kwargs):return Result()
            def submit(self,kind,actions,**kwargs):
                submitted.append((kind,actions,kwargs));return Result()
        module=types.ModuleType('rcb_client');module.Robot=Robot
        module.EpisodeOver=type('EpisodeOver',(RuntimeError,),{})
        action={'action_type':'ee_delta','action':[0]*11+[-1],'repeat':1,'summary':'Move with native controls.'}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'config.json';instruction=root/'instruction.txt';logs=root/'logs'
            config.write_text(json.dumps({'mode':'task','model':'test-model','max_calls_per_goal':4,
                'max_task_calls':2,'wall_seconds':600,'max_output_tokens':32768}))
            instruction.write_text('Turn the faucet')
            with patch.dict(sys.modules,{'rcb_client':module}), patch('sys.argv',['bridge','--config',str(config),'--instruction',str(instruction),'--logs',str(logs)]), patch('reproductions.leact.cluster.efe_action_loop.response',return_value=(json.dumps(action),{},'test-model')):
                main()
            self.assertEqual(submitted[0][1],[action['action']])
            self.assertEqual(submitted[0][2]['dt'],0.05)
            self.assertEqual(json.loads((logs/'action-metrics.json').read_text())['valid_actions'],1)


if __name__=='__main__':
    unittest.main()
