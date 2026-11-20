"""Simulator processes must not initialize the model-side training stack."""
from pathlib import Path
import subprocess
import sys


def test_reasoning_simulator_import_does_not_require_verl_or_torch():
    path=Path(__file__).parents[1]/'cluster/efe_pear_reasoning_env.py'
    program='''
import importlib.abc,runpy,sys,types
class BlockTraining(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in {'torch','verl'} or fullname=='pear.envs.tool':
   raise RuntimeError('Simulator imported model-side training libraries: '+fullname)
sys.meta_path.insert(0,BlockTraining())
def stub(name,**values):
 module=types.ModuleType(name);module.__dict__.update(values);sys.modules[name]=module
stub('pear');stub('pear.sim');stub('pear.sim.env',StatefulEnv=object)
stub('efe_pear_sim_process',SpawnedSimulatorProcess=object)
stub('efe_pear_reasoning_scenes',SceneEpisode=object)
loaded=runpy.run_path(sys.argv[1])
assert 'ReasoningEnv' in loaded and 'ReasoningDevelopmentEnv' in loaded
'''
    result=subprocess.run([sys.executable,'-c',program,str(path)],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
