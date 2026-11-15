"""Harbor adapter for the shared fixed observation/action bridge."""
import json
from pathlib import Path
import shlex

from harbor.agents.base import BaseAgent


class FixedActionAgent(BaseAgent):
    @staticmethod
    def name():
        return 'efe-fixed-action'

    def version(self):
        return '3'

    async def setup(self, environment):
        parent = Path(__file__).resolve().parent
        await environment.upload_file(parent/'efe_action_loop.py','/tmp/efe_action_loop.py')
        await environment.upload_file(parent/'action-config.json','/tmp/efe_action_config.json')

    async def run(self, instruction, environment, context):
        self.logs_dir.mkdir(parents=True,exist_ok=True)
        source = self.logs_dir/'instruction.txt'
        source.write_text(instruction)
        await environment.upload_file(source,'/tmp/efe_action_instruction.txt')
        logs = str(self.environment_logs_dir)
        config = json.loads((Path(__file__).resolve().parent/'action-config.json').read_text())
        command = ('PYTHONPATH=/opt/tools python3 -u /tmp/efe_action_loop.py '
            '--config /tmp/efe_action_config.json --instruction /tmp/efe_action_instruction.txt '
            '--logs '+shlex.quote(logs)+' > '+shlex.quote(logs+'/console.txt')+' 2>&1')
        result = await environment.exec(command,env={
            'OPENAI_API_KEY':self._get_env('OPENAI_API_KEY'),
            'OPENAI_BASE_URL':self._get_env('OPENAI_BASE_URL')},
            timeout_sec=config['wall_seconds']+1000)
        missing=[]
        # Capture startup failures before trying optional artifacts: a failure
        # before robot.spec() previously hid the console behind a missing file.
        for name in ['console.txt','native-spec.json','action-metrics.json','action-contract.json','trajectory.json']:
            if not (self.logs_dir/name).exists():
                try:
                    await environment.download_file(logs+'/'+name,self.logs_dir/name)
                except Exception as error:
                    missing.append({'file':name,'error':str(error)})
        if missing:(self.logs_dir/'artifact-download-errors.json').write_text(json.dumps(missing))
        metrics_path = self.logs_dir/'action-metrics.json'
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            context.n_input_tokens = metrics['input_tokens']
            context.n_output_tokens = metrics['output_tokens']
            context.metadata = metrics
        if result.return_code:
            console=(self.logs_dir/'console.txt').read_text()[-4000:] if (self.logs_dir/'console.txt').exists() else 'Console unavailable'
            token=self._get_env('OPENAI_API_KEY')
            if token:console=console.replace(token,'[REDACTED]')
            raise RuntimeError('Fixed-action bridge exited '+str(result.return_code)+'; '+console)
        if missing:
            raise RuntimeError('Fixed-action bridge completed without required evidence: '+','.join(r['file'] for r in missing))
