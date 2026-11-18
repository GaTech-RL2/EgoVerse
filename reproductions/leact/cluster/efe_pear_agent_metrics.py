"""PEAR's unchanged agent loop with per-episode diagnostic counters."""
import os
import time

from pear.agent_loop.agent_loop import PearAgentLoop
from efe_pear_telemetry import WorkerDiagnostics

_DIAGNOSTICS=None


def diagnostics():
    global _DIAGNOSTICS
    if _DIAGNOSTICS is None:_DIAGNOSTICS=WorkerDiagnostics(os.environ['EFE_PEAR_METRICS_DIR'])
    return _DIAGNOSTICS


class MeasuredPearAgentLoop(PearAgentLoop):
    async def run(self,sampling_params,**kwargs):
        start=time.monotonic();self._generated=0;self._thought=0;self._thought_measured=True;self._turns=0
        try:output=await super().run(sampling_params,**kwargs)
        except Exception:
            diagnostics().add({'infrastructure_or_runtime_failed_attempts':1,
                'failed_attempt_seconds':time.monotonic()-start,'failed_attempt_generated_tokens':self._generated})
            raise
        reason=self.end_reason.value
        values={'completed_episodes':1,'episode_seconds_sum':time.monotonic()-start,
            'generated_tokens_sum':self._generated,'turns_sum':self._turns,
            'end_reason/'+reason:1,'reward_sum':float(self.episode.success)}
        if self._thought_measured:values.update(thinking_tokens_sum=self._thought,thinking_measured_episodes=1)
        diagnostics().add(values)
        output.extra_fields['pear_diagnostics']=values
        return output

    async def _log_turn(self,agent_data):
        await super()._log_turn(agent_data)
        ids=list(agent_data.response_ids);self._generated+=len(ids);self._turns+=1
        end=self.tokenizer.convert_tokens_to_ids('</think>')
        if end is not None and end!=self.tokenizer.unk_token_id and end in ids:
            self._thought+=ids.index(end)
        else:self._thought_measured=False
