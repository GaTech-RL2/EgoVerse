"""Discarded PEAR on-policy/score-centering qualification, never a study run."""
import json
import math
import os
from pathlib import Path

import hydra
import pear
import ray
import torch
import transfer_queue as tq

from pear.trainer.config import RunConfig
from pear.trainer.replay_buffer import PearReplayBuffer
from pear.trainer.runner import PearTaskRunner
from pear.trainer.trainer import PearTrainer
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


def record(phase, **fields):
    value={'phase':phase,'primary_rl_updates':0,**fields}
    folder=Path(os.environ['EFE_PEAR_QUALIFICATION_DIR']);folder.mkdir(parents=True,exist_ok=True)
    target=folder/'status.json';temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,allow_nan=False)+'\n');temporary.replace(target)
    print('PEAR_ONPOLICY_CHECK '+json.dumps(value,allow_nan=False),flush=True)


class BoundedCheckBuffer(PearReplayBuffer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs);self.judged=0;self.mixed=0

    def _on_group_judged(self, partition_id, uid, values):
        super()._on_group_judged(partition_id,uid,values)
        if partition_id!='train':return
        if len(values)!=8 or any(v not in (0,1) for v in values):
            raise ValueError('Qualification requires native binary eight-rollout groups')
        self.judged+=1;self.mixed+=int(len(set(values))>1)
        record('judging_on_policy_groups',judged_groups=self.judged,mixed_reward_groups=self.mixed)
        # Never weaken the verifier, inject rewards, or wait forever for a
        # nonzero update if the policy cannot solve any of these training scenes.
        if self.judged>=128 and self.mixed<8:
            raise RuntimeError('Too few mixed-reward groups after 1024 genuine policy episodes')


class QualificationTrainer(PearTrainer):
    def __init__(self, config):
        if os.environ.get('PEAR_PRIMARY_TRAINING')!='0':
            raise ValueError('A discarded qualification must not run as primary training')
        super().__init__(config)
        if (int(config.data.train_batch_size)!=8 or int(config.actor_rollout_ref.rollout.n)!=8
                or int(config.trainer.total_training_steps)!=1
                or not config.actor_rollout_ref.actor.policy_loss.rollout_correction.score_centering
                or int(config.actor_rollout_ref.rollout.topk_log_probs)!=32):
            raise ValueError('Unexpected discarded qualification recipe')
        self.update_checked=False;self.checkpoint_reloaded=False

    def _replay_buffer_class(self, sampler_cls):
        super()._replay_buffer_class(sampler_cls);return BoundedCheckBuffer

    def _update_actor(self, batch, metrics):
        data=tq.kv_batch_get(keys=batch.keys,partition_id=batch.partition_id,
            select_fields=['rollout_topk_ids','rollout_topk_log_probs','response_mask','advantages'])
        nonzero=0;tokens=0
        for ids,probs,mask,advantages in zip(*(data[k].unbind() for k in
                ['rollout_topk_ids','rollout_topk_log_probs','response_mask','advantages']),strict=True):
            mask=mask.bool()
            if ids.shape[-1]!=32 or probs.shape!=ids.shape:raise ValueError('Sampler lost its top-32 head')
            # PEAR stores sampler heads at the predicting full-sequence row,
            # while response_mask excludes the initial prompt. Preserve that
            # one-token offset when inspecting the actual generated tokens.
            prompt_tokens=probs.shape[0]-mask.shape[0]
            if prompt_tokens<1:raise ValueError('Sampler head has no initial prompt alignment')
            selected=probs[prompt_tokens-1:prompt_tokens-1+mask.shape[0]][mask]
            if not torch.isfinite(selected).all():raise FloatingPointError('Sampler head has nonfinite log probabilities')
            tokens+=int(mask.sum());nonzero+=int((advantages[mask]!=0).sum())
        if not tokens or not nonzero:raise ValueError('On-policy batch has no nonzero policy advantage')
        record('score_centering_optimizer_update',sampled_response_tokens=tokens,nonzero_advantage_tokens=nonzero)
        batch=super()._update_actor(batch,metrics)
        grad=metrics.get('actor/grad_norm')
        if grad is None or not math.isfinite(float(grad)) or float(grad)<=0:
            raise FloatingPointError('The actual RL update did not report a positive finite gradient norm')
        self.update_checked=True
        record('score_centering_optimizer_update_passed',discarded_optimizer_updates=1,
            grad_norm=float(grad),sampled_response_tokens=tokens)
        return batch

    def _save_checkpoint(self):
        if not self.update_checked:raise RuntimeError('Cannot qualify a checkpoint before its real RL update')
        super()._save_checkpoint()
        folder=Path(self.config.trainer.default_local_dir)/f'global_step_{self.global_steps}'
        self.actor_rollout_wg.load_checkpoint(str(folder/'actor'),del_local_after_load=False)
        self.checkpoint_reloaded=True
        from efe_pear_checkpoint_archive import archive
        saved=archive(folder,Path(os.environ['EFE_PEAR_QUALIFICATION_DIR'])/'rp_jhh_discarded_sc_step000001')
        record('score_centering_checkpoint_reload_passed',discarded_optimizer_updates=1,
            durable_readback_verified=True,checkpoint_bytes=saved['bytes'],
            reload_scope='Actual saved actor/optimizer loaded through the distributed engine API in the same workers; fresh-process NLL reload was checked separately.')

    def _compute_metrics(self,batch,metrics,timing_raw,global_steps,epoch):
        result=super()._compute_metrics(batch,metrics,timing_raw,global_steps,epoch)
        from efe_pear_telemetry import scalars
        if not self.update_checked or not self.checkpoint_reloaded:raise RuntimeError('Incomplete on-policy qualification')
        receipts=list((Path(os.environ['EFE_PEAR_QUALIFICATION_DIR'])/'optimizer-checks').glob('rank-*.json'))
        if len(receipts)!=8:raise ValueError('Missing frozen-router optimizer receipts')
        checks=[json.loads(p.read_text())for p in receipts]
        if any(not c['routers_unchanged'] or not c['trainable_probe_changed']for c in checks):
            raise ValueError('The score-centering update did not preserve frozen routers and update the backbone')
        receipt={'passed':True,'primary_rl_updates':0,'discarded_optimizer_updates':1,
            'on_policy_score_centering_validated':True,'router_unchanged_after_on_policy_update':True,
            'on_policy_checkpoint_api_reload_passed':True,'durable_checkpoint_readback_verified':True,
            'sampled_topk':32,'rollouts_per_group':8,'retained_groups':8,
            'qualification_only':True,'metrics':scalars(metrics)}
        target=Path(os.environ['EFE_PEAR_QUALIFICATION_DIR'])/'result.json'
        target.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        record('on_policy_score_centering_qualification_passed',**receipt)
        return result


@ray.remote
class QualificationRunner(PearTaskRunner):
    def build_trainer(self,config):return QualificationTrainer(config)


@hydra.main(config_path=str(Path(pear.__file__).resolve().parent/'configs'),config_name='training',version_base=None)
def main(config):
    RunConfig(config).prepare();auto_set_device(config)
    validate_config(config=config,use_reference_policy=need_reference_policy(config),use_critic=need_critic(config))
    run_ppo(config,task_runner_class=QualificationRunner)


if __name__=='__main__':main()
