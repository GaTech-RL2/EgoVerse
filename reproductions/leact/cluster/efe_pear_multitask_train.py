"""PEAR's trainer with explicit retained-family quotas and SGS reset sampling.

This entry point keeps PEAR's actor, score-centering loss, native verifier,
asynchronous rollout and checkpoint engine. It requires admitted reset banks;
merely enumerating simulator names cannot enable primary training.
"""
from collections import Counter
import json
import os
from pathlib import Path

import hydra
import pear
import ray
import torch
import transfer_queue as tq
from tensordict import TensorDict

from pear.trainer.config import RunConfig
from pear.trainer.replay_buffer import PearReplayBuffer
from pear.trainer.runner import PearTaskRunner
from pear.trainer.trainer import PearTrainer
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.device import auto_set_device
from verl.utils import tensordict_utils as tu

from efe_pear_curriculum import retained_selection, token_mean_factors
from efe_pear_curriculum_state import CurriculumState, identity
from efe_pear_telemetry import Telemetry, reward_metrics


class MultitaskReplayBuffer(PearReplayBuffer):
    def __init__(self, *, curriculum, telemetry, **kwargs):
        self.curriculum=curriculum
        self.telemetry=telemetry
        super().__init__(**kwargs)
        if self.rollout_n!=8 or self.filter_groups_metric!='reward':
            raise ValueError('The curriculum requires eight-rollout binary-reward groups')

    def _on_group_judged(self, partition_id, uid, values):
        super()._on_group_judged(partition_id,uid,values)
        if partition_id=='train':
            self.curriculum.observe(uid,values)
            g=self.curriculum.groups[uid]
            self.telemetry.group(g,self.curriculum.configs[g['config_id']])

    def _selection(self, keys):
        return retained_selection([self.curriculum.groups[uid] for uid in keys],self.curriculum.quotas,self.curriculum.stage)

    def _has_enough_samples(self, global_steps, partition_id, batch_size, sampleable_keys):
        enough=super()._has_enough_samples(global_steps,partition_id,batch_size,sampleable_keys)
        if partition_id=='val':return enough
        if batch_size!=128:raise ValueError('The authorized retained batch is 128 groups')
        return enough and self._selection(sampleable_keys) is not None

    def _select_prompt_uids(self, partition_id, sampleable_keys, batch_size):
        if partition_id=='val':return super()._select_prompt_uids(partition_id,sampleable_keys,batch_size)
        selected=self._selection(sampleable_keys)
        if selected is None:raise RuntimeError('Attempted to train without every family quota')
        return selected,dict(self.partitions[partition_id]),dict(self.prompt_global_steps[partition_id])

    def family_counts(self):
        def count(uids, usable_only=False):
            result=Counter()
            for uid in uids:
                g=self.curriculum.groups[uid]
                if usable_only and (not g['infrastructure_ok'] or len(set(g['outcomes']))<2):continue
                result[g['family']]+=1
            return result
        return (count(self.finished_keys['train'],True),
                count(self.pending_keys['train']|self.running_keys['train']))

    def _refill_free_room(self, partition_id):
        # PEAR calls refill before its sample-ready check. Avoid extending a
        # finished batch simply because the sampler has capacity for more work.
        if partition_id=='train':
            ready,_=self.family_counts()
            if all(ready[f]>=n for f,n in self.curriculum.quotas.items()):return False
        return super()._refill_free_room(partition_id)

    def sample(self, global_steps, partition_id, batch_size):
        batch,metrics=super().sample(global_steps,partition_id,batch_size)
        if partition_id=='train':
            for family,n in self.curriculum.quotas.items():
                metrics[f'curriculum/retained_groups/{family}']=n
                metrics[f'curriculum/loss_weight/{family}']=self.curriculum.loss_weights[family]
            metrics['curriculum/adaptations_applied']=len(self.curriculum.adaptations)
        return batch,metrics


class MultitaskTrainer(PearTrainer):
    def __init__(self, config):
        manifest=json.loads(Path(os.environ['EFE_PEAR_CURRICULUM']).read_text())
        self.curriculum=CurriculumState(manifest,seed=int(config.data.get('seed',1701)))
        if self.curriculum.stage!='expanded_full':
            raise ValueError('Primary RL requires every authorized control and reasoning family')
        self.telemetry=Telemetry(os.environ['EFE_PEAR_METRICS_DIR'],
                                 str(config.trainer.experiment_name),heartbeat=True)
        self.telemetry.metadata({'model_revision':os.environ.get('MODEL_REVISION'),
            'router_policy':'frozen_expert_selection_router'if str(config.actor_rollout_ref.actor.fsdp_config.kernels).endswith('FrozenRouterTrainingKernels')else'not_applicable',
            'retained_groups':int(config.data.train_batch_size),'rollouts_per_group':int(config.actor_rollout_ref.rollout.n),
            'learning_rate':float(config.actor_rollout_ref.actor.optim.lr),'parameter_update':'full_finetuning',
            'ppo_epochs':int(config.actor_rollout_ref.actor.ppo_epochs),
            'score_centering':bool(config.actor_rollout_ref.actor.policy_loss.rollout_correction.score_centering),
            'entropy_coefficient':float(config.actor_rollout_ref.actor.entropy_coeff),
            'kl_penalty':bool(config.actor_rollout_ref.actor.use_kl_loss or config.algorithm.use_kl_in_reward),
            'family_quotas':self.curriculum.quotas,'curriculum_sha256':self.curriculum.manifest_sha256,
            'curriculum_stage':self.curriculum.stage,
            'workflow':os.environ.get('EFE_WORKFLOW_ID'),'sgs_uniform_fraction':.2})
        self._row_by_config=None
        super().__init__(config)
        if (int(config.data.train_batch_size)!=128 or int(config.actor_rollout_ref.rollout.n)!=8 or
            int(config.actor_rollout_ref.actor.ppo_epochs)!=1 or
            config.actor_rollout_ref.actor.loss_agg_mode!='token-mean'):
            raise ValueError('Curriculum reduction requires one update of the full 128x8 token-mean batch')

    def _replay_buffer_class(self, sampler_cls):
        super()._replay_buffer_class(sampler_cls)
        return MultitaskReplayBuffer

    def _replay_buffer_kwargs(self, kwargs):
        return {**super()._replay_buffer_kwargs(kwargs),'curriculum':self.curriculum,'telemetry':self.telemetry}

    def _index_rows(self):
        expected={identity(c['env_class'],c['env_args']):key for key,c in self.curriculum.configs.items()}
        mapping={}
        for i,row in enumerate(self.train_dataset.dataframe):
            extra=row['extra_info'];tools=extra['tool_selection']
            if len(tools)!=1:raise ValueError('Exactly one simulator tool is required per episode')
            config_id=expected.get(identity(tools[0],json.loads(extra['env'])))
            if config_id is None or config_id in mapping:
                raise ValueError('Episode files differ from the frozen admitted curriculum')
            mapping[config_id]=i
        if set(mapping)!=set(self.curriculum.configs):
            raise ValueError('An admitted curriculum reset is missing from the episode files')
        self._row_by_config=mapping

    def _next_dataloader_batch(self):
        # The separate-async trainer fetches one prompt at a time. Its checkpoint
        # still saves its dataloader; this sampler additionally saves all of its
        # RNG, histories, reset counts and group identities in the same step.
        if self._row_by_config is None:self._index_rows()
        ready,inflight=self.replay_buffer.family_counts()
        key=self.curriculum.draw(ready,inflight,step=self.global_steps)
        row=self.train_dataset[self._row_by_config[key]]
        row['extra_info']={**row['extra_info'],'curriculum_config_id':key,
                           'curriculum_dispatch':self.curriculum.last_draw}
        return collate_fn([row])

    def _prompt_tags(self,batch):
        tags=super()._prompt_tags(batch)
        extras=tu.get(batch,'extra_info')
        for uid,extra,tag in zip(list(batch['uid']),extras,tags,strict=True):
            key=extra['curriculum_config_id']
            self.curriculum.register(str(uid),key,self.global_steps)
            self.telemetry.step=int(self.global_steps)
            self.telemetry.dispatch(str(uid),extra['curriculum_dispatch'])
            tag.update(curriculum_config_id=key,curriculum_family=self.curriculum.configs[key]['family'])
        return tags

    def _compute_advantage(self,batch,metrics):
        batch=super()._compute_advantage(batch,metrics)
        if batch.partition_id!='train':return batch
        # All DP ranks' trajectories are still in this one global TransferQueue
        # batch. Count generated tokens here, before the actor partitions it.
        data=tq.kv_batch_get(keys=batch.keys,partition_id=batch.partition_id,
                             select_fields=['advantages','response_mask'])
        families=[self.curriculum.groups[key.split('_',1)[0]]['family'] for key in batch.keys]
        counts=Counter()
        for family,mask in zip(families,data['response_mask'].unbind(),strict=True):
            counts[family]+=int(mask.sum().item())
        factors=token_mean_factors(dict(counts),self.curriculum.loss_weights)
        scaled=[values*factors[family] for values,family in
                zip(data['advantages'].unbind(),families,strict=True)]
        advantages=torch.nested.as_nested_tensor(scaled,layout=torch.jagged)
        for family,n in counts.items():
            metrics[f'curriculum/generated_tokens/{family}']=n
            metrics[f'curriculum/advantage_scale/{family}']=factors[family]
        # Only advantages change, once and after GRPO normalization. The response
        # mask and score-centering probabilities keep their original meaning.
        return tq.kv_batch_put(keys=batch.keys,partition_id=batch.partition_id,
            fields=TensorDict({'advantages':advantages},batch_size=len(batch)))

    def _compute_metrics(self,batch,metrics,timing_raw,global_steps,epoch):
        result=super()._compute_metrics(batch,metrics,timing_raw,global_steps,epoch)
        metrics.update(reward_metrics(self.curriculum.groups,self.curriculum.configs))
        self.telemetry.metrics(global_steps,metrics)
        return result

    def _save_checkpoint(self):
        folder=Path(self.config.trainer.default_local_dir)/f'global_step_{self.global_steps}'
        folder.mkdir(parents=True,exist_ok=True)
        target=folder/'curriculum.json';temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.curriculum.state_dict())+'\n');temporary.replace(target)
        # The upstream completion pointer is written only after this state and
        # the model/optimizer/TransferQueue are saved.
        super()._save_checkpoint()
        archive_root=os.environ.get('PEAR_CHECKPOINT_ARCHIVE')
        if archive_root:
            from efe_pear_checkpoint_archive import archive
            name='rp_jhh_qwen36_moe_warmstart496_pear_expanded_step'+f'{self.global_steps:06d}'
            manifest=archive(folder,Path(archive_root)/name)
            self.telemetry.emit('checkpoint_archive','archive/'+str(self.global_steps),step=int(self.global_steps),
                                name=name,bytes=manifest['bytes'],readback_verified=True)
        self.telemetry.emit('checkpoint','checkpoint/'+str(self.global_steps),step=int(self.global_steps),
            name='rp_jhh_'+self.telemetry.run_id.replace('-','_')+'_step'+str(self.global_steps),complete=True)

    def _validate(self):
        metrics=super()._validate()
        from efe_pear_telemetry import scalars
        self.telemetry.emit('evaluation','evaluation/'+str(self.global_steps),step=int(self.global_steps),
                            metrics=scalars(metrics))
        return metrics

    def _load_checkpoint(self):
        super()._load_checkpoint()
        if self.global_steps:
            if self.config.trainer.resume_mode=='resume_path':
                folder=Path(self.config.trainer.resume_from_path)
            else:folder=Path(self.config.trainer.default_local_dir)/f'global_step_{self.global_steps}'
            self.curriculum.restore(json.loads((folder/'curriculum.json').read_text()))


@ray.remote
class MultitaskRunner(PearTaskRunner):
    def build_trainer(self,config):return MultitaskTrainer(config)


@hydra.main(config_path=str(Path(pear.__file__).resolve().parent/'configs'),config_name='training',version_base=None)
def main(config):
    RunConfig(config).prepare()
    from efe_pear_launch_contract import validate_driver_files
    validate_driver_files(config);auto_set_device(config)
    validate_config(config=config,use_reference_policy=need_reference_policy(config),use_critic=need_critic(config))
    run_ppo(config,task_runner_class=MultitaskRunner)


if __name__=='__main__':main()
