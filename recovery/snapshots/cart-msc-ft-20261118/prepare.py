import ast,hashlib,json,shutil
from pathlib import Path
R=Path('/coc/flash7/cxin34/cart-msc-ft-20261118')
BASE=Path('/coc/flash7/cxin34/cart-source-finetune-20261115')
payload=Path(__file__).parent
assert not R.exists(),'Reconcile existing root; never overwrite experiments'
R.mkdir()
for d in ['configs','reports','runs','scripts']: (R/d).mkdir()
shutil.copytree(BASE/'campaign',R/'campaign',ignore=shutil.ignore_patterns('__pycache__'))
for d in ['source','assets','releases']:(R/d).symlink_to(BASE/d,target_is_directory=True)
for name in ['env.sh','train.sbatch','formal.sbatch']:shutil.copy2(BASE/'scripts'/name,R/'scripts'/name)
shutil.copy2(BASE/'configs/hd2-reference.yaml',R/'configs/hd2-reference.yaml')
for name in ['photometric.py','pointcept_color.py','POINTCEPT_LICENSE']:shutil.copy2(payload/name,R/'campaign'/name)
p=R/'campaign/policy.py';s=p.read_text()
s=s.replace('    def forward(self,batch):','''    def forward(self,batch):
        if self.training and hasattr(self,'photometric_config'):
            from .photometric import training_forward
            return training_forward(self,batch)
        return self._forward_once(batch)

    def _forward_once(self,batch,return_clean=False):''')
s=s.replace('        return la+aux,metrics','        return (la+aux,metrics,clean) if return_clean else (la+aux,metrics)')
ast.parse(s);p.write_text(s)
p=R/'campaign/train.py';s=p.read_text()
needle='    optimizer=torch.optim.AdamW'
s=s.replace(needle,"    model.photometric_config=cfg['photometric']\n    for encoder in model.policy.encoders.values():encoder.colour_brightness_jitter=0.;encoder.colour_channel_jitter=0.\n"+needle)
s=s.replace("T_max=cfg['steps']","T_max=cfg['scheduler_steps']")
ast.parse(s);p.write_text(s)
cfg=json.loads((BASE/'configs/cart-source-v1-ft-B-p180000-s42.json').read_text())
(R/'runs'/cfg['parent']).symlink_to(BASE/'runs'/cfg['parent'],target_is_directory=True)
runs=[]
for suffix,weight in [('aug',0.),('aug-consistency',.1)]:
    c=dict(cfg,id='cart-B180-msc-'+suffix+'-ft10k-s42',steps=10000,scheduler_steps=60000)
    c['photometric']=dict(recipe='pointcept_msc',brightness=.4,contrast=.4,saturation=.2,hue=.02,p=.8,color_noise_std=.05,color_noise_p=.95,consistency_weight=weight,shared_across_clouds=True,paired_rng=True,gamma=False,original_encoder_color_jitter_disabled=True)
    (R/'configs'/(c['id']+'.json')).write_text(json.dumps(c,indent=2));runs.append(c)
plan=dict(runs=runs,baseline=str(BASE),status='PREPARED',validation='unchanged clean full3027 every5k',
    comparison='Official MSC photometry replaces ALL encoder color jitter; no gamma. Shared per-sample transform of concatenated valid global/local RGB; 60k cosine truncated at10k.',
    parent_sha256=cfg['parent_checkpoint_sha256'])
(R/'reports/plan.json').write_text(json.dumps(plan,indent=2))
print(json.dumps(plan,indent=2))
