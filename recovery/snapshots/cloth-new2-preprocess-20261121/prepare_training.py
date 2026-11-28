from pathlib import Path
import shutil,json
R=Path('/coc/flash7/cxin34/cloth-new2-scratch-20261121');old=Path('/coc/flash7/cxin34/cart-msc-ft-20261118')
for n in ['train.py','contracts.py']:shutil.copy2(R/'scripts'/n,R/'campaign'/n)
shutil.copytree(old/'assets',R/'assets',dirs_exist_ok=True)
shutil.copy2(old/'configs/hd2-reference.yaml',R/'configs/hd2-reference.yaml')
for name in ['huggingface','torch']:
 dest=R/'cache'/name
 if not dest.exists():dest.symlink_to(old/'cache'/name,target_is_directory=True)
c=json.loads((old/'configs/cart-B180-msc-aug-ft10k-s42.json').read_text())
for k in ['parent','parent_step','parent_checkpoint_sha256','source_identity','selection']:c.pop(k,None)
c.update(id='cloth-new2-B-msc-scratch-s42',task='cloth_new2',stage='scratch',steps=180000,scheduler_steps=180000,release_id='cloth-new2-scratch',normalization_fit_partition='target_train',normalization_recipe='hd2_quantile_sparse_minmax_eps1e-6',spatial_augmentation='hd2_original',train_recordings=50,valid_recordings=10)
(R/'configs/cloth-new2-B-msc-scratch-s42.json').write_text(json.dumps(c,indent=2))
print('SCRATCH180K_DRAFT_READY')
