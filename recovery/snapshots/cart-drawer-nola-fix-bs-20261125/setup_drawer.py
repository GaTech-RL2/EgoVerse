import ast,shutil,json
from pathlib import Path
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125')
src=Path('/coc/flash7/cxin34/egoa-target-retarget-fix-20261124/source/aria_egoposer');dst=R/'drawer/source/aria_egoposer'
if not dst.exists():shutil.copytree(src,dst,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
original=Path('/coc/flash7/cxin34/human-ondevice-pilot-20261110/source/sew/projects/xhand_teleop/aria_egoposer/aria_vrs_reader.py')
s=original.read_text()
s=s.replace('    R_dev_head = R_dev_rgb @ CAM2HEAD.T','''    T_dev_cpf=np.asarray(dp.get_device_calibration().get_transform_device_cpf().to_matrix())
    FLU=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
    R_dev_head=T_dev_cpf[:3,:3]@FLU''')
s=s.replace('    head_p = p_dev_g','    head_p = p_dev_g + np.einsum("nij,j->ni",R_world_dev_g,T_dev_cpf[:3,3])')
s=s.replace('n=n, vrs=str(vrs_path))','n=n, vrs=str(vrs_path), dev_p=p_dev_g, dev_R=R_world_dev_g, cpf_p=head_p.copy(), T_device_cpf=T_dev_cpf)')
s=s.replace('valid_g = s["valid"][near]','valid_g = s["valid"][j] & s["valid"][jm] & ((t_hand[j]-t_hand[jm])<=50000)')
ast.parse(s);(dst/'aria_vrs_reader.py').write_text(s)
bs=R/'drawer/source/bs'
if not bs.exists():shutil.copytree('/coc/flash7/scratch/cxin34/cloth50-recovered-bs-20261125/source',bs,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
(R/'drawer/reports').mkdir(exist_ok=True)
print('OWN_ONDEVICE_CPF_READER_PREPARED')
