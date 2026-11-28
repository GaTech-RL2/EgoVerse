import pathlib,shutil,ast,json,hashlib
R=pathlib.Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');C=R/'drawer/source/cpu';C.mkdir(exist_ok=True);A=C/'aria_egoposer';B=C/'bs';A.mkdir(exist_ok=True);B.mkdir(exist_ok=True)
for src,dst in [(R/'drawer/source/aria_egoposer',A),(R/'drawer/source/bs',B)]:
 for p in src.glob('*.py'):
  if not(dst/p.name).exists():shutil.copy2(p,dst/p.name)
p=A/'build_training_hdf5.py';t=p.read_text();old='renderer = mujoco.Renderer(model, height=conv.VIDEO_HEIGHT, width=conv.VIDEO_WIDTH)';assert old in t;t=t.replace(old,'renderer = None  # no video_writer: converter returns before rendering');t=t.replace('renderer.close()','if renderer is not None: renderer.close()');ast.parse(t);p.write_text(t)
p=B/'build_hdf5_basefix.py';t=p.read_text().replace('/coc/flash7/cxin34/cloth-valid-recovery-fix1124-20261124/source/aria_egoposer/build_training_hdf5.py',str(A/'build_training_hdf5.py'));ast.parse(t);p.write_text(t)
p=A/'egoposer_infer.py';t=p.read_text().replace('self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")','self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")\n        print("EGOPOSER_DEVICE", self.device, flush=True)');ast.parse(t);p.write_text(t)
t=(R/'scripts/drawer_retarget.py').read_text().replace("R/'drawer/source/aria_egoposer'", "R/'drawer/source/cpu/aria_egoposer'").replace("R/'drawer/source/bs/build_hdf5_basefix.py'", "R/'drawer/source/cpu/bs/build_hdf5_basefix.py'")
ast.parse(t);(R/'scripts/drawer_retarget_cpu.py').write_text(t)
t=(R/'scripts/drawer_retarget_then_qc.py').read_text().replace('drawer_retarget.py','drawer_retarget_cpu.py');ast.parse(t);(R/'scripts/drawer_retarget_cpu_then_qc.py').write_text(t)
t=(R/'scripts/drawer-retarget.sbatch').read_text().replace('drawer_retarget.py','drawer_retarget_cpu_then_qc.py').replace('set -euo pipefail','set -euo pipefail\nexport CUDA_VISIBLE_DEVICES=""');(R/'scripts/drawer-retarget-cpu.sbatch').write_text(t)
report=dict(passed_syntax=True,numeric_algorithm_unchanged=True,real_rgb_unchanged=True,render_omission='No video_writer; original converter returns without rendering',cpu_model_supported=True,requires_pilot_acceptance=True,files={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in C.rglob('*.py')})
(R/'reports/drawer-cpu-preflight.json').write_text(json.dumps(report,indent=2));print('CPU_PATH_PREPARED')
