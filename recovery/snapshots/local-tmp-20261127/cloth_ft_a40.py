import subprocess,json,pathlib,time
S='/opt/slurm/Ubuntu-24.04/current/bin/scontrol';p=pathlib.Path('/coc/flash7/cxin34/cloth-valid-recovery-fix1124-20261124/ft/reports/gpu-update.json')
assert not p.exists()
before=subprocess.check_output([S,'show','job','4855714','-o'],text=True)
assert 'JobState=PENDING' in before and 'cloth50-recovered-fix1124' in before
z=subprocess.run([S,'update','JobId=4855714','Gres=gpu:a40:1'],capture_output=True,text=True)
after=subprocess.check_output([S,'show','job','4855714','-o'],text=True)
p.write_text(json.dumps(dict(before=before,returncode=z.returncode,stdout=z.stdout,stderr=z.stderr,after=after,time=time.time(),reason='Only new pending job moved to supported A40; existing L40S training untouched; same jobID, no resubmission'),indent=2))
print(z.returncode,z.stdout,z.stderr,after)
