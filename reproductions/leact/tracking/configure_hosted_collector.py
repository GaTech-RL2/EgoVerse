"""Install the read-only collector on the already authenticated tracker VM."""
import json
from pathlib import Path
import shlex
import subprocess

from .reasoning import BASE,write


REMOTE='''from pathlib import Path
import json,os,subprocess
os.umask(0o077)
base=Path.home()/'leact-tracker';assert(base/'.leact-tracker-managed').is_file()
binary=Path.home()/'.local/share/pear-tracker-osmo/osmo/osmo';assert binary.is_file()
login=Path.home()/'.config/osmo/login.yaml';assert login.is_file();login.chmod(0o600)
script=base/'ensure-collector.sh'
script.write_text("""#!/usr/bin/env bash
set -euo pipefail
base=\"$HOME/leact-tracker\"
exec 9>\"$base/collector-supervisor.lock\"
flock -n 9 || exit 0
if [[ -f \"$base/collector.pid\" ]]; then
 read -r pid < \"$base/collector.pid\"
 if [[ \"$pid\" =~ ^[0-9]+$ && -r /proc/$pid/cmdline ]] && grep -aFq 'current/pear_collect.py' /proc/$pid/cmdline; then
  exit 0
 fi
fi
nohup \"$base/emimic/bin/python\" -u \"$base/current/pear_collect.py\" --db \"$base/data/experiments.sqlite3\" --osmo \"$HOME/.local/share/pear-tracker-osmo/osmo/osmo\" >> \"$base/logs/collector.log\" 2>&1 < /dev/null 9>&- &
echo \"$!\" > \"$base/collector.pid\"
""")
script.chmod(0o700)
existing=subprocess.run(['crontab','-l'],capture_output=True,text=True)
if existing.returncode and 'no crontab'not in existing.stderr.lower():raise RuntimeError('Cannot read user crontab')
lines=[line for line in existing.stdout.splitlines()if '# pear-collector-managed'not in line]
lines.append('* * * * * '+str(script)+' # pear-collector-managed')
subprocess.run(['crontab','-'],input='\\n'.join(lines)+'\\n',text=True,check=True)
subprocess.run([str(script)],check=True)
print(json.dumps({'installed':True,'supervisor':'user cron','mac_required':False}))
'''


def main():
    result=subprocess.run(['ssh','-S','/tmp/reasoning-itss-ssh-rpunamiya/control','-o','BatchMode=yes',
        '-o','StrictHostKeyChecking=yes','rpunamiya@rpunamiya.nvidia.com','python3 -c '+shlex.quote(REMOTE)],
        capture_output=True,text=True,check=True,timeout=45)
    record=json.loads(result.stdout);write(BASE/'artifacts/reasoning-workspace/hosted-collector/configuration.json',record)
    print(json.dumps(record))


if __name__=='__main__':main()
