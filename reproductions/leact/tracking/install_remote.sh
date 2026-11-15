#!/usr/bin/env bash
# Run on ITSS after extracting a verified release into ~/leact-tracker/releases.
set -Eeuo pipefail
umask 077
release="${1:?release directory}"
base="$HOME/leact-tracker"
[[ "$release" == "$base"/releases/* && "$release" != *'/../'* ]]
test -f "$base/.leact-tracker-managed"
test -f "$release/server.py"
mkdir -p "$base/data" "$base/logs"
if [[ ! -f "$base/emimic/bin/activate" ]]; then
    python3 -m venv --without-pip "$base/emimic"
fi
cd "$base"
source emimic/bin/activate
python "$release/server.py" import --db data/experiments.sqlite3 --snapshot "$release/snapshot.json"
python "$release/server.py" import-records --db data/experiments.sqlite3 --snapshot "$release/records.jsonl"

# Port ownership is checked before selecting or restarting a release.
python - <<'PY'
import json, socket, urllib.request
with socket.socket() as listener:
    try:
        listener.bind(('0.0.0.0', 8126))
    except OSError:
        with urllib.request.urlopen('http://127.0.0.1:8126/api/health', timeout=3) as response:
            if json.load(response).get('service') != 'leact-tracker':
                raise SystemExit('Port 8126 belongs to a different service')
PY
ln -sfn "$release" "$base/current"
cat > "$base/run.sh" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/leact-tracker"
source emimic/bin/activate
exec python -u current/server.py serve --db data/experiments.sqlite3 --host 0.0.0.0 --port 8126
RUN
chmod 700 "$base/run.sh"

if systemctl --user show-environment >/dev/null 2>&1 && \
   [[ "$(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null)" == yes ]]; then
    mkdir -p "$HOME/.config/systemd/user"
    cat > "$HOME/.config/systemd/user/leact-tracker.service" <<UNIT
[Unit]
Description=LeAct experiment tracker
After=network.target
[Service]
Type=simple
WorkingDirectory=$base
ExecStart=$base/run.sh
Restart=on-failure
RestartSec=5
UMask=0077
[Install]
WantedBy=default.target
UNIT
    systemctl --user daemon-reload
    systemctl --user enable leact-tracker.service
    systemctl --user restart leact-tracker.service
    printf 'systemd-user-with-linger\n' > "$base/service-manager.txt"
else
    # A per-user cron entry keeps the service available after logout/reboot.
    # Other applications' processes, ports and crontab entries are preserved.
    command -v crontab >/dev/null
    command -v flock >/dev/null
    cat > "$base/ensure.sh" <<'ENSURE'
#!/usr/bin/env bash
set -euo pipefail
base="$HOME/leact-tracker"
exec 9>"$base/service.lock"
flock -n 9 || exit 0
if [[ -f "$base/service.pid" ]]; then
    read -r pid < "$base/service.pid"
    if [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] && \
       tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq 'current/server.py serve --db data/experiments.sqlite3'; then
        exit 0
    fi
fi
nohup "$base/run.sh" >> "$base/logs/server.log" 2>&1 < /dev/null 9>&- &
echo "$!" > "$base/service.pid"
ENSURE
    chmod 700 "$base/ensure.sh"
    # Stop only a previously installed LeAct daemon when updating its release.
    if [[ -f "$base/service.pid" ]]; then
        read -r pid < "$base/service.pid"
        if [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] && \
           tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq 'current/server.py serve --db data/experiments.sqlite3'; then
            kill "$pid"
            for attempt in $(seq 1 20); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 0.25
            done
        fi
    fi
    python - <<'PY'
import os, shlex, subprocess
existing = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
if existing.returncode and 'no crontab' not in existing.stderr.lower():
    raise SystemExit(existing.stderr or 'Cannot inspect existing crontab')
lines = [line for line in existing.stdout.splitlines() if '# leact-tracker-managed' not in line]
base = os.path.expanduser('~/leact-tracker')
lines.append('* * * * * '+shlex.quote(base+'/ensure.sh')+' >> '+shlex.quote(base+'/logs/supervisor.log')+' 2>&1 # leact-tracker-managed')
subprocess.run(['crontab', '-'], input='\n'.join(lines)+'\n', text=True, check=True)
PY
    "$base/ensure.sh"
    printf 'user-cron-supervisor\n' > "$base/service-manager.txt"
fi
for attempt in $(seq 1 15); do
    if curl --fail --silent http://127.0.0.1:8126/api/health; then
        printf '\n'
        cat "$base/service-manager.txt"
        exit 0
    fi
    sleep 1
done
echo 'Tracker did not pass its health check' >&2
exit 1
