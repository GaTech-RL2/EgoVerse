#!/bin/bash
set -euo pipefail
umask 077
control_dir="/tmp/reasoning-itss-ssh-$(id -un)"
mkdir -p "$control_dir"
chmod 700 "$control_dir"
echo 'Authenticate through Entra SSO to publish your reasoning dashboard.'
echo 'A browser opens if your Teleport certificate needs renewal. No password is collected here.'
if ssh -S "$control_dir/control" -O check rpunamiya@rpunamiya.nvidia.com 2>/dev/null; then
    echo 'An authenticated connection is already available.'
else
    teleport_client="$(command -v tsh)"
    "$teleport_client" login --proxy=nv-prd-it.teleport.sh --auth=entra
    "$teleport_client" config --proxy=nv-prd-it.teleport.sh > "$control_dir/teleport.conf"
    if ! "$teleport_client" ssh --request-mode=off --no-relogin \
            rpunamiya@rpunamiya.nvidia.com true; then
        echo 'The VM must finish provisioning and register in Teleport before deployment.' >&2
        exit 1
    fi
    ssh -F "$control_dir/teleport.conf" -M -S "$control_dir/control" -o ControlPersist=24h \
        -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
        -o BatchMode=yes -o StrictHostKeyChecking=yes \
        -o ConnectTimeout=15 -fN rpunamiya@rpunamiya.nvidia.com.nv-prd-it.teleport.sh
fi
echo 'Authentication is ready. Codex can now publish the reasoning dashboard.'
