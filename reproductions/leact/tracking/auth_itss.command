#!/bin/bash
set -euo pipefail
umask 077
control_dir="/tmp/leact-itss-ssh-$(id -un)"
mkdir -p "$control_dir"
chmod 700 "$control_dir"
echo 'LeAct tracker: authenticate to your existing ITSS instance.'
echo 'Enter your NVIDIA password only at the SSH prompt and complete Silverfort if prompted.'
echo 'No password is recorded by this script.'
if ssh -S "$control_dir/control" -O check rpunamiya@yam-coverage-rpunamiya.nvidia.com 2>/dev/null; then
    echo 'An authenticated connection is already available.'
else
    ssh -M -S "$control_dir/control" -o ControlPersist=24h \
        -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
        -o NumberOfPasswordPrompts=1 -o StrictHostKeyChecking=yes \
        -o ConnectTimeout=15 -fN rpunamiya@yam-coverage-rpunamiya.nvidia.com
fi
echo 'Authentication is ready. Codex can now deploy the LeAct tracker.'
