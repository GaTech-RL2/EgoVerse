# Source (do not execute) from a Slurm batch script, after wt-env.sh:
#   source ./stage-runtime.sh || exit 1
# Copies the venv, its uv Python and this tree's code to node-local /ephemeral on
# every node of the job, then exports EGOVERSE_PYTHON and EGOVERSE_CODE:
#   "$EGOVERSE_PYTHON" "$EGOVERSE_CODE/egomimic/trainHydra.py" ...
# Imports are thousands of small-file lookups per rank; on /workspace NFS they
# are what livelocks nodes. The venv is copied once per node and reused while
# unchanged. Each sourcing snapshots the code into one tarball on NFS (with its
# git provenance) that every node unpacks; a Slurm requeue reuses it, so it
# reruns the same code. All copies are single sequential tar streams. The cwd
# stays in the tree, so run dirs, logs and checkpoints still land on /workspace.
#
# The work runs in child processes (--snapshot, then --copy on every node), all
# reading the _RT_* paths computed once here: errexit is ignored in a function
# or subshell on the left of ||, which would let a failed copy be marked complete.

_rt_copy() {
    mkdir -p "$_RT_ROOT"
    exec 9>"$_RT_ROOT/.lock"
    flock 9
    # Under the lock, any .partial is left from a killed or failed copy.
    rm -rf "$_RT_ROOT"/env-*.partial "$_RT_ROOT"/code-*.partial
    if [ ! -e "$_RT_ENV/.complete" ]; then
        rm -rf "$_RT_ENV"
        mkdir -p "$_RT_ENV.partial"
        tar -C / -cf - "${_RT_VENV#/}" "${_RT_PYHOME#/}" | tar -C "$_RT_ENV.partial" -xf -
        local bin="$_RT_ENV.partial$_RT_VENV/bin"
        ln -sfn "$_RT_ENV$_RT_PYTHON" "$bin/python"
        sed -i "s#^home = .*#home = $_RT_ENV$_RT_PYHOME/bin#" "$_RT_ENV.partial$_RT_VENV/pyvenv.cfg"
        # Console scripts (torchrun, wandb, ...) would otherwise start the NFS python.
        { grep -lZ -m1 "^#!$_RT_VENV/bin/python" "$bin"/* 2>/dev/null || true; } \
            | xargs -0 -r sed -i "1s|^#!$_RT_VENV/bin/python|#!$_RT_ENV$_RT_VENV/bin/python|"
        mv "$_RT_ENV.partial" "$_RT_ENV"
        if ! "$_RT_ENV$_RT_VENV/bin/python" -c "import sys, torch; assert sys.prefix == '$_RT_ENV$_RT_VENV', sys.prefix"; then
            rm -rf "$_RT_ENV"
            echo "stage-runtime: the staged interpreter at $_RT_ENV does not run" >&2
            return 1
        fi
        touch "$_RT_ENV/.complete"
    fi
    touch "$_RT_ENV"
    if [ ! -d "$_RT_CODE" ]; then
        mkdir -p "$_RT_CODE.partial"
        tar -C "$_RT_CODE.partial" -xf "$_RT_TARBALL"
        mv "$_RT_CODE.partial" "$_RT_CODE"
    fi
    # Jobs share nodes, so only drop copies no queued or running job of ours can use:
    # code dirs are named code-<jobid>-..., and an env is touched by every job that starts on it.
    local live d
    if live="$(squeue -h -u "$(id -un)" -o %A)"; then
        for d in "$_RT_ROOT"/code-*; do
            [ -d "$d" ] || continue
            grep -qx "$(basename "$d" | cut -d- -f2)" <<<"$live" || rm -rf "$d"
        done
    fi
    ls -1dt "$_RT_ROOT"/env-* | tail -n +3 | while read -r d; do
        [ -n "$(find "$d" -maxdepth 0 -mtime +2)" ] && rm -rf "$d"
    done || true
}

_rt_snapshot() {
    local tmp prov
    tmp="$(mktemp -d)"
    prov="$tmp/.git_provenance"
    mkdir "$prov"
    # The same commands as egomimic/utils/git_info.py (DIFF_ARGS, UNTRACKED_ARGS).
    git -C "$_RT_TREE" rev-parse HEAD >"$prov/sha" 2>"$prov/sha.error" && rm "$prov/sha.error" || rm -f "$prov/sha"
    git -C "$_RT_TREE" diff HEAD --submodule=diff --no-ext-diff --no-color >"$prov/diff" 2>"$prov/diff.error" && rm "$prov/diff.error" || rm -f "$prov/diff"
    git -C "$_RT_TREE" ls-files --others --exclude-standard >"$prov/untracked" 2>"$prov/untracked.error" && rm "$prov/untracked.error" || rm -f "$prov/untracked"
    mkdir -p "$(dirname "$_RT_TARBALL")"
    tar -cf "$_RT_TARBALL.partial" -C "$_RT_TREE" --exclude=__pycache__ egomimic external/openpi/src -C "$tmp" .git_provenance
    mv "$_RT_TARBALL.partial" "$_RT_TARBALL"
    rm -rf "$tmp"
    # Generous: a requeued job may have waited in the queue a long time.
    find "$(dirname "$_RT_TARBALL")" -maxdepth 1 -name 'code-*.tar*' -mtime +30 -delete 2>/dev/null || true
}

case "${1:-}" in
    --copy) set -euo pipefail; _rt_copy; exit ;;
    --snapshot) set -euo pipefail; _rt_snapshot; exit ;;
esac
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "stage-runtime.sh must be sourced: source ./stage-runtime.sh" >&2
    exit 1
fi

_rt_main() {
    if [ -z "${SLURM_JOB_ID:-}" ]; then
        echo "stage-runtime.sh: not in a Slurm job; nothing staged" >&2
        return 1
    fi
    local tree="${EGOVERSE_ROOT:-}" venv="${VIRTUAL_ENV:-}" cache="${EGOVERSE_CACHE_DIR:-}"
    if [ -z "$tree" ] || [ -z "$venv" ] || [ -z "$cache" ]; then
        echo "stage-runtime.sh: source wt-env.sh first" >&2
        return 1
    fi
    local python pyhome key
    python="$(readlink -f "$venv/bin/python")" || true
    pyhome="$(dirname "$(dirname "$python")")"
    # A dangling link would make pyhome "." and the copy tar the whole filesystem.
    if [ ! -x "$python" ] || [ "${pyhome#/}" = "$pyhome" ] || [ "$pyhome" = / ]; then
        echo "stage-runtime.sh: $venv/bin/python does not resolve to an interpreter" >&2
        return 1
    fi
    # Directory mtimes change when packages are added or removed, not when a
    # file inside one is edited in place.
    key="$(set -o pipefail; stat -c '%n %Y' "$venv/pyvenv.cfg" "$venv"/lib/python3*/site-packages "$pyhome" | sha256sum | cut -c1-16)" || return 1
    export _RT_ROOT="/ephemeral/loaner-jobs/$(id -u)/runtime"
    export _RT_TREE="$tree" _RT_VENV="$venv" _RT_PYTHON="$python" _RT_PYHOME="$pyhome"
    export _RT_ENV="$_RT_ROOT/env-$key"
    export _RT_TARBALL="$cache/runtime/code-$SLURM_JOB_ID-$(printf %s "$tree" | sha256sum | cut -c1-12).tar"
    if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ] && [ -e "$_RT_TARBALL" ]; then
        echo "stage-runtime: requeue, reusing $_RT_TARBALL"
    else
        bash "$tree/stage-runtime.sh" --snapshot || return 1
    fi
    # A fresh snapshot unpacks to a fresh dir; a requeue finds its old one.
    export _RT_CODE="$_RT_ROOT/$(basename "$_RT_TARBALL" .tar)-$(stat -c %Y "$_RT_TARBALL")"
    srun --nodes="${SLURM_JOB_NUM_NODES:-1}" --ntasks-per-node=1 --ntasks="${SLURM_JOB_NUM_NODES:-1}" \
        -c 1 --mem=0 --gres=none --overlap \
        bash "$tree/stage-runtime.sh" --copy || return 1
    export EGOVERSE_PYTHON="$_RT_ENV$_RT_VENV/bin/python"
    export EGOVERSE_CODE="$_RT_CODE"
    export PYTHONPATH="$_RT_CODE:$_RT_CODE/external/openpi/src"
    export PATH="$_RT_ENV$_RT_VENV/bin:$PATH"
    # torch.compile/triton caches are many small files; keep them off NFS too.
    export TRITON_CACHE_DIR="$_RT_ROOT/triton"
    echo "stage-runtime: $_RT_ENV, $_RT_CODE"
}

unset EGOVERSE_PYTHON EGOVERSE_CODE
_rt_main
_rt_status=$?
[ "$_rt_status" = 0 ] || echo "stage-runtime: staging failed" >&2
unset _RT_ROOT _RT_TREE _RT_VENV _RT_PYTHON _RT_PYHOME _RT_ENV _RT_TARBALL _RT_CODE
unset -f _rt_main _rt_copy _rt_snapshot
return $_rt_status
