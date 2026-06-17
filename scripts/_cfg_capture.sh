#!/usr/bin/env bash
# Capture resolved YAML for every selectable leaf config in each group.
# Underscore-prefixed files are bases (not primary-selectable) -> skipped;
# they are verified transitively through the leaves that compose them.
# Usage: scripts/_cfg_capture.sh <OUTDIR>
set -u
OUTDIR="$1"
ROOT="egomimic/hydra_configs"
CFG_GROUPS="model data evaluator callbacks trainer logger"
FAILLOG="$OUTDIR/_failures.txt"
mkdir -p "$OUTDIR"
: > "$FAILLOG"
for g in $CFG_GROUPS; do
  mkdir -p "$OUTDIR/$g"
  for f in "$ROOT/$g"/*.yaml; do
    base="$(basename "$f" .yaml)"
    case "$base" in _*) continue;; esac
    out="$OUTDIR/$g/$base.yaml"
    if ! PYTHONPATH=. .venv/bin/python scripts/_cfg_resolve.py "$g" "$base" > "$out" 2> "$out.err"; then
      echo "$g/$base" >> "$FAILLOG"
      rm -f "$out"
    fi
    [ -s "$out.err" ] || rm -f "$out.err"
  done
done
echo "captured -> $OUTDIR"
echo "failures:"; cat "$FAILLOG"
