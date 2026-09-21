#!/bin/bash
# Serve one abl0921 checkpoint on its assigned port (one server per policy;
# the client side picks the same port via ABL=<variant>).
#
#   bash serve_abl_0921.sh V0            # port 8100, 16 denoising steps
#   bash serve_abl_0921.sh H5 --num-inference-steps 8   # extra args pass through
#
# Suggested first session (spec: offline MAE in parens, V0=0.1405):
#   V0 (control — must behave like known-good dp3c_dual first), H5 (0.1311),
#   G6 (0.1360) or G7 (0.1359), D3 (0.1356), G1 (0.1303, proprio-leaky probe),
#   then A5 / others.
set -eu
cd "$(dirname "$0")"
PY=emimic/bin/python
V="${1:?usage: serve_abl_0921.sh <variant> [extra serve_policy args]}"
shift || true

declare -A PORTS=(
  [V0]=8100 [A1]=8101 [A2]=8102 [A3]=8103 [A4]=8104 [A5]=8105
  [B2]=8106 [C3]=8107 [C4]=8108 [D1]=8109 [D3]=8110 [E4]=8111
  [G1]=8112 [G4]=8113 [G6]=8114 [G7]=8115 [H3]=8116 [H4]=8117 [H5]=8118
  [J1]=8119 [J2]=8120 [J3]=8121 [J3b]=8122 [J4]=8123 [J4b]=8124 [J5]=8125
)
[[ -z "${PORTS[$V]:-}" ]] && { echo "unknown variant '$V'" >&2; exit 1; }
CK="checkpoints/abl0921/${V}.ckpt"
[[ -f "$CK" ]] || { echo "missing $CK — run pull_abl_0921.sh" >&2; exit 1; }
echo "[serve] abl0921 $V -> port ${PORTS[$V]} (16 denoising steps unless overridden)"
exec "$PY" egomimic/scripts/serve_policy.py --checkpoint "$CK" --port "${PORTS[$V]}" "$@"
