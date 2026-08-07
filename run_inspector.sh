#!/usr/bin/env bash
# Launch the latent inspector against a chosen run.
# Run from the EgoVerse3 repo root.
set -euo pipefail

ROOT_DIR="logs/pick_place/latent_eval"
ZARR_ROOT="/storage/project/r-dxu345-0/agao81/pick_place"
SAMPLE="${SAMPLE:-5000}"
HOST="${HOST:-127.0.0.1}"

# Auto-pick a free port if $PORT isn't set. Walks 8050..8199 and binds
# to the first one the kernel accepts. If you really want a specific
# port, set PORT explicitly (e.g. PORT=9090 ./run_inspector.sh).
if [[ -z "${PORT:-}" ]]; then
    PORT=$(python -c "
import socket
for p in range(8050, 8200):
    s = socket.socket()
    try:
        s.bind(('$HOST', p))
        print(p)
        break
    except OSError:
        continue
    finally:
        s.close()
")
    if [[ -z "$PORT" ]]; then
        echo "ERROR: no free port in 8050..8199" >&2
        exit 1
    fi
fi

if [[ ! -d "$ROOT_DIR" ]]; then
    echo "ERROR: root dir not found: $ROOT_DIR" >&2
    echo "(run this script from the EgoVerse3 repo root)" >&2
    exit 1
fi

echo ">> scanning runs under: $ROOT_DIR"
echo ">> serving on: http://$HOST:$PORT"
echo ">> tunnel from laptop with:"
echo "     ssh -N -L 8000:localhost:$PORT -J paphiwetsa3@login-phoenix-gnr-3.pace.gatech.edu paphiwetsa3@$(hostname -s)"
exec python -m egomimic.scripts.data_visualization.latent_inspector \
    --root "$ROOT_DIR" \
    --zarr-root "$ZARR_ROOT" \
    --sample "$SAMPLE" \
    --host "$HOST" \
    --port "$PORT"
