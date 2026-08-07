#!/bin/bash
# Build the self-contained chunkviz.html from an exported npz and (re)serve it
# over HTTP for SSH tunnelling. Run on a PACE login node.
#
#   bash egomimic/eval/explorer/serve.sh [NPZ] [PORT]
#
# Then, on your LAPTOP (round-robin-proof — targets THIS login node by name):
#   ssh -L <PORT>:<thishost>:<PORT> pacerh9
#   open http://localhost:<PORT>/chunkviz.html
#
# The exact tunnel command is printed at the end with the real hostname/port.
set -euo pipefail
REPO=/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm
NPZ="${1:-$REPO/fixsweep/chunkviz_data.npz}"
PORT="${2:-8765}"
SERVE_DIR="$(dirname "$NPZ")"
HTML="$SERVE_DIR/chunkviz.html"

cd "$REPO"
echo "[build] $NPZ -> $HTML"
PYTHONPATH=. .venv/bin/python -m egomimic.eval.explorer.build_html \
    --data "$NPZ" --out "$HTML"

echo "[serve] (re)starting no-cache server on :$PORT"
pkill -f "http.server $PORT" 2>/dev/null || true
pkill -f "nocache_server.py $PORT" 2>/dev/null || true
sleep 1
cd "$SERVE_DIR"
setsid python3 "$REPO/egomimic/eval/explorer/nocache_server.py" "$PORT" \
    >"$HOME/chunkviz_server.log" 2>&1 < /dev/null &
sleep 2
H=$(hostname)
curl -s -o /dev/null -w "[check] http://$H:$PORT/chunkviz.html -> HTTP %{http_code} (%{size_download}b)\n" \
    "http://$H:$PORT/chunkviz.html" || true
echo
echo "================ TUNNEL (run on your laptop) ================"
echo "  ssh -L $PORT:$H:$PORT pacerh9"
echo "  open http://localhost:$PORT/chunkviz.html"
echo "============================================================"
