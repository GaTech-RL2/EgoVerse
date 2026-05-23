"""Interactive UMAP/t-SNE explorer with click-to-image / click-to-language.

Spawns a local Dash web app (default http://localhost:8050) that renders a
3D scatter of latent UMAP coords from a per-layer CSV. Click a point and
the right pane shows:
  - the JPEG frame for that (episode, frame_idx),
  - the language prompt for that frame (if available),
  - metadata (video_hash, frame_idx, token_idx, embodiment).

Requirements (one-time):
    pip install dash

Usage:
    python egomimic/scripts/data_visualization/latent_inspector.py \\
        --latent-dir logs/pick_place/latent_eval/.../latents/epoch_0 \\
        --zarr-root /storage/project/r-dxu345-0/agao81/pick_place \\
        --sample 5000 \\
        --port 8050

Then open http://localhost:8050 (or set up an SSH tunnel if you're running
on a remote node: `ssh -N -L <your-computer-port>:localhost:8050 <node>`).

Thin CLI entry point — see `inspector_lib/` for the actual implementation.
"""

from __future__ import annotations

import argparse
import logging
import os
import os as _os
import sys as _sys

# When run as a script (`python latent_inspector.py`), ensure this dir is on
# sys.path so the `inspector_lib` sibling package resolves. When imported as
# part of `egomimic.scripts.data_visualization`, the package import below
# would conflict with the sibling `egomimic/scripts/data_visualization.py`
# module, so we deliberately import via the local `inspector_lib` name.
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

from inspector_lib.app import build_app  # noqa: E402
from inspector_lib.io import (  # noqa: E402
    discover_runs,
    list_layer_csvs,
)

logger = logging.getLogger("latent_inspector")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--root",
        default=None,
        help="Top-level directory to scan for per-run latent CSVs "
        "(e.g. logs/pick_place/latent_eval). Every subdir "
        "containing per-layer *.csv files becomes a 'Run' "
        "in the dropdown, sorted by name.",
    )
    src.add_argument(
        "--latent-dir",
        default=None,
        help="Single per-run CSV directory (e.g. logs/.../latents/epoch_0). "
        "Use this if you only want one run available.",
    )
    p.add_argument(
        "--zarr-root",
        required=True,
        help="Root dir containing per-episode zarrs (e.g. /storage/.../agao81/pick_place).",
    )
    p.add_argument(
        "--image-key",
        default="images.front_1",
        help="Zarr key for the front camera images.",
    )
    p.add_argument(
        "--lang-key",
        default=None,
        help="Zarr key (or dotted path like 'annotations/lang') where the "
        "language prompt lives. If unset, the inspector tries common "
        "names and auto-walks the zarr for any string-typed array.",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=5000,
        help="Default points per layer (changeable in the UI).",
    )
    p.add_argument("--port", type=int, default=8050)
    p.add_argument(
        "--host", default="0.0.0.0", help="Set to 127.0.0.1 to bind localhost only."
    )
    args = p.parse_args()

    if args.root:
        runs = discover_runs(args.root)
        if not runs:
            raise SystemExit(f"No per-layer CSVs found anywhere under {args.root}")
        logger.info("Discovered %d runs under %s:", len(runs), args.root)
        for disp, _ in runs:
            logger.info("  %s", disp)
    else:
        # Single-run mode: synthesize a one-element runs list using the
        # latent dir's basename as the display label.
        if not list_layer_csvs(args.latent_dir):
            raise SystemExit(f"No CSVs found in {args.latent_dir}")
        runs = [
            (
                os.path.basename(args.latent_dir.rstrip("/")),
                os.path.abspath(args.latent_dir),
            )
        ]

    app = build_app(
        runs=runs,
        zarr_root=args.zarr_root,
        image_key=args.image_key,
        default_sample=args.sample,
        lang_key=args.lang_key,
    )
    logger.info("Starting Dash on http://%s:%d", args.host, args.port)
    # threaded=True so /thumbnail requests fan out instead of blocking each
    # other; the grid fires one request per visible card and the default
    # single-thread server serializes them.
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
