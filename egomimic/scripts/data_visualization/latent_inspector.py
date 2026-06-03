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

from inspector_lib.app import build_app, build_dataset_app  # noqa: E402
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
    src.add_argument(
        "--dataset-path",
        default=None,
        help="Launch the DATASET BROWSER instead of the latent scatter: a "
        "folder of per-episode zarrs. Browse/search any episode, overlay "
        "actions (cartesian/orientation/keypoint), toggle annotations.",
    )
    p.add_argument(
        "--zarr-root",
        default=None,
        help="Root dir containing per-episode zarrs (required for the latent "
        "scatter/browser; not needed with --dataset-path).",
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
    p.add_argument(
        "--pair-rank",
        type=int,
        default=0,
        metavar="N",
        help="Report mode (no server): for each layer, sample N rows and "
        "print the average place of the perfect-pair action among "
        "opposite-embodiment actions (1 = nearest). E.g. --pair-rank 1000.",
    )
    p.add_argument(
        "--pair-rank-space",
        default="umap",
        choices=["umap", "pca_umap", "tsne2d"],
        help="Coordinate space for the --pair-rank distances (CSV-baked "
        "reductions; same spaces the scatter view plots). Ignored when "
        "--pair-rank-sweep is set.",
    )
    p.add_argument(
        "--pair-rank-seed",
        type=int,
        default=0,
        help="Sampling seed for --pair-rank.",
    )
    p.add_argument(
        "--pair-rank-sweep",
        action="store_true",
        help="Run the embedding-space sweep instead of the CSV-coord report: "
        "for a few focus layers, report MEAN place in raw-key space, PCA "
        "space, and PCA with the top-k aria-vs-eva Fisher dims removed "
        "(removal swept in steps of --pair-rank-drop-step). Needs "
        "<layer>_keys.pt files in each run dir.",
    )
    p.add_argument(
        "--pair-rank-layers",
        default="default",
        help="Comma-separated layer names for --pair-rank-sweep, or 'default' "
        "(expert 0/12/17 + paligemma 0/17 combined & lang) or 'all'.",
    )
    p.add_argument(
        "--pair-rank-pca",
        type=int,
        default=50,
        help="PCA components for the --pair-rank-sweep PCA / Fisher spaces.",
    )
    p.add_argument(
        "--pair-rank-drop-step",
        type=int,
        default=10,
        help="Fisher-removal step for --pair-rank-sweep: sweep drop_k = step, "
        "2*step, ... up to (but excluding) --pair-rank-pca.",
    )
    args = p.parse_args()

    # ----- Dataset Browser mode (no latents/zarr-root needed) -------------
    if args.dataset_path:
        app = build_dataset_app(
            dataset_path=os.path.abspath(args.dataset_path.rstrip("/")),
            image_key=args.image_key,
            lang_key=args.lang_key,
        )
        logger.info("Starting Dataset Browser on http://%s:%d", args.host, args.port)
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
        return

    if not args.zarr_root:
        raise SystemExit("--zarr-root is required for the latent scatter/browser "
                         "(omit it only with --dataset-path).")

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

    if args.pair_rank > 0:
        if args.pair_rank_sweep:
            from inspector_lib.pair_rank import (
                DEFAULT_SWEEP_LAYERS,
                pair_rank_sweep,
            )

            if args.pair_rank_layers in ("default", ""):
                layers = list(DEFAULT_SWEEP_LAYERS)
            elif args.pair_rank_layers == "all":
                layers = None  # resolved per-run below
            else:
                layers = [
                    s.strip() for s in args.pair_rank_layers.split(",") if s.strip()
                ]
            pca = int(args.pair_rank_pca)
            step = max(1, int(args.pair_rank_drop_step))
            drop_steps = tuple(range(step, pca, step))
            for disp, run_dir in runs:
                print(f"\n===== run: {disp} =====")
                run_layers = layers
                if run_layers is None:
                    from inspector_lib.caches import LayerStore

                    run_layers = LayerStore().layers_for(run_dir)
                pair_rank_sweep(
                    run_dir,
                    args.zarr_root,
                    n_samples=args.pair_rank,
                    seed=args.pair_rank_seed,
                    layers=run_layers,
                    n_components=pca,
                    drop_steps=drop_steps,
                )
            return

        from inspector_lib.pair_rank import pair_rank_report

        for disp, run_dir in runs:
            print(f"\n===== run: {disp} =====")
            pair_rank_report(
                run_dir,
                args.zarr_root,
                n_samples=args.pair_rank,
                space=args.pair_rank_space,
                seed=args.pair_rank_seed,
            )
        return

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
