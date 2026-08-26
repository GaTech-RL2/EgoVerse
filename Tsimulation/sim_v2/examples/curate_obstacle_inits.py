"""Build and visualize level-specific ChainGripper obstacle initializations.

Example::

    python -m Tsimulation.sim_v2.examples.curate_obstacle_inits \
      --output obstacle_inits.json \
      --plot obstacle_init_routes.png \
      --silhouette-plot obstacle_init_silhouettes.png
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.patches import Polygon as PlotPolygon
from shapely.geometry import LineString  # noqa: E402

from Tsimulation.sim_v2.collect.obstacle_init import (  # noqa: E402
    DEFAULT_CRITERIA,
    curate_manifest,
    level_entries,
    load_manifest,
    write_manifest,
)
from Tsimulation.sim_v2.pushshapes.obstacles import (  # noqa: E402
    OBSTACLE_LEVELS,
    SKETCH_FAMILY_NAMES,
    WALL_RADIUS,
)
from Tsimulation.sim_v2.pushshapes.shapes import object_polygon  # noqa: E402

WORLD_SIZE = 512.0


def parse_levels(value: str) -> list[int]:
    """Parse comma-separated levels and inclusive ranges such as ``1-4,9``."""
    levels: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_raw, stop_raw = token.split("-", maxsplit=1)
            start, stop = int(start_raw), int(stop_raw)
            if start > stop:
                raise argparse.ArgumentTypeError(f"descending range {token!r}")
            levels.extend(range(start, stop + 1))
        else:
            levels.append(int(token))
    if not levels:
        raise argparse.ArgumentTypeError("select at least one level")
    if len(levels) != len(set(levels)):
        raise argparse.ArgumentTypeError("levels must not repeat")
    unknown = [level for level in levels if level <= 0 or level not in OBSTACLE_LEVELS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown nonzero level(s): {unknown}")
    return levels


def _figure_grid(manifest: dict):
    levels = sorted(int(level) for level in manifest["levels"])
    columns = min(6, len(levels))
    rows = math.ceil(len(levels) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.35 * columns, 3.25 * rows),
        squeeze=False,
    )
    return levels, figure, axes


def _draw_obstacles(axis, level: int) -> None:
    for start, stop in OBSTACLE_LEVELS[level]:
        axis.plot(
            [start[0], stop[0]],
            [start[1], stop[1]],
            color="#22262b",
            linewidth=5.0,
            solid_capstyle="round",
            zorder=10,
        )


def _style_axis(axis, level: int, count: int) -> None:
    axis.set(
        xlim=(0.0, WORLD_SIZE),
        ylim=(WORLD_SIZE, 0.0),
        aspect="equal",
        xticks=[],
        yticks=[],
        title=f"L{level:02d} · {SKETCH_FAMILY_NAMES[level]} · n={count}",
    )
    axis.set_facecolor("#f5f5f3")
    for spine in axis.spines.values():
        spine.set_color("#b5b8bb")
        spine.set_linewidth(0.7)
    axis.title.set_fontsize(9)


def _finish_figure(
    figure,
    axes,
    level_count: int,
    *,
    title: str,
    legend: list,
    destination: str | Path,
) -> Path:
    for axis in list(axes.flat)[level_count:]:
        axis.axis("off")
    figure.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.978),
        ncol=len(legend),
        frameon=False,
        fontsize=9,
    )
    figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.95), h_pad=1.25, w_pad=0.8)
    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    return output


def plot_manifest(manifest: dict, destination: str | Path) -> Path:
    """Render all selected starts, goals, agents, and blocked direct paths."""
    levels, figure, axes = _figure_grid(manifest)

    for axis, level in zip(axes.flat, levels, strict=False):
        entries = level_entries(manifest, level)
        for entry in entries:
            start = np.asarray(entry["object_pose"][:2], dtype=np.float64)
            goal = np.asarray(entry["goal_pose"][:2], dtype=np.float64)
            agent = np.asarray(entry["agent_pos"], dtype=np.float64)
            axis.plot(
                [start[0], goal[0]],
                [start[1], goal[1]],
                color="#77808c",
                linewidth=0.65,
                alpha=0.20,
                zorder=1,
            )
            collision_alpha = float(np.median(entry["collision_alphas"]))
            collision = start + collision_alpha * (goal - start)
            axis.scatter(
                collision[0],
                collision[1],
                marker="x",
                s=11,
                linewidths=0.75,
                color="#d64541",
                alpha=0.65,
                zorder=3,
            )
            axis.scatter(
                agent[0],
                agent[1],
                marker="+",
                s=16,
                linewidths=0.75,
                color="#198754",
                alpha=0.72,
                zorder=3,
            )
            axis.scatter(
                start[0],
                start[1],
                marker="o",
                s=12,
                linewidths=0,
                color="#2878b5",
                alpha=0.76,
                zorder=4,
            )
            axis.scatter(
                goal[0],
                goal[1],
                marker="^",
                s=15,
                linewidths=0,
                color="#e68613",
                alpha=0.76,
                zorder=4,
            )

        _draw_obstacles(axis, level)
        _style_axis(axis, level, len(entries))

    legend = [
        Line2D(
            [], [], marker="+", linestyle="none", color="#198754", label="chain anchor"
        ),
        Line2D([], [], marker="o", linestyle="none", color="#2878b5", label="T start"),
        Line2D([], [], marker="^", linestyle="none", color="#e68613", label="goal"),
        Line2D(
            [],
            [],
            marker="x",
            linestyle="none",
            color="#d64541",
            label="blocked direct sweep",
        ),
        Line2D([], [], linewidth=1.0, color="#77808c", label="paired direct path"),
    ]
    return _finish_figure(
        figure,
        axes,
        len(levels),
        title="ChainGripper obstacle collection · level-specific initialization bank",
        legend=legend,
        destination=destination,
    )


def _add_silhouette(axis, shape: str, pose: list[float], color: str) -> None:
    geometry = object_polygon(shape, tuple(pose[:2]), float(pose[2]))
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    for polygon in polygons:
        axis.add_patch(
            PlotPolygon(
                np.asarray(polygon.exterior.coords),
                closed=True,
                facecolor=color,
                edgecolor=color,
                linewidth=0.55,
                alpha=0.10,
                zorder=2,
            )
        )


def plot_silhouette_manifest(manifest: dict, destination: str | Path) -> Path:
    """Render exact T silhouettes to audit arena and obstacle clearances."""
    levels, figure, axes = _figure_grid(manifest)
    arena_margin = float(manifest["criteria"]["min_arena_clearance"])
    obstacle_margin = float(manifest["criteria"]["min_obstacle_clearance"])
    shape = str(manifest["object_shape"])
    for axis, level in zip(axes.flat, levels, strict=False):
        entries = level_entries(manifest, level)
        for start, stop in OBSTACLE_LEVELS[level]:
            clearance = LineString([start, stop]).buffer(WALL_RADIUS + obstacle_margin)
            axis.add_patch(
                PlotPolygon(
                    np.asarray(clearance.exterior.coords),
                    closed=True,
                    facecolor="#6f6674",
                    edgecolor="none",
                    alpha=0.08,
                    zorder=1,
                )
            )
        for entry in entries:
            _add_silhouette(axis, shape, entry["object_pose"], "#2878b5")
            _add_silhouette(axis, shape, entry["goal_pose"], "#e68613")
        axis.add_patch(
            Rectangle(
                (arena_margin, arena_margin),
                WORLD_SIZE - 2.0 * arena_margin,
                WORLD_SIZE - 2.0 * arena_margin,
                fill=False,
                edgecolor="#b54c63",
                linewidth=0.9,
                linestyle=(0, (4, 3)),
                zorder=9,
            )
        )
        _draw_obstacles(axis, level)
        _style_axis(axis, level, len(entries))

    legend = [
        Patch(
            facecolor="#2878b5",
            edgecolor="#2878b5",
            alpha=0.20,
            label="T start silhouette",
        ),
        Patch(
            facecolor="#e68613",
            edgecolor="#e68613",
            alpha=0.20,
            label="goal silhouette",
        ),
        Line2D(
            [],
            [],
            color="#b54c63",
            linestyle=(0, (4, 3)),
            label=f"{arena_margin:g}-unit arena clearance boundary",
        ),
        Patch(
            facecolor="#6f6674",
            edgecolor="none",
            alpha=0.16,
            label=f"{obstacle_margin:g}-unit obstacle clearance envelope",
        ),
        Line2D([], [], linewidth=5.0, color="#22262b", label="physical obstacle"),
    ]
    return _finish_figure(
        figure,
        axes,
        len(levels),
        title="T start/goal silhouette clearance audit",
        legend=legend,
        destination=destination,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--output", type=Path, help="generate and write a manifest")
    source.add_argument(
        "--from-manifest",
        type=Path,
        help="render an existing manifest without resampling",
    )
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--silhouette-plot", type=Path)
    parser.add_argument("--levels", type=parse_levels, default=parse_levels("1-30"))
    parser.add_argument("--per-level", type=int, default=32)
    parser.add_argument("--seed-limit", type=int, default=10_000)
    parser.add_argument("--pool-multiplier", type=int, default=4)
    args = parser.parse_args(argv)
    if args.per_level < 1:
        parser.error("--per-level must be positive")
    if args.seed_limit < 1:
        parser.error("--seed-limit must be positive")
    if args.pool_multiplier < 1:
        parser.error("--pool-multiplier must be positive")

    if args.from_manifest is not None:
        manifest = load_manifest(args.from_manifest)
        print(f"loaded manifest from {args.from_manifest.resolve()}", flush=True)
    else:
        criteria = replace(
            DEFAULT_CRITERIA,
            seed_limit=args.seed_limit,
            pool_multiplier=args.pool_multiplier,
        )
        manifest = curate_manifest(
            levels=args.levels,
            count=args.per_level,
            criteria=criteria,
        )
        manifest_path = write_manifest(manifest, args.output)
        print(
            f"wrote {len(args.levels)} levels x {args.per_level} entries "
            f"to {manifest_path}",
            flush=True,
        )
    if args.plot is not None:
        plot_path = plot_manifest(manifest, args.plot)
        print(f"wrote visualization to {plot_path}", flush=True)
    if args.silhouette_plot is not None:
        silhouette_path = plot_silhouette_manifest(manifest, args.silhouette_plot)
        print(f"wrote silhouette visualization to {silhouette_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
