# plotting.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def _fig_to_rgb(fig) -> np.ndarray:
    canvas = FigureCanvas(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb

def _draw_grid(ax, x_min, x_max, y_min, y_max, grid_size=10):
    x_grid = np.linspace(x_min, x_max, grid_size + 1)
    y_grid = np.linspace(y_min, y_max, grid_size + 1)
    for xi, xv in enumerate(x_grid[:-1]):
        for yi, yv in enumerate(y_grid[:-1]):
            cell = yi * grid_size + xi
            xc = 0.5 * (xv + x_grid[xi + 1]); yc = 0.5 * (yv + y_grid[yi + 1])
            ax.text(xc, yc, f"{cell}", color="black", fontsize=8, ha="center", va="center", alpha=0.65)
    for xv in x_grid:
        ax.axvline(x=xv, color="gray", linestyle="--", linewidth=0.5, alpha=0.65)
    for yv in y_grid:
        ax.axhline(y=yv, color="gray", linestyle="--", linewidth=0.5, alpha=0.65)

def plot_many(Z_2d: np.ndarray, y: np.ndarray, names, title: str, grid_size: int = 10) -> np.ndarray:
    """
    Multi-class scatter with grid overlay.
      - Z_2d: (N, 2)
      - y: (N,) int labels
      - names: list[str], label -> name
    Returns RGB image (H, W, 3)
    """
    if Z_2d.ndim != 2 or Z_2d.shape[1] != 2:
        raise ValueError("plot_many: Z_2d must be (N,2).")

    x_min, x_max = Z_2d[:, 0].min(), Z_2d[:, 0].max()
    y_min, y_max = Z_2d[:, 1].min(), Z_2d[:, 1].max()
    dx, dy = (x_max - x_min) * 0.05, (y_max - y_min) * 0.05
    x_min -= dx; x_max += dx; y_min -= dy; y_max += dy

    fig, ax = plt.subplots(figsize=(10, 10))
    for idx, name in enumerate(names):
        pts = Z_2d[y == idx]
        if pts.size == 0:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], label=name, alpha=0.7, s=16)  # default color cycle

    _draw_grid(ax, x_min, x_max, y_min, y_max, grid_size=grid_size)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", frameon=True)

    return _fig_to_rgb(fig)
