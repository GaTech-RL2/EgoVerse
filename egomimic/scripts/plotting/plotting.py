import matplotlib.pyplot as plt
import matplotlib as mpl


class ColorsPalette:
    PACBLUE = ["#eff5f7", "#cfe1e5", "#b7d3d8", "#96bfc7", "#82b2bc", "#639fab", "#5a919c", "#467179", "#36575e", "#2a4348"]
    WILLOWGREEN = ["#f4faf3", "#dcf0db", "#cbe9c9", "#b3e0b1", "#a5d9a1", "#8ed08a", "#81bd7e", "#659462", "#4e724c", "#3c573a"]
    TIGERFLAME = ["#fdf0eb", "#f9cfc0", "#f6b8a2", "#f29877", "#ef845d", "#eb6534", "#d65c2f", "#a74825", "#81381d", "#632a16"]
    LILAC = ["#f6f4f7", "#e4dee7", "#d7cedc", "#c5b7cc", "#baa9c2", "#a994b3", "#9a87a3", "#78697f", "#5d5162", "#473e4b"]


def plot_multi_line_chart(lines, x_label, y_label, title):
    """
    Plot multiple lines on a single chart.

    Args:
        lines (dict): {line_name: (x_array, y_array, color, sem)}
        x_label (str): Label for x-axis
        y_label (str): Label for y-axis
        title (str): Chart title
    """
    mpl.rcParams["font.family"] = "monospace"
    mpl.rcParams["font.monospace"] = [
        "DejaVu Sans Mono"
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    for line_name, (x_vals, y_vals, color, sem) in lines.items():
        ax.plot(x_vals, y_vals, label=line_name, color=color, marker="o", markersize=4)
        if sem is not None:
            ax.fill_between(x_vals, [y - s for y, s in zip(y_vals, sem)], [y + s for y, s in zip(y_vals, sem)], color=color, alpha=0.2, linewidth=0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return fig, ax
