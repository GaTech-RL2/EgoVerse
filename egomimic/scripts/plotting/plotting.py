import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


class ColorsPalette:
    PACBLUE = [
        "#eff5f7",
        "#cfe1e5",
        "#b7d3d8",
        "#96bfc7",
        "#82b2bc",
        "#639fab",
        "#5a919c",
        "#467179",
        "#36575e",
        "#2a4348",
    ]
    WILLOWGREEN = [
        "#f4faf3",
        "#dcf0db",
        "#cbe9c9",
        "#b3e0b1",
        "#a5d9a1",
        "#8ed08a",
        "#81bd7e",
        "#659462",
        "#4e724c",
        "#3c573a",
    ]
    TIGERFLAME = [
        "#fdf0eb",
        "#f9cfc0",
        "#f6b8a2",
        "#f29877",
        "#ef845d",
        "#eb6534",
        "#d65c2f",
        "#a74825",
        "#81381d",
        "#632a16",
    ]
    LILAC = [
        "#f6f4f7",
        "#e4dee7",
        "#d7cedc",
        "#c5b7cc",
        "#baa9c2",
        "#a994b3",
        "#9a87a3",
        "#78697f",
        "#5d5162",
        "#473e4b",
    ]


def _set_monospace_style():
    mpl.rcParams["font.family"] = "monospace"
    mpl.rcParams["font.monospace"] = ["DejaVu Sans Mono"]


def _resolve_group_positions(group_x_spacing, num_groups):
    if group_x_spacing is None:
        if num_groups == 1:
            return [0.5]
        return [i / (num_groups - 1) for i in range(num_groups)]
    if len(group_x_spacing) != num_groups:
        raise ValueError("group_x_spacing must have the same length as groups")
    return list(group_x_spacing)


def _parse_groups(groups):
    if not groups:
        raise ValueError(
            "groups must be a non-empty dict of {group_name: (values, error_bars, colors)}"
        )
    group_names = list(groups.keys())
    parsed_groups = {}
    group_means = {}
    max_bars = 1
    for group_name, group_data in groups.items():
        if not isinstance(group_data, (list, tuple)) or len(group_data) < 3:
            raise ValueError("Each group must be (values, error_bars, colors)")
        values = list(group_data[0])
        error_bars = group_data[1]
        colors = group_data[2]
        if len(group_data) > 3:
            group_means[group_name] = group_data[3]
        parsed_groups[group_name] = (values, error_bars, colors)
        max_bars = max(max_bars, len(values))
    return group_names, parsed_groups, group_means, max_bars


def _resolve_bar_labels(bar_labels, max_bars):
    if bar_labels is None:
        bar_labels = [f"Bar {i + 1}" for i in range(max_bars)]
    if len(bar_labels) != max_bars:
        raise ValueError(
            "bar_labels must have the same length as the number of bars per group"
        )
    return bar_labels


def _compute_layout(group_positions, max_bars, group_width_scale=0.6):
    if len(group_positions) > 1:
        sorted_positions = sorted(group_positions)
        diffs = [b - a for a, b in zip(sorted_positions, sorted_positions[1:]) if b > a]
        min_spacing = min(diffs) if diffs else 1.0 / len(group_positions)
    else:
        min_spacing = 1.0
    base_group_width = min_spacing * group_width_scale
    bar_width = base_group_width / max_bars
    bar_gap = bar_width * 0.2
    group_width = bar_width * max_bars + bar_gap * (max_bars - 1)
    offsets = [
        (-group_width / 2 + bar_width / 2) + i * (bar_width + bar_gap)
        for i in range(max_bars)
    ]
    return bar_width, group_width, offsets


def _resolve_color(colors, value_index):
    if isinstance(colors, (list, tuple)):
        color = colors[value_index] if value_index < len(colors) else colors[-1]
    else:
        color = colors
    return mpl.colors.to_rgba(color, alpha=1.0)


def _resolve_yerr(error_bars, value_index):
    if error_bars is None:
        return None
    if isinstance(error_bars, (list, tuple)):
        if value_index < len(error_bars):
            return error_bars[value_index]
        return None
    return error_bars


def _draw_group_mean(ax, group_position, group_width, mean_value):
    base = mean_value if mean_value < 0 else 0
    height = -mean_value if mean_value < 0 else mean_value
    mean_width = group_width * 1.05
    ax.add_patch(
        mpl.patches.Rectangle(
            (group_position - mean_width / 2, base),
            mean_width,
            height,
            fill=False,
            edgecolor="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
            zorder=10,
        )
    )


def _wrap_label(label, width):
    label_str = str(label)
    if width is None or width <= 0:
        return label_str
    wrapped = textwrap.wrap(
        label_str, width=width, break_long_words=False, break_on_hyphens=False
    )
    if len(wrapped) == 1 and len(label_str) > width:
        wrapped = textwrap.wrap(
            label_str, width=width, break_long_words=True, break_on_hyphens=False
        )
    return "\n".join(wrapped) if wrapped else label_str


def plot_multi_line_chart(lines, x_label, y_label, title):
    """
    Plot multiple lines on a single chart.

    Args:
        lines (dict): {line_name: (x_array, y_array, color, sem)}
        x_label (str): Label for x-axis
        y_label (str): Label for y-axis
        title (str): Chart title
    """
    _set_monospace_style()

    fig, ax = plt.subplots(figsize=(8, 5))
    for line_name, (x_vals, y_vals, color, sem) in lines.items():
        ax.plot(x_vals, y_vals, label=line_name, color=color, marker="o", markersize=4)
        if sem is not None:
            ax.fill_between(
                x_vals,
                [y - s for y, s in zip(y_vals, sem)],
                [y + s for y, s in zip(y_vals, sem)],
                color=color,
                alpha=0.2,
                linewidth=0,
            )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return fig, ax


def plot_bar_chart(
    groups,
    title,
    group_x_spacing=None,
    bar_labels=None,
    legend_title="Bars",
    figsize=(10, 5),
    rotation=0,
    group_width_scale=0.6,
    group_mean=False,
):
    """
    groups: dict of {group_name: (values, error_bars, colors)}
    group_x_spacing: list of x positions for each group normalized between 0 and 1.  Default, even spacing
    group_means: dict of {group_name: mean_value}. plot a dashed transparent box over the group at this mean value
    If each group tuple has a 4th value, it is interpreted as that group's mean value unless group_mean is True.
    bar_labels: list of group names for each bar index (color), used in the legend.
    legend_title: title for the legend. Defaults to "Bars".
    title: str
    group_width_scale: fraction of the available group spacing used for bars.
    group_mean: when True, compute and plot mean of each group's bars.
    """
    _set_monospace_style()
    group_names, parsed_groups, group_means, max_bars = _parse_groups(groups)
    if group_mean:
        for group_name, (values, _error_bars, _colors) in parsed_groups.items():
            if group_name not in group_means:
                group_means[group_name] = sum(values) / len(values) if values else 0.0
    group_positions = _resolve_group_positions(group_x_spacing, len(group_names))
    bar_labels = _resolve_bar_labels(bar_labels, max_bars)
    bar_width, group_width, offsets = _compute_layout(
        group_positions, max_bars, group_width_scale=group_width_scale
    )
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axisbelow(True)

    for group_index, group_name in enumerate(group_names):
        values, error_bars, colors = parsed_groups[group_name]
        for value_index, value in enumerate(values):
            color = _resolve_color(colors, value_index)
            yerr = _resolve_yerr(error_bars, value_index)
            ax.bar(
                group_positions[group_index] + offsets[value_index],
                value,
                width=bar_width * 0.9,
                color=color,
                yerr=yerr,
                capsize=3,
                linewidth=0.0,
                edgecolor="black",
                alpha=1.0,
                zorder=3,
                label=bar_labels[value_index] if group_index == 0 else None,
            )

        if group_name in group_means:
            _draw_group_mean(
                ax, group_positions[group_index], group_width, group_means[group_name]
            )

    ax.set_xticks(group_positions)
    wrap_width = max(10, int(figsize[0] * 1.4))
    wrapped_group_names = [_wrap_label(name, wrap_width) for name in group_names]
    ax.set_xticklabels(wrapped_group_names, rotation=rotation, ha="center")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    if legend_title is not None:
        ax.legend(title=legend_title)
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
    ax.set_xlim(min(group_positions) - group_width, max(group_positions) + group_width)
    return fig, ax


def plot_specified_bars(
    df,
    task,
    bar_order,
    *,
    score_cols=None,
    normalization_col="Normalization",
    task_col="Task",
    human_hours_col="Human Data (hrs)",
    aggregate_col="Aggregate Score",
    bar_color=None,
    bar_colors=None,
    legend_title="Metric",
    figsize=(12, 5),
    include_remaining=True,
    rotation=0,
    group_width_scale=0.6,
):
    """
    Plot a single metric bar per group for a specified task.

    Args:
        df (pd.DataFrame): Preprocessed dataframe (e.g., after renaming labels).
        task (str): Task name to filter.
        bar_order (list[str]): Desired ordering for bar groups.
        score_cols (list[str], optional): Columns used to compute aggregate score.
        normalization_col (str): Column used to normalize aggregate score.
        task_col (str): Column name that contains task labels.
        human_hours_col (str): Column name that contains bar group labels.
        aggregate_col (str): Column name to store the aggregate score.
        bar_color (str, optional): Bar color. Defaults to ColorsPalette.PACBLUE[4].
        bar_colors (list[str] | dict[str, str], optional): Per-bar colors aligned with bar_order or keyed by label.
        legend_title (str): Legend title passed to plot_bar_chart.
        figsize (tuple): Figure size passed to plot_bar_chart.
        include_remaining (bool): Append any unseen groups after bar_order.
        group_width_scale (float): Fraction of available group spacing used for bars.
    """
    if score_cols is None:
        score_cols = [
            "In-domain (20 rollouts)",
            "New object (10 rollouts)",
            "New object + new scene (10 rollouts)",
        ]

    df = df.copy()
    for col in score_cols + [normalization_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[score_cols] = df[score_cols].fillna(0)
    df[normalization_col] = df[normalization_col].fillna(1.0).replace(0, 1.0)
    df[aggregate_col] = (
        df[score_cols].sum(axis=1) / df[normalization_col] / len(score_cols)
    )

    df_task = df[df[task_col] == task].copy()
    if include_remaining:
        remaining_order = [
            h for h in df_task[human_hours_col].tolist() if h not in bar_order
        ]
        bar_order = list(bar_order) + remaining_order
    else:
        bar_order = list(bar_order)

    if bar_color is None:
        bar_color = ColorsPalette.PACBLUE[4]

    groups = {}
    for idx, label in enumerate(bar_order):
        row = df_task[df_task[human_hours_col] == label]
        value = float(row[aggregate_col].iloc[0]) if not row.empty else 0.0
        if isinstance(bar_colors, dict):
            color = bar_colors.get(label, bar_color)
        elif isinstance(bar_colors, (list, tuple)):
            color = bar_colors[idx] if idx < len(bar_colors) else bar_color
        else:
            color = bar_color
        groups[label] = ([value], None, [color])

    fig, ax = plot_bar_chart(
        groups,
        title=f"{task} Aggregate Score",
        bar_labels=["Aggregate Score"],
        legend_title=legend_title,
        figsize=figsize,
        rotation=rotation,
        group_width_scale=group_width_scale,
    )
    return fig, ax


def _format_flagship_value(value, decimals=1, as_percent=True):
    if pd.isna(value):
        return "--"
    numeric_value = float(value)
    if as_percent:
        return f"{numeric_value * 100:.{decimals}f}\\%"
    return f"{numeric_value:.{decimals}f}"


def print_flagship_latex_table(
    df_flagship,
    *,
    caption="Flagship benchmark results.",
    label="tab:flagship",
    teams=None,
    tasks=None,
    include_ood=True,
    metrics=None,
    decimals=1,
    as_percent=True,
    bold_best=True,
    team_col="Team",
    task_col="Task Name",
    print_table=True,
):
    """
    Build and print a paper-ready LaTeX table for df_flagship using booktabs rules.

    Args:
        df_flagship (pd.DataFrame): Flagship dataframe.
        caption (str | None): LaTeX table caption.
        label (str | None): LaTeX table label.
        teams (list[str] | None): Optional ordered team subset.
        tasks (list[str] | None): Optional ordered task subset.
        include_ood (bool): Compute/include OOD metrics when available.
        metrics (list[str] | None): Explicit metric columns to include.
        decimals (int): Decimal places for metric formatting.
        as_percent (bool): Format numeric values as percentages.
        bold_best (bool): Bold the better value in Robot-Only vs Co-train pairs.
        team_col (str): Team column name.
        task_col (str): Task column name.
        print_table (bool): Print LaTeX output to stdout.
    """
    required = [team_col, task_col]
    missing_required = [col for col in required if col not in df_flagship.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    df = df_flagship.copy()

    if include_ood:
        robot_ood = "Robot-Only (OOD)"
        cotrain_ood = "Co-train (OOD)"
        robot_no = "Robot-Only (New Object)"
        robot_nos = "Robot-Only (New Object + Scene)"
        cotrain_no = "Co-train (New Object)"
        cotrain_nos = "Co-train (New Object + Scene)"
        if (
            robot_ood not in df.columns
            and robot_no in df.columns
            and robot_nos in df.columns
        ):
            df[robot_ood] = (
                pd.to_numeric(df[robot_no], errors="coerce")
                + pd.to_numeric(df[robot_nos], errors="coerce")
            ) / 2.0
        if (
            cotrain_ood not in df.columns
            and cotrain_no in df.columns
            and cotrain_nos in df.columns
        ):
            df[cotrain_ood] = (
                pd.to_numeric(df[cotrain_no], errors="coerce")
                + pd.to_numeric(df[cotrain_nos], errors="coerce")
            ) / 2.0

    if metrics is None:
        metrics = ["Robot-Only (ID)", "Co-train (ID)"]
        if (
            include_ood
            and "Robot-Only (OOD)" in df.columns
            and "Co-train (OOD)" in df.columns
        ):
            metrics.extend(["Robot-Only (OOD)", "Co-train (OOD)"])

    missing_metrics = [col for col in metrics if col not in df.columns]
    if missing_metrics:
        raise ValueError(f"Missing metric columns: {missing_metrics}")

    for col in metrics:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if teams is not None:
        teams = list(teams)
        df = df[df[team_col].isin(teams)].copy()
        df[team_col] = pd.Categorical(df[team_col], categories=teams, ordered=True)
    if tasks is not None:
        tasks = list(tasks)
        df = df[df[task_col].isin(tasks)].copy()
        df[task_col] = pd.Categorical(df[task_col], categories=tasks, ordered=True)

    df = df[[team_col, task_col] + metrics].sort_values(
        [team_col, task_col], kind="stable"
    )

    formatted = df[[team_col, task_col]].copy()
    formatted[task_col] = (
        formatted[task_col].astype(str).str.replace("_", " ", regex=False)
    )
    for col in metrics:
        formatted[col] = df[col].apply(
            _format_flagship_value, decimals=decimals, as_percent=as_percent
        )

    if bold_best:
        metric_pairs = []
        if "Robot-Only (ID)" in metrics and "Co-train (ID)" in metrics:
            metric_pairs.append(("Robot-Only (ID)", "Co-train (ID)"))
        if "Robot-Only (OOD)" in metrics and "Co-train (OOD)" in metrics:
            metric_pairs.append(("Robot-Only (OOD)", "Co-train (OOD)"))
        for left_col, right_col in metric_pairs:
            for row_idx in df.index:
                left_val = df.at[row_idx, left_col]
                right_val = df.at[row_idx, right_col]
                if pd.isna(left_val) and pd.isna(right_val):
                    continue
                if pd.isna(right_val) or (
                    not pd.isna(left_val) and left_val > right_val
                ):
                    if formatted.at[row_idx, left_col] != "--":
                        formatted.at[row_idx, left_col] = (
                            f"\\textbf{{{formatted.at[row_idx, left_col]}}}"
                        )
                elif pd.isna(left_val) or right_val > left_val:
                    if formatted.at[row_idx, right_col] != "--":
                        formatted.at[row_idx, right_col] = (
                            f"\\textbf{{{formatted.at[row_idx, right_col]}}}"
                        )
                else:
                    if formatted.at[row_idx, left_col] != "--":
                        formatted.at[row_idx, left_col] = (
                            f"\\textbf{{{formatted.at[row_idx, left_col]}}}"
                        )
                    if formatted.at[row_idx, right_col] != "--":
                        formatted.at[row_idx, right_col] = (
                            f"\\textbf{{{formatted.at[row_idx, right_col]}}}"
                        )

    to_latex_kwargs = {
        "index": False,
        "escape": False,
        "column_format": "ll" + ("c" * len(metrics)),
        "na_rep": "--",
    }
    if caption is not None:
        to_latex_kwargs["caption"] = caption
    if label is not None:
        to_latex_kwargs["label"] = label

    latex_table = formatted.to_latex(**to_latex_kwargs)

    if print_table:
        print(latex_table)
    return latex_table
