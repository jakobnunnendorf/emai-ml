"""Learning-curve plots (same style for all datasets)."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from lab.experiment import summarize

TRAIN_STYLE = {"color": "#2a78d6", "marker": "o", "label": "training"}
VAL_STYLE = {"color": "#eb6834", "marker": "s", "label": "validation"}
BASELINE_STYLE = {"color": "#888888", "linestyle": "--", "linewidth": 1.2}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def _compact_number(value, _position=None):
    """50, 2k, 1M"""
    for limit, suffix in [(1e6, "M"), (1e3, "k")]:
        if value >= limit:
            return f"{value / limit:g}{suffix}"
    return f"{value:g}"


def plot_learning_curve(curve, metric_name, title, baseline=None, log_x=False, log_y=False, ylim=None, path=None):
    """Training and validation score vs. training-set size, one panel per model.

    Line = median over the repeats, band = 25-75% quantiles. `baseline` is drawn
    as a dashed line; the figure is saved to `path` if given.
    """
    stats = summarize(curve)
    kinds = list(curve["model"].unique())

    fig, axes = plt.subplots(1, len(kinds), figsize=(5 * len(kinds), 3.8), sharey=True, squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        s = stats[stats["model"] == kind]
        for column, style in [("train_score", TRAIN_STYLE), ("val_score", VAL_STYLE)]:
            ax.plot(s["size"], s[(column, "median")], markersize=4, linewidth=2, **style)
            ax.fill_between(s["size"], s[(column, "q25")], s[(column, "q75")],
                            color=style["color"], alpha=0.15, linewidth=0)
        if baseline is not None:
            ax.axhline(baseline, label="baseline (val)", **BASELINE_STYLE)

        ax.set_title(kind)
        ax.set_xlabel("training set size")
        if log_x:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
            ax.xaxis.set_major_formatter(FuncFormatter(_compact_number))
            ax.xaxis.set_minor_formatter(NullFormatter())
        if log_y:
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)

    axes[0][0].set_ylabel(f"{metric_name} (median, 25-75% band)")
    axes[0][0].legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()

    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig
