"""Plots for the lab, in one shared style so all four datasets look alike."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from lab.experiment import summarize

# Fixed colors: train is always blue, validation always orange (a colorblind-
# safe pair). Different markers too, so the plot also works in black & white.
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
    """Axis label like 50, 2k, 1M instead of 5x10^1, 2x10^3, 10^6."""
    for limit, suffix in [(1e6, "M"), (1e3, "k")]:
        if value >= limit:
            return f"{value / limit:g}{suffix}"
    return f"{value:g}"


def plot_learning_curve(curve, metric_name, title, baseline=None, log_x=False, log_y=False, ylim=None, path=None):
    """Plot training and validation score against the training-set size.

    One panel per model kind (e.g. linear | quadratic), sharing the y-axis so
    the models are directly comparable. The line is the MEDIAN over the
    repeated samples and the shaded band spans the 25%-75% quantiles. We use
    these instead of mean +/- std because a few unlucky small samples have
    huge losses that would dominate the mean (see lab.experiment.summarize),
    and because mean - std can be negative, which breaks a log axis.

    curve       DataFrame from lab.experiment.run_learning_curve
    metric_name y-axis label, e.g. "MSE" or "log-loss"
    baseline    optional validation score of the trivial baseline model
    log_x/log_y logarithmic axes (log_x for the big datasets, log_y when a
                curve spans several orders of magnitude)
    ylim        optional (bottom, top) for the y-axis, e.g. to cut off a
                training loss of exactly 0 on a log axis
    path        if given, the figure is saved there (folders are created)
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
            # Ticks at 1, 2, 5, 10, 20, 50, ... labelled compactly (1k, 2k, 1M).
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
