import matplotlib.pyplot as plt
import numpy as np

from ..metrics.performance import Metrics


def plot_distributions(metrics_list: list[Metrics]) -> plt.Figure:
    """Plot Monte Carlo distributions of Sharpe, Sortino, Max DD, Calmar, and final PnL."""
    data = [
        ([m.sharpe for m in metrics_list], "Sharpe Ratio", "blue"),
        ([m.sortino for m in metrics_list], "Sortino Ratio", "green"),
        ([m.max_drawdown for m in metrics_list], "Max Drawdown", "red"),
        ([m.calmar for m in metrics_list], "Calmar Ratio", "purple"),
        ([m.final_pnl for m in metrics_list], "Final PnL", "orange"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    for i, (vals, label, color) in enumerate(data):
        ax = axes.flatten()[i]
        ax.hist(vals, bins=40, color=color, alpha=0.6, edgecolor="black")
        median = np.median(vals)
        ax.axvline(median, color="black", linestyle="--", label=f"Median: {median:.2f}")
        ax.set_title(f"Distribution of {label}")
        ax.legend()

    fig.delaxes(axes[2, 1])
    plt.tight_layout()
    return fig
