import matplotlib.pyplot as plt
import numpy as np

from ..simulation.engine import SimResult


def plot_simulation(result: SimResult, title: str = "Market Making Simulation") -> plt.Figure:
    """Plot price/quotes, PnL, inventory, and drawdown for a single simulation run."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    running_max = np.maximum.accumulate(result.pnl)
    drawdown = running_max - result.pnl

    ax = axes[0, 0]
    ax.plot(result.time, result.midprice, label="Mid-price", color="gray", alpha=0.5, lw=1)
    ax.plot(result.time, result.bid_quotes, label="Bid", color="green", linestyle="--", alpha=0.8)
    ax.plot(result.time, result.ask_quotes, label="Ask", color="red", linestyle="--", alpha=0.8)
    ax.set_title(f"{title} — Price and Quotes")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(result.time, result.pnl, color="gold", lw=2)
    ax.set_title("PnL")
    ax.set_xlabel("Time")
    ax.set_ylabel("PnL")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(result.time, result.inventory, color="purple")
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
    ax.set_title("Inventory")
    ax.set_xlabel("Time")
    ax.set_ylabel("Units")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.fill_between(result.time, 0, drawdown, alpha=0.5, color="red")
    ax.plot(result.time, drawdown, color="darkred", lw=1.5)
    ax.set_title(f"Drawdown (Max: {np.max(drawdown):.2f})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
