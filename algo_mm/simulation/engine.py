from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..models.base import MarketMaker


@dataclass
class SimResult:
    time: np.ndarray
    midprice: np.ndarray
    bid_quotes: np.ndarray
    ask_quotes: np.ndarray
    inventory: np.ndarray
    pnl: np.ndarray


def run_simulation(
    model: MarketMaker,
    sigma: float,
    S0: float = 100.0,
    T: float = 1.0,
    dt: float = 0.005,
    q0: int = 0,
    seed: Optional[int] = None,
) -> SimResult:
    """Run a single market making simulation.

    Args:
        model:  MarketMaker instance that provides quoting and fill-probability logic.
        sigma:  Volatility of the underlying asset (used for the Brownian price process).
        S0:     Initial mid-price.
        T:      Trading horizon.
        dt:     Time-step size.
        q0:     Initial inventory.
        seed:   Optional RNG seed for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)

    n_steps = int(T / dt)
    time = np.linspace(0, T, n_steps)

    midprice = np.zeros(n_steps)
    bid_quotes = np.zeros(n_steps)
    ask_quotes = np.zeros(n_steps)
    inventory = np.zeros(n_steps)
    pnl = np.zeros(n_steps)

    s = S0
    q = q0
    cash = 0.0

    for i in range(n_steps):
        t = time[i]
        s += sigma * np.sqrt(dt) * np.random.normal()
        midprice[i] = s

        bid, ask = model.compute_quotes(s, q, t)
        bid_quotes[i] = bid
        ask_quotes[i] = ask

        prob_bid_fill, prob_ask_fill = model.fill_probs(s, bid, ask, dt)

        if np.random.random() < prob_bid_fill:
            q += 1
            cash -= bid

        if np.random.random() < prob_ask_fill:
            q -= 1
            cash += ask

        inventory[i] = q
        pnl[i] = cash + q * s

    return SimResult(
        time=time,
        midprice=midprice,
        bid_quotes=bid_quotes,
        ask_quotes=ask_quotes,
        inventory=inventory,
        pnl=pnl,
    )
