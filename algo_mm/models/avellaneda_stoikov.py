from __future__ import annotations

import numpy as np

from .base import MarketMaker


class AvellanedaStoikovMarketMaker(MarketMaker):
    """Optimal market making via the Avellaneda-Stoikov (2008) framework.

    Quotes are derived from the reservation price (inventory-adjusted fair value)
    and an optimal spread that balances fill probability against inventory risk.

    Reference: Avellaneda & Stoikov (2008), "High-frequency trading in a limit
    order book", Quantitative Finance 8(3), 217-224.
    """

    def __init__(
        self,
        sigma: float,
        gamma: float,
        kappa: float,
        A: float,
        T: float = 1.0,
    ) -> None:
        self.sigma = sigma   # asset volatility
        self.gamma = gamma   # risk aversion
        self.kappa = kappa   # order book depth (fill-probability decay)
        self.A = A           # base order arrival intensity
        self.T = T           # trading horizon

    def compute_quotes(self, mid: float, inventory: int, t: float) -> tuple[float, float]:
        time_remaining = self.T - t
        reservation = mid - inventory * self.gamma * self.sigma**2 * time_remaining
        spread = (
            (2 / self.gamma) * np.log(1 + self.gamma / self.kappa)
            + self.gamma * self.sigma**2 * time_remaining
        )
        return reservation - spread / 2, reservation + spread / 2

    def fill_probs(self, mid: float, bid: float, ask: float, dt: float) -> tuple[float, float]:
        lambda_bid = self.A * np.exp(-self.kappa * (mid - bid))
        lambda_ask = self.A * np.exp(-self.kappa * (ask - mid))
        prob_bid_fill = 1 - np.exp(-lambda_bid * dt)
        prob_ask_fill = 1 - np.exp(-lambda_ask * dt)
        return prob_bid_fill, prob_ask_fill
