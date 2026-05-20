from __future__ import annotations

import numpy as np

from .base import MarketMaker


class NaiveMarketMaker(MarketMaker):
    """Fixed symmetric spread around mid-price; no inventory adjustment."""

    def __init__(
        self,
        half_spread: float,
        lambda_buy: float = 100.0,
        lambda_sell: float = 100.0,
    ) -> None:
        self.half_spread = half_spread
        self.lambda_buy = lambda_buy
        self.lambda_sell = lambda_sell

    def compute_quotes(self, mid: float, inventory: int, t: float) -> tuple[float, float]:
        return mid - self.half_spread, mid + self.half_spread

    def fill_probs(self, mid: float, bid: float, ask: float, dt: float) -> tuple[float, float]:
        prob_bid_fill = 1 - np.exp(-self.lambda_sell * dt)
        prob_ask_fill = 1 - np.exp(-self.lambda_buy * dt)
        return prob_bid_fill, prob_ask_fill
