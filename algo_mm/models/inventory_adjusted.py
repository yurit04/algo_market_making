from __future__ import annotations

import numpy as np

from .base import MarketMaker


class InventoryAdjustedMarketMaker(MarketMaker):
    """Fixed base spread with linear inventory skew.

    Quotes shift up when long (to encourage selling) and down when short
    (to encourage buying), keeping the spread width constant.
    """

    def __init__(
        self,
        base_spread: float,
        skew: float,
        q_target: int = 0,
        lambda_buy: float = 100.0,
        lambda_sell: float = 100.0,
    ) -> None:
        self.base_spread = base_spread
        self.skew = skew          # quote shift per unit of inventory deviation from q_target
        self.q_target = q_target
        self.lambda_buy = lambda_buy
        self.lambda_sell = lambda_sell

    def compute_quotes(self, mid: float, inventory: int, t: float) -> tuple[float, float]:
        adj = self.skew * (inventory - self.q_target)
        bid = mid - self.base_spread / 2 + adj
        ask = mid + self.base_spread / 2 + adj
        return bid, ask

    def fill_probs(self, mid: float, bid: float, ask: float, dt: float) -> tuple[float, float]:
        prob_bid_fill = 1 - np.exp(-self.lambda_sell * dt)
        prob_ask_fill = 1 - np.exp(-self.lambda_buy * dt)
        return prob_bid_fill, prob_ask_fill
