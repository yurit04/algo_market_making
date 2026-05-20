from __future__ import annotations

from abc import ABC, abstractmethod


class MarketMaker(ABC):
    @abstractmethod
    def compute_quotes(self, mid: float, inventory: int, t: float) -> tuple[float, float]:
        """Return (bid, ask) for the given market state."""
        ...

    @abstractmethod
    def fill_probs(self, mid: float, bid: float, ask: float, dt: float) -> tuple[float, float]:
        """Return (prob_bid_fill, prob_ask_fill).

        prob_bid_fill: probability a sell order hits our bid  → we buy  (q += 1)
        prob_ask_fill: probability a buy  order hits our ask  → we sell (q -= 1)
        """
        ...
