"""
State features for the market-making agent (Spooner et al. 2018, §4.3).

Market-state features (partially observable, computed at each decision epoch):

1. Market spread ``s``           - best_ask - best_bid, in ticks.
2. Mid-price move ``dmid``       - change in mid over the last interval, in ticks.
3. Book/queue imbalance          - (bid_vol - ask_vol) / (bid_vol + ask_vol) over top-k.
4. Signed volume                 - rolling sum of aggressor-signed trade size.
5. Volatility                    - std of recent mid moves (ticks).
6. Relative strength index (RSI) - standard RSI of the mid series.

Agent-state features (fully observable) are appended by the environment: normalised
inventory and the active quoting distances theta_a / theta_b. Keeping the market and
agent features separable matters for Phase 3, where the paper's linear combination of
tile codings approximates the agent-, market- and full-state independently.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from algo_mm.rl.book import ASK, BID, OrderBook, PRICE_SCALE
from algo_mm.rl.config import FeatureConfig

MARKET_FEATURE_NAMES: tuple[str, ...] = (
    "spread_ticks",
    "dmid_ticks",
    "book_imbalance",
    "signed_volume",
    "volatility",
    "rsi",
)


class FeatureExtractor:
    """Rolling market-state features updated once per decision epoch."""

    def __init__(self, config: FeatureConfig, tick_size: float) -> None:
        self.cfg = config
        self.tick = tick_size
        self._mid_moves: deque[float] = deque(maxlen=config.volatility_lookback)
        self._gains: deque[float] = deque(maxlen=config.rsi_lookback)
        self._losses: deque[float] = deque(maxlen=config.rsi_lookback)
        self._signed_vol: deque[float] = deque(maxlen=config.signed_volume_lookback)
        self._last_mid: float | None = None

    def reset(self) -> None:
        self._mid_moves.clear()
        self._gains.clear()
        self._losses.clear()
        self._signed_vol.clear()
        self._last_mid = None

    def _book_imbalance(self, book: OrderBook) -> float:
        bids, asks = book.depth(self.cfg.imbalance_levels)
        bid_vol = float(sum(sz for _, sz in bids))
        ask_vol = float(sum(sz for _, sz in asks))
        denom = bid_vol + ask_vol
        return (bid_vol - ask_vol) / denom if denom > 0 else 0.0

    def update(self, book: OrderBook, interval_signed_volume: float) -> np.ndarray:
        """
        Update rolling state with the latest book snapshot and interval trade flow,
        and return the market-feature vector (in :data:`MARKET_FEATURE_NAMES` order).

        ``interval_signed_volume`` is the aggressor-signed traded volume since the last
        decision epoch (buys positive, sells negative).
        """
        bb, ba = book.best_bid(), book.best_ask()
        mid = book.mid()
        spread_ticks = ((ba - bb) * PRICE_SCALE / self.tick) if (bb is not None and ba is not None) else 0.0

        if mid is None:
            dmid_ticks = 0.0
        else:
            mid_px = mid * PRICE_SCALE
            if self._last_mid is None:
                dmid_ticks = 0.0
            else:
                dmid_ticks = (mid_px - self._last_mid) / self.tick
            self._last_mid = mid_px

        self._mid_moves.append(dmid_ticks)
        self._gains.append(max(0.0, dmid_ticks))
        self._losses.append(max(0.0, -dmid_ticks))
        self._signed_vol.append(interval_signed_volume)

        volatility = float(np.std(self._mid_moves)) if len(self._mid_moves) > 1 else 0.0
        imbalance = self._book_imbalance(book)
        signed_volume = float(sum(self._signed_vol))
        rsi = self._rsi()

        return np.array(
            [spread_ticks, dmid_ticks, imbalance, signed_volume, volatility, rsi],
            dtype=np.float64,
        )

    def _rsi(self) -> float:
        avg_gain = float(np.mean(self._gains)) if self._gains else 0.0
        avg_loss = float(np.mean(self._losses)) if self._losses else 0.0
        if avg_loss == 0.0:
            return 100.0 if avg_gain > 0.0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)
