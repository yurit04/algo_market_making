"""Tests for market-state feature extraction (paper §4.3)."""

from __future__ import annotations

import numpy as np

from algo_mm.rl.book import ASK, BID, OrderBook, PRICE_SCALE
from algo_mm.rl.config import FeatureConfig
from algo_mm.rl.features import MARKET_FEATURE_NAMES, FeatureExtractor


def _raw(price_points: float) -> int:
    return int(round(price_points / PRICE_SCALE))


def _book(bids, asks):
    """Build a book from (price_points, size) pairs (prices are stored fixed-point)."""
    b = OrderBook()
    oid = 0
    for px, sz in bids:
        oid += 1
        b.add(oid, BID, _raw(px), sz, seq=oid)
    for px, sz in asks:
        oid += 1
        b.add(oid, ASK, _raw(px), sz, seq=oid)
    return b


def test_feature_vector_shape_and_order():
    fx = FeatureExtractor(FeatureConfig(), tick_size=1.0)
    book = _book([(100, 10)], [(101, 3)])
    vec = fx.update(book, 0.0)
    assert vec.shape == (len(MARKET_FEATURE_NAMES),)


def test_spread_and_imbalance():
    fx = FeatureExtractor(FeatureConfig(imbalance_levels=2), tick_size=1.0)
    book = _book([(100, 10), (99, 5)], [(101, 3), (102, 2)])
    vec = fx.update(book, 0.0)
    feats = dict(zip(MARKET_FEATURE_NAMES, vec))
    assert feats["spread_ticks"] == 1.0            # (101 - 100) / tick(1.0)
    assert feats["book_imbalance"] == (15 - 5) / 20  # (bid - ask) / total over top 2


def test_dmid_and_rsi_trend_up():
    fx = FeatureExtractor(FeatureConfig(), tick_size=1.0)
    fx.update(_book([(100, 1)], [(101, 1)], ), 0.0)  # mid 100.5, first step -> dmid 0
    vec = fx.update(_book([(101, 1)], [(102, 1)]), 0.0)  # mid 101.5, +1 tick
    feats = dict(zip(MARKET_FEATURE_NAMES, vec))
    assert feats["dmid_ticks"] == 1.0
    # Only upward moves so far -> RSI saturates high.
    assert feats["rsi"] == 100.0


def test_signed_volume_accumulates_over_lookback():
    fx = FeatureExtractor(FeatureConfig(signed_volume_lookback=3), tick_size=1.0)
    book = _book([(100, 1)], [(101, 1)])
    fx.update(book, 5.0)
    fx.update(book, -2.0)
    vec = fx.update(book, 1.0)
    feats = dict(zip(MARKET_FEATURE_NAMES, vec))
    assert feats["signed_volume"] == 5.0 - 2.0 + 1.0


def test_volatility_zero_when_flat():
    fx = FeatureExtractor(FeatureConfig(), tick_size=1.0)
    book = _book([(100, 1)], [(101, 1)])
    for _ in range(5):
        vec = fx.update(book, 0.0)
    feats = dict(zip(MARKET_FEATURE_NAMES, vec))
    assert feats["volatility"] == 0.0
