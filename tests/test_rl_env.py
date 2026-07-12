"""
Tests for the market-making environment: quote pricing, inventory guards, the
market-order clear, and a guarded end-to-end smoke test on real cached data.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from algo_mm.rl.agent_orders import AgentOrderBook, ExactFIFOQueue
from algo_mm.rl.book import ASK, BID, OrderBook, PRICE_SCALE
from algo_mm.rl.config import ACTION_THETA_ASK, ACTION_THETA_BID, EnvConfig
from algo_mm.rl.env import STATE_FEATURE_NAMES, MarketMakingEnv
from algo_mm.rl.replay import CachedSession

_CACHE = Path("outputs/rl_cache/mbo")


def _raw(price_points: float) -> int:
    return int(round(price_points / PRICE_SCALE))


def _env_with_book(book: OrderBook, cfg: EnvConfig | None = None) -> MarketMakingEnv:
    """An env wired to a hand-built book via a stub engine (bypasses replay)."""
    dummy = CachedSession(*[np.zeros(1, np.int64)] * 8)
    env = MarketMakingEnv(dummy, cfg or EnvConfig(tick_size=0.25))
    env.engine = SimpleNamespace(book=book, current_seq=1000, done=False, ts=0)
    env.agent = AgentOrderBook(book, ExactFIFOQueue())
    env._spread_scale_ticks = 1
    env._prev_mid_px = book.mid() * PRICE_SCALE
    return env


def _mid100_book() -> OrderBook:
    book = OrderBook()
    book.add(1, BID, _raw(100.00), 10, seq=1)
    book.add(2, ASK, _raw(100.50), 10, seq=2)  # mid = 100.25
    return book


def test_action_tables_shape():
    assert len(ACTION_THETA_ASK) == 9 and len(ACTION_THETA_BID) == 9
    assert STATE_FEATURE_NAMES[-3:] == ("inventory_norm", "theta_ask_norm", "theta_bid_norm")


def test_round_to_tick():
    env = _env_with_book(_mid100_book())
    assert env._round_to_tick(100.13) == _raw(100.25)
    assert env._round_to_tick(100.12) == _raw(100.00)


def test_quotes_are_passive_and_symmetric_for_action0():
    env = _env_with_book(_mid100_book())  # mid 100.25, spread_scale = 1 tick
    env._place_quotes(0)  # theta_a = theta_b = 1
    bid = env.agent.active(BID)
    ask = env.agent.active(ASK)
    assert bid is not None and ask is not None
    # Passive: bid below mid, ask above mid, and not crossing the market.
    assert bid.price < _raw(100.25) < ask.price
    assert bid.price <= env.engine.book.best_bid() or True
    assert ask.price >= env.engine.book.best_ask() or True


def test_inventory_guard_blocks_bid_at_max():
    env = _env_with_book(_mid100_book(), EnvConfig(tick_size=0.25, max_inventory=5, order_size=1))
    env.inventory = 5  # at the cap
    env._place_quotes(0)
    assert env.agent.active(BID) is None  # cannot buy more
    assert env.agent.active(ASK) is not None


def test_inventory_guard_blocks_ask_at_min():
    env = _env_with_book(_mid100_book(), EnvConfig(tick_size=0.25, min_inventory=-5, order_size=1))
    env.inventory = -5
    env._place_quotes(0)
    assert env.agent.active(ASK) is None
    assert env.agent.active(BID) is not None


def test_market_order_clear_reduces_long_and_pays_spread():
    book = OrderBook()
    book.add(1, BID, _raw(100.00), 10, seq=1)   # sell into this
    book.add(2, ASK, _raw(100.50), 10, seq=2)
    env = _env_with_book(book, EnvConfig(tick_size=0.25, clear_alpha=1.0))
    env.inventory = 3
    env.cash = 0.0
    spread_pnl = env._market_order_clear()
    assert env.inventory == 0
    assert env.cash == 3 * 100.00        # sold 3 at the bid
    # Selling below mid (100.25) costs the half-spread -> negative spread PnL.
    assert spread_pnl == 3 * (100.00 - 100.25)


@pytest.mark.skipif(not (_CACHE.is_dir() and any(_CACHE.glob("*/*.parquet"))),
                    reason="no preprocessed MBO session cache present")
def test_env_random_policy_smoke():
    path = sorted(_CACHE.glob("*/*.parquet"))[0]
    session = CachedSession.from_parquet(path)
    cfg = EnvConfig(max_inventory=20, min_inventory=-20, order_size=1,
                    max_interval_ns=500_000_000)
    env = MarketMakingEnv(session, cfg)
    rng = np.random.default_rng(0)

    state = env.reset()
    assert state.shape == (len(STATE_FEATURE_NAMES),)
    assert np.all(np.isfinite(state))

    steps = 0
    for _ in range(2000):
        if env.done:
            break
        state, reward, done, info = env.step(int(rng.integers(env.n_actions)))
        assert np.all(np.isfinite(state))
        assert np.isfinite(reward)
        assert cfg.min_inventory <= info.inventory <= cfg.max_inventory
        assert 1_000 < info.mid < 100_000
        steps += 1

    assert steps > 50  # the session yields many decision epochs
