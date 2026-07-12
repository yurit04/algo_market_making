"""
Market-making environment (Spooner et al. 2018, §4).

An episode is one preprocessed trading session. At each decision epoch the agent picks
one of ten actions (Table 1): nine (ask, bid) quoting-distance pairs, or a market order
that clears inventory. Quotes are placed relative to the mid at distances that are
integer multiples of a spread scale factor (a moving average of the market half-spread,
rounded to a whole number of ticks), per Eqs. 1-2:

    p_ask = mid + theta_a * spread_scale
    p_bid = mid - theta_b * spread_scale

The environment then lets the historical order flow run for an interval, fills the
agent's resting quotes via the exact-FIFO (or uniform-cancel) queue model, and returns
the paper's reward (Eqs. 4-6). Inventory is bounded; near a bound only inventory-
reducing quotes are posted, and action 9 force-clears with a market order.

The environment holds no learning logic — it is a plain ``reset``/``step`` interface so
the same object is driven identically from the training CLI and from a notebook.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from algo_mm.rl.agent_orders import AgentOrderBook, ExactFIFOQueue, QueueModel
from algo_mm.rl.book import ASK, BID, PRICE_SCALE
from algo_mm.rl.config import (
    ACTION_THETA_ASK,
    ACTION_THETA_BID,
    CLEAR_ACTION_ID,
    N_ACTIONS,
    EnvConfig,
)
from algo_mm.rl.features import MARKET_FEATURE_NAMES, FeatureExtractor
from algo_mm.rl.replay import CachedSession, MarketReplay
from algo_mm.rl.rewards import compute_reward

AGENT_FEATURE_NAMES: tuple[str, ...] = ("inventory_norm", "theta_ask_norm", "theta_bid_norm")
STATE_FEATURE_NAMES: tuple[str, ...] = MARKET_FEATURE_NAMES + AGENT_FEATURE_NAMES


@dataclass
class StepInfo:
    """Per-step diagnostics (not part of the RL signal, used for evaluation/plots)."""

    reward: float
    spread_pnl: float
    inventory_pnl: float
    inventory: int
    mid: float
    equity: float
    n_fills: int
    action: int
    theta_ask: int
    theta_bid: int
    spread_scale_ticks: int


class MarketMakingEnv:
    """Event-driven market-making environment over one cached MBO session."""

    n_actions: int = N_ACTIONS

    def __init__(
        self,
        session: CachedSession,
        config: EnvConfig | None = None,
        queue_model: QueueModel | None = None,
    ) -> None:
        self.session = session
        self.cfg = config or EnvConfig()
        self._queue_factory = queue_model
        self.tick = self.cfg.tick_size
        self.tick_raw = int(round(self.tick / PRICE_SCALE))
        self.features = FeatureExtractor(self.cfg.features, self.tick)

        # Episode state (populated by reset()).
        self.engine: MarketReplay | None = None
        self.agent: AgentOrderBook | None = None
        self.inventory = 0
        self.cash = 0.0
        self._prev_mid_px = 0.0
        self._half_spreads: deque[float] = deque(maxlen=self.cfg.spread_ma_window)
        self._theta_ask = 0
        self._theta_bid = 0
        self._spread_scale_ticks = 1

    @property
    def feature_names(self) -> tuple[str, ...]:
        return STATE_FEATURE_NAMES

    # -- episode lifecycle ----------------------------------------------------
    def reset(self) -> np.ndarray:
        self.engine = MarketReplay(self.session)
        model = self._queue_factory or ExactFIFOQueue()
        self.agent = AgentOrderBook(self.engine.book, model)
        self.engine.attach_agent(self.agent)
        self.features.reset()
        self.inventory = 0
        self.cash = 0.0
        self._half_spreads.clear()
        self._theta_ask = 0
        self._theta_bid = 0

        self.engine.warmup()
        self._prev_mid_px = self._mid_px()
        self._update_spread_scale()
        market = self.features.update(self.engine.book, 0.0)
        return self._state(market)

    @property
    def done(self) -> bool:
        return self.engine is None or self.engine.done

    # -- stepping -------------------------------------------------------------
    def step(self, action: int) -> tuple[np.ndarray, float, bool, StepInfo]:
        if self.engine is None:
            raise RuntimeError("call reset() before step()")
        if not (0 <= action < N_ACTIONS):
            raise ValueError(f"action {action} out of range [0, {N_ACTIONS})")

        self.agent.cancel(BID)
        self.agent.cancel(ASK)
        spread_pnl = 0.0

        if action == CLEAR_ACTION_ID:
            spread_pnl += self._market_order_clear()
            self._theta_ask = 0
            self._theta_bid = 0
        else:
            spread_pnl += self._place_quotes(action)

        inv_before = self.inventory
        mid_before = self._mid_px()

        # Let the market run to the next decision epoch, filling resting quotes.
        signed_volume = self._advance_one_decision()

        # Realise limit-order fills accumulated over the interval.
        mid_after = self._mid_px()
        for fill in self.agent.fills:
            is_buy = fill.side == BID
            spread_pnl += self._apply_fill(is_buy, fill.price * PRICE_SCALE, fill.size, mid_after)
        n_fills = len(self.agent.fills)
        self.agent.fills.clear()

        dmid = mid_after - mid_before
        breakdown = compute_reward(
            spread_pnl, inv_before, dmid, kind=self.cfg.reward, eta=self.cfg.eta
        )

        self._update_spread_scale()
        market = self.features.update(self.engine.book, signed_volume)
        state = self._state(market)
        equity = self.cash + self.inventory * mid_after
        info = StepInfo(
            reward=breakdown.reward,
            spread_pnl=breakdown.spread_pnl,
            inventory_pnl=breakdown.inventory_pnl,
            inventory=self.inventory,
            mid=mid_after,
            equity=equity,
            n_fills=n_fills,
            action=action,
            theta_ask=self._theta_ask,
            theta_bid=self._theta_bid,
            spread_scale_ticks=self._spread_scale_ticks,
        )
        return state, breakdown.reward, self.done, info

    # -- quoting --------------------------------------------------------------
    def _place_quotes(self, action: int) -> float:
        theta_a = ACTION_THETA_ASK[action]
        theta_b = ACTION_THETA_BID[action]
        self._theta_ask = theta_a
        self._theta_bid = theta_b

        mid_px = self._mid_px()
        scale_px = self._spread_scale_ticks * self.tick
        ask_raw = self._round_to_tick(mid_px + theta_a * scale_px)
        bid_raw = self._round_to_tick(mid_px - theta_b * scale_px)
        seq = self.engine.current_seq + 1

        # Inventory guard: only post a side if a full fill keeps us within bounds.
        if self.inventory <= self.cfg.max_inventory - self.cfg.order_size:
            self.agent.place(BID, bid_raw, self.cfg.order_size, seq)
        if self.inventory >= self.cfg.min_inventory + self.cfg.order_size:
            self.agent.place(ASK, ask_raw, self.cfg.order_size, seq)
        return 0.0

    def _market_order_clear(self) -> float:
        """Action 9: market order to clear alpha * inventory. Returns its spread PnL."""
        qty = int(round(self.cfg.clear_alpha * abs(self.inventory)))
        if qty <= 0:
            return 0.0
        mid_px = self._mid_px()
        is_buy = self.inventory < 0  # buy to cover a short, sell to reduce a long
        fills = self._walk_book(is_buy, qty)
        spread_pnl = 0.0
        for price_px, size in fills:
            spread_pnl += self._apply_fill(is_buy, price_px, size, mid_px)
        return spread_pnl

    def _walk_book(self, is_buy: bool, qty: int) -> list[tuple[float, int]]:
        """Walk the opposite side of the book for a marketable order (no book impact)."""
        book = self.engine.book
        bids, asks = book.depth(64)
        levels = asks if is_buy else bids  # buy lifts asks, sell hits bids
        fills: list[tuple[float, int]] = []
        remaining = qty
        for price_raw, size in levels:
            if remaining <= 0:
                break
            take = min(remaining, size)
            fills.append((price_raw * PRICE_SCALE, take))
            remaining -= take
        return fills

    # -- fills / accounting ---------------------------------------------------
    def _apply_fill(self, is_buy: bool, price_px: float, size: int, mid_px: float) -> float:
        """Update inventory/cash and return the fill's spread PnL relative to ``mid_px``."""
        if is_buy:
            self.inventory += size
            self.cash -= size * price_px
            return size * (mid_px - price_px)
        self.inventory -= size
        self.cash += size * price_px
        return size * (price_px - mid_px)

    # -- replay control -------------------------------------------------------
    def _advance_one_decision(self) -> float:
        start_ts = self.engine.ts
        start_bb = self.engine.book.best_bid()
        start_ba = self.engine.book.best_ask()
        cfg = self.cfg

        def stop(engine: MarketReplay) -> bool:
            elapsed = engine.ts - start_ts
            if elapsed >= cfg.max_interval_ns:
                return True
            if elapsed < cfg.min_interval_ns:
                return False
            tob_changed = (
                engine.book.best_bid() != start_bb or engine.book.best_ask() != start_ba
            )
            return tob_changed

        _, signed_volume = self.engine.advance(stop)
        return signed_volume

    # -- helpers --------------------------------------------------------------
    def _mid_px(self) -> float:
        mid = self.engine.book.mid()
        return mid * PRICE_SCALE if mid is not None else self._prev_mid_px

    def _round_to_tick(self, price_px: float) -> int:
        return int(round(price_px / self.tick)) * self.tick_raw

    def _update_spread_scale(self) -> None:
        spread = self.engine.book.spread()
        if spread is not None:
            self._half_spreads.append((spread * PRICE_SCALE) / 2.0)
        if self._half_spreads:
            mean_half = float(np.mean(self._half_spreads))
            self._spread_scale_ticks = max(1, int(round(mean_half / self.tick)))
        else:
            self._spread_scale_ticks = 1

    def _state(self, market: np.ndarray) -> np.ndarray:
        inv_norm = self.inventory / self.cfg.max_inventory
        agent = np.array([inv_norm, self._theta_ask / 5.0, self._theta_bid / 5.0], dtype=np.float64)
        return np.concatenate([market, agent])
