"""
Replay a cached MBO session through the order book (and, optionally, an agent).

A cached session (see :mod:`algo_mm.data.preprocess_mbo`) is a set of parallel arrays
of compact-coded events for a single instrument over a single trading session. The
replay applies them to an :class:`~algo_mm.rl.book.OrderBook` in order, exposing:

* book snapshots at ``F_LAST`` packet boundaries (the only consistent states),
* the trade stream (signed by aggressor side), and
* agent fills, when an :class:`~algo_mm.rl.agent_orders.AgentOrderBook` is attached.

The replay is responsible for distinguishing *genuine* cancellations from
*fill-cancels* (the ``C`` messages that remove a resting order after it traded): a
``C`` whose ``order_id`` appeared in an ``F`` at the same ``ts_event`` is a
fill-cancel, already accounted for by the trade walk, so it must not also reduce the
agent's queue in the uniform-cancel model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from algo_mm.rl.agent_orders import AgentOrderBook
from algo_mm.rl.book import (
    CANCEL,
    FILL,
    F_LAST,
    MODIFY,
    OrderBook,
    TRADE,
)

_COLUMNS = ("ts_event", "action", "side", "price", "size", "order_id", "sequence", "flags")


@dataclass
class CachedSession:
    """Parallel event arrays for one instrument / one session (compact codes)."""

    ts_event: np.ndarray   # int64 ns
    action: np.ndarray     # int8 compact action code
    side: np.ndarray       # int8 (+1 bid, -1 ask, 0 none)
    price: np.ndarray      # int64 raw fixed-point (or UNDEF for non-book events)
    size: np.ndarray       # int64
    order_id: np.ndarray   # int64
    sequence: np.ndarray   # int64
    flags: np.ndarray      # uint8
    symbol: str | None = None
    date: str | None = None

    def __len__(self) -> int:
        return int(self.ts_event.shape[0])

    @classmethod
    def from_parquet(cls, path: str | Path) -> "CachedSession":
        path = Path(path)
        df = pd.read_parquet(path)
        meta = {"symbol": None, "date": path.stem}
        if df.attrs:
            meta.update({k: df.attrs[k] for k in ("symbol", "date") if k in df.attrs})
        return cls(
            ts_event=df["ts_event"].to_numpy(np.int64),
            action=df["action"].to_numpy(np.int8),
            side=df["side"].to_numpy(np.int8),
            price=df["price"].to_numpy(np.int64),
            size=df["size"].to_numpy(np.int64),
            order_id=df["order_id"].to_numpy(np.int64),
            sequence=df["sequence"].to_numpy(np.int64),
            flags=df["flags"].to_numpy(np.uint8),
            symbol=meta["symbol"],
            date=meta["date"],
        )


@dataclass
class Trade:
    ts_event: int
    price: int
    size: int
    aggressor_side: int  # +1 buy aggressor lifts asks, -1 sell aggressor hits bids


def replay_session(
    session: CachedSession,
    book: OrderBook | None = None,
    agent: AgentOrderBook | None = None,
    *,
    collect_trades: bool = False,
    on_snapshot=None,
) -> dict:
    """
    Drive ``session`` through ``book`` (and optionally ``agent``).

    Parameters
    ----------
    book
        Order book to mutate (a fresh one is created if omitted).
    agent
        If given, trades are routed to it for fill simulation and genuine cancels are
        reported for the uniform-cancel queue model.
    collect_trades
        Accumulate the trade stream into the result.
    on_snapshot
        Optional callback ``fn(i, ts_event, book)`` invoked at every ``F_LAST``
        boundary (where the book is guaranteed consistent).

    Returns
    -------
    dict with keys ``book``, ``trades`` (list[Trade] or None) and ``fills``
    (list[Fill] or None).
    """
    if book is None:
        book = OrderBook()

    ts = session.ts_event
    act = session.action
    side = session.side
    price = session.price
    size = session.size
    oid = session.order_id
    seq = session.sequence
    flags = session.flags
    n = len(session)

    trades: list[Trade] | None = [] if collect_trades else None
    fill_oids: set[int] = set()
    cur_ts = ts[0] if n else 0

    for i in range(n):
        ts_i = ts[i]
        if ts_i != cur_ts:
            fill_oids.clear()
            cur_ts = ts_i
        a = act[i]

        if a == TRADE:
            if agent is not None:
                agent.on_trade(int(side[i]), int(price[i]), int(size[i]), int(seq[i]))
            if trades is not None:
                trades.append(Trade(int(ts_i), int(price[i]), int(size[i]), int(side[i])))
            continue

        if a == FILL:
            fill_oids.add(int(oid[i]))
            continue

        if a in (CANCEL, MODIFY) and agent is not None:
            # A cancel/modify whose order_id was just filled is a fill-cancel, already
            # captured by the trade walk; only report genuine reductions to the agent.
            if int(oid[i]) not in fill_oids:
                agent.on_book_reduce(int(side[i]), int(price[i]), int(size[i]))

        book.step(int(a), int(side[i]), int(price[i]), int(size[i]), int(oid[i]), int(seq[i]))

        if on_snapshot is not None and (flags[i] & F_LAST) and not book.is_crossed():
            on_snapshot(i, int(ts_i), book)

    return {
        "book": book,
        "trades": trades,
        "fills": agent.fills if agent is not None else None,
    }


class MarketReplay:
    """
    Stateful, resumable replay of one session for use inside the RL environment.

    Unlike :func:`replay_session` (a one-shot batch pass), this advances event by event
    and yields control at decision epochs, so the environment can place quotes, let the
    market run for an interval, and read back the fills. It owns the book and (once
    attached) the agent order book, and performs the same fill-cancel disambiguation.
    """

    def __init__(self, session: CachedSession) -> None:
        self.session = session
        self.book = OrderBook()
        self.agent: AgentOrderBook | None = None
        self.i = 0
        self.n = len(session)
        # Convert the hot columns to Python lists once: native-int list indexing is
        # ~1.5x faster than boxing numpy scalars per event in the replay loop.
        self._ts = session.ts_event.tolist()
        self._act = session.action.tolist()
        self._side = session.side.tolist()
        self._price = session.price.tolist()
        self._size = session.size.tolist()
        self._oid = session.order_id.tolist()
        self._seq = session.sequence.tolist()
        self._flags = session.flags.tolist()
        self.cur_ts = self._ts[0] if self.n else 0
        self.current_seq = 0
        self._fill_oids: set[int] = set()

    def attach_agent(self, agent: AgentOrderBook) -> None:
        self.agent = agent

    @property
    def done(self) -> bool:
        return self.i >= self.n

    @property
    def ts(self) -> int:
        return self.cur_ts

    def advance(self, stop) -> tuple[bool, float]:
        """
        Advance until ``stop(self)`` is True at a consistent boundary, or the session
        ends. Returns ``(done, signed_volume)`` where ``signed_volume`` is the
        aggressor-signed traded volume over the interval (buys +, sells -).
        """
        ts, act, side, price, size, oid, seq, flags = (
            self._ts, self._act, self._side, self._price,
            self._size, self._oid, self._seq, self._flags,
        )
        agent = self.agent
        book = self.book
        fill_oids = self._fill_oids
        signed_volume = 0.0

        while self.i < self.n:
            i = self.i
            ts_i = ts[i]
            if ts_i != self.cur_ts:
                fill_oids.clear()
                self.cur_ts = ts_i
            a = act[i]

            if a == TRADE:
                aggressor = side[i]
                if agent is not None:
                    agent.on_trade(aggressor, price[i], size[i], seq[i])
                signed_volume += aggressor * size[i]
                self.i = i + 1
                continue
            if a == FILL:
                fill_oids.add(oid[i])
                self.i = i + 1
                continue
            if a in (CANCEL, MODIFY) and agent is not None and oid[i] not in fill_oids:
                agent.on_book_reduce(side[i], price[i], size[i])

            book.step(a, side[i], price[i], size[i], oid[i], seq[i])
            self.current_seq = seq[i]
            self.i = i + 1

            if (flags[i] & F_LAST) and not book.is_crossed() and stop(self):
                return self.done, signed_volume

        return True, signed_volume

    def warmup(self) -> bool:
        """Advance to the first boundary where both sides of the book are populated."""
        def both_sides(engine: "MarketReplay") -> bool:
            return engine.book.best_bid() is not None and engine.book.best_ask() is not None

        _, _ = self.advance(both_sides)
        return self.book.best_bid() is not None and self.book.best_ask() is not None
