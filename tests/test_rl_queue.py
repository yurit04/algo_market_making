"""
Tests for the agent queue models (exact FIFO vs. uniform-cancel) and replay-driven
fill simulation, including fill-cancel disambiguation.
"""

from __future__ import annotations

import numpy as np

from algo_mm.rl.agent_orders import (
    AgentOrder,
    AgentOrderBook,
    ExactFIFOQueue,
    UniformCancelQueue,
)
from algo_mm.rl.book import ASK, BID, OrderBook
from algo_mm.rl.replay import CachedSession, replay_session


def _bid_book_with(*orders):
    book = OrderBook()
    for oid, px, sz, seq in orders:
        book.add(oid, BID, px, sz, seq)
    return book


def test_exact_fifo_fills_only_after_queue_consumed():
    book = _bid_book_with((1, 100, 3, 1), (2, 100, 2, 2))  # 5 ahead
    order = AgentOrder(side=BID, price=100, size=5, placement_seq=100)
    model = ExactFIFOQueue()

    # Trade of 3 < 5 ahead -> no fill.
    assert model.on_trade(book, order, 3) == 0
    assert order.remaining == 5
    book.cancel(1, 3)  # the trade removed order 1 from the book

    # Now 2 ahead; trade of 3 -> 1 unit reaches the agent.
    assert model.on_trade(book, order, 3) == 1
    assert order.remaining == 4


def test_uniform_matches_exact_when_no_cancels():
    book = _bid_book_with((1, 100, 3, 1), (2, 100, 2, 2))
    order = AgentOrder(side=BID, price=100, size=5, placement_seq=100)
    model = UniformCancelQueue()
    model.on_place(book, order)  # ahead = 5

    assert model.on_trade(book, order, 3) == 0
    assert model.on_trade(book, order, 3) == 1
    assert order.remaining == 4


def test_exact_and_uniform_diverge_on_cancel_with_volume_behind():
    # Orders 1,2 ahead (10). Agent joins the back. Order 3 arrives behind (5).
    book = _bid_book_with((1, 100, 4, 1), (2, 100, 6, 2))
    exact_order = AgentOrder(side=BID, price=100, size=1, placement_seq=1000)
    unif_order = AgentOrder(side=BID, price=100, size=1, placement_seq=1000)
    exact = ExactFIFOQueue()
    uniform = UniformCancelQueue()
    exact.on_place(book, exact_order)
    uniform.on_place(book, unif_order)  # ahead = 10
    book.add(3, BID, 100, 5, seq=1001)  # behind the agent

    # A genuine cancel of order 1 (4 units), which is ahead of the agent.
    uniform.on_cancel(book, unif_order, 4)  # queried before the book is mutated
    book.cancel(1, 4)

    exact_ahead = book.volume_ahead(BID, 100, exact_order.placement_seq)
    assert exact_ahead == 6  # order 2 only; order 3 is behind
    # Uniform smears the cancel across the (ahead+behind) queue, over-estimating ahead.
    assert unif_order.ahead > exact_ahead
    assert abs(unif_order.ahead - (10 - 4 * (10 / 15))) < 1e-9


def _session(events) -> CachedSession:
    """Build a CachedSession from (ts, action, side, price, size, oid, seq, flags) tuples."""
    cols = list(zip(*events))
    return CachedSession(
        ts_event=np.asarray(cols[0], np.int64),
        action=np.asarray(cols[1], np.int8),
        side=np.asarray(cols[2], np.int8),
        price=np.asarray(cols[3], np.int64),
        size=np.asarray(cols[4], np.int64),
        order_id=np.asarray(cols[5], np.int64),
        sequence=np.asarray(cols[6], np.int64),
        flags=np.asarray(cols[7], np.uint8),
    )


def test_replay_exact_fill_with_fill_cancel_disambiguation():
    from algo_mm.rl.book import ADD, CANCEL, FILL, TRADE, F_LAST

    # Two resting bids (3 + 2) at price 100, then two sell aggressor sweeps.
    adds = [
        (1, ADD, BID, 100, 3, 1, 1, F_LAST),
        (1, ADD, BID, 100, 2, 2, 2, F_LAST),
    ]
    # First sweep of 3: trade + fill + fill-cancel of order 1.
    sweep1 = [
        (10, TRADE, ASK, 100, 3, 999, 10, 0),
        (10, FILL, BID, 100, 3, 1, 10, 0),
        (10, CANCEL, BID, 100, 3, 1, 11, F_LAST),
    ]
    # Second sweep of 3 against remaining 2: order 2 fully filled, 1 unit reaches agent.
    sweep2 = [
        (20, TRADE, ASK, 100, 3, 998, 20, 0),
        (20, FILL, BID, 100, 2, 2, 20, 0),
        (20, CANCEL, BID, 100, 2, 2, 21, F_LAST),
    ]

    book = OrderBook()
    replay_session(_session(adds), book=book)  # build the resting queue

    agent = AgentOrderBook(book, ExactFIFOQueue())
    agent.place(BID, 100, 5, seq=100)  # joins the back, 5 ahead

    replay_session(_session(sweep1 + sweep2), book=book, agent=agent)

    assert len(agent.fills) == 1
    assert agent.fills[0].size == 1
    assert agent.fills[0].side == BID
    assert agent.fills[0].price == 100
