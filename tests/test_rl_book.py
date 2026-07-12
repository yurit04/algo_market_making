"""Unit tests for L3 order-book reconstruction from MBO-coded events."""

from __future__ import annotations

from algo_mm.rl.book import (
    ADD,
    ASK,
    BID,
    CANCEL,
    CLEAR,
    MODIFY,
    OrderBook,
    action_from_char,
    side_from_char,
)


def test_char_encoding():
    assert side_from_char("B") == BID
    assert side_from_char("A") == ASK
    assert action_from_char("A") == ADD
    assert action_from_char("C") == CANCEL
    assert action_from_char("R") == CLEAR


def test_add_and_top_of_book():
    book = OrderBook()
    book.add(1, BID, 100, 5, seq=1)
    book.add(2, BID, 99, 3, seq=2)
    book.add(3, ASK, 101, 4, seq=3)
    book.add(4, ASK, 102, 2, seq=4)

    assert book.best_bid() == 100
    assert book.best_ask() == 101
    assert book.mid() == 100.5
    assert book.spread() == 1
    assert book.size_at(BID, 100) == 5
    assert not book.is_crossed()

    bids, asks = book.depth(2)
    assert bids == [(100, 5), (99, 3)]
    assert asks == [(101, 4), (102, 2)]


def test_aggregate_size_and_multiple_orders_per_level():
    book = OrderBook()
    book.add(1, BID, 100, 5, seq=1)
    book.add(2, BID, 100, 7, seq=2)
    assert book.size_at(BID, 100) == 12


def test_partial_and_full_cancel():
    book = OrderBook()
    book.add(1, BID, 100, 5, seq=1)
    book.cancel(1, 2)
    assert book.size_at(BID, 100) == 3
    book.cancel(1, 3)  # cancels the full remaining
    assert book.size_at(BID, 100) == 0
    assert book.best_bid() is None
    assert 1 not in book.orders


def test_modify_size_reduction_keeps_priority():
    book = OrderBook()
    book.add(1, BID, 100, 5, seq=1)
    book.add(2, BID, 100, 4, seq=2)
    book.modify(1, BID, 100, 3, seq=10)  # pure size reduction, keeps position
    assert book.size_at(BID, 100) == 7
    # order 1 still ahead of order 2 (seq preserved -> volume_ahead of a late arrival sees both)
    assert book.volume_ahead(BID, 100, placement_seq=1000) == 7


def test_modify_price_change_requeues():
    book = OrderBook()
    book.add(1, BID, 100, 5, seq=1)
    book.modify(1, BID, 99, 5, seq=10)  # price change -> re-queued at 99
    assert book.size_at(BID, 100) == 0
    assert book.size_at(BID, 99) == 5
    assert book.best_bid() == 99


def test_clear():
    book = OrderBook()
    book.add(1, BID, 100, 5, seq=1)
    book.add(2, ASK, 101, 5, seq=2)
    book.clear()
    assert book.best_bid() is None and book.best_ask() is None
    assert not book.orders


def test_volume_ahead_fifo_order():
    book = OrderBook()
    book.add(1, BID, 100, 3, seq=1)
    book.add(2, BID, 100, 2, seq=2)
    book.add(3, BID, 100, 4, seq=5)
    # An order arriving at seq=3 sits behind seq 1 and 2 only.
    assert book.volume_ahead(BID, 100, placement_seq=3) == 5
    # An order arriving first sees nothing ahead.
    assert book.volume_ahead(BID, 100, placement_seq=0) == 0
    # A late order sees everything.
    assert book.volume_ahead(BID, 100, placement_seq=100) == 9


def test_step_dispatch_ignores_trade_and_fill():
    from algo_mm.rl.book import FILL, TRADE

    book = OrderBook()
    book.step(ADD, BID, 100, 5, 1, 1)
    book.step(TRADE, ASK, 100, 2, 999, 2)  # no book mutation
    book.step(FILL, BID, 100, 2, 1, 2)     # no book mutation
    assert book.size_at(BID, 100) == 5
