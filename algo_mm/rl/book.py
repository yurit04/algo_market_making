"""
Level-3 (order-by-order) limit order book reconstructed from Databento MBO events.

CME MDP3 / Databento MBO semantics (verified against GLBX.MDP3 ES data)
----------------------------------------------------------------------
* The book is mutated **only** by ``A`` (add), ``C`` (cancel/reduce), ``M`` (modify)
  and ``R`` (clear) actions. ``T`` (trade) and ``F`` (fill) messages are trade-flow
  notifications: when a resting order is executed the venue emits a ``T`` (the
  aggressor print) plus one ``F`` per resting order plus a matching ``C`` that
  performs the actual book removal. So we ignore ``T``/``F`` for book maintenance
  and consume them separately for signed volume and agent-fill logic.
* The book is only guaranteed internally consistent (best_bid < best_ask) at
  ``F_LAST`` packet boundaries (``flags & 0x80``); it can be transiently crossed
  mid-packet while a sweep is being applied.
* Prices are fixed-point integers scaled by 1e-9 (``FIXED_PRICE_SCALE``); we keep
  them as raw integers internally and only convert to float at the edges.

The book stores, per price level, an insertion-ordered mapping of resting orders so
that :meth:`OrderBook.volume_ahead` can report the exact FIFO queue volume in front
of a given arrival sequence — the piece the Spooner et al. (2018) paper had to
*approximate* because it only had aggregated market-by-price data.
"""

from __future__ import annotations

from typing import Iterable

# --- side encoding -----------------------------------------------------------
BID = 1
ASK = -1
NO_SIDE = 0

_SIDE_FROM_CHAR = {"B": BID, "A": ASK, "N": NO_SIDE}

# --- action encoding (compact codes used by cached sessions) -----------------
ADD = 0
CANCEL = 1
MODIFY = 2
TRADE = 3
FILL = 4
CLEAR = 5
NONE = 6

_ACTION_FROM_CHAR = {
    "A": ADD,
    "C": CANCEL,
    "M": MODIFY,
    "T": TRADE,
    "F": FILL,
    "R": CLEAR,
    "N": NONE,
}

# Databento fixed-point price scale (raw integer price * PRICE_SCALE = float price).
PRICE_SCALE = 1e-9
UNDEF_PRICE = 9223372036854775807

# Databento record flag: last message in an event packet. The book is only
# guaranteed internally consistent (uncrossed) at F_LAST boundaries.
F_LAST = 0x80


def side_from_char(ch: str) -> int:
    """Map a Databento side character (``B``/``A``/``N``) to ``BID``/``ASK``/``NO_SIDE``."""
    return _SIDE_FROM_CHAR[ch]


def action_from_char(ch: str) -> int:
    """Map a Databento action character to a compact integer code."""
    return _ACTION_FROM_CHAR[ch]


class Order:
    """A single resting limit order. ``seq`` is the arrival sequence used for FIFO priority."""

    __slots__ = ("order_id", "side", "price", "size", "seq")

    def __init__(self, order_id: int, side: int, price: int, size: int, seq: int) -> None:
        self.order_id = order_id
        self.side = side
        self.price = price
        self.size = size
        self.seq = seq

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        s = "BID" if self.side == BID else "ASK"
        return f"Order(id={self.order_id}, {s}, px={self.price}, sz={self.size}, seq={self.seq})"


class OrderBook:
    """
    Order-by-order limit order book.

    Aggregate size per price (``bid_sz`` / ``ask_sz``) is kept for O(1) depth/feature
    queries; a per-level insertion-ordered dict of orders (``bid_levels`` /
    ``ask_levels``) is kept for exact FIFO queue accounting.
    """

    def __init__(self) -> None:
        self.orders: dict[int, Order] = {}
        self.bid_levels: dict[int, dict[int, Order]] = {}
        self.ask_levels: dict[int, dict[int, Order]] = {}
        self.bid_sz: dict[int, int] = {}
        self.ask_sz: dict[int, int] = {}
        # Cached best prices; None means "invalidated, recompute lazily". Maintaining
        # these incrementally turns best_bid/best_ask/is_crossed (called at every
        # F_LAST boundary) into O(1) reads instead of O(levels) max/min scans.
        self._best_bid: int | None = None
        self._best_ask: int | None = None

    # -- internal helpers -----------------------------------------------------
    def _levels(self, side: int) -> dict[int, dict[int, Order]]:
        return self.bid_levels if side == BID else self.ask_levels

    def _sizes(self, side: int) -> dict[int, int]:
        return self.bid_sz if side == BID else self.ask_sz

    def _insert(self, order: Order) -> None:
        self.orders[order.order_id] = order
        self._levels(order.side).setdefault(order.price, {})[order.order_id] = order
        sizes = self._sizes(order.side)
        sizes[order.price] = sizes.get(order.price, 0) + order.size
        if order.side == BID:
            if self._best_bid is None or order.price > self._best_bid:
                self._best_bid = order.price
        else:
            if self._best_ask is None or order.price < self._best_ask:
                self._best_ask = order.price

    def _remove(self, order: Order) -> None:
        levels = self._levels(order.side)
        sizes = self._sizes(order.side)
        level = levels.get(order.price)
        if level is not None:
            level.pop(order.order_id, None)
            if not level:
                levels.pop(order.price, None)
        remaining = sizes.get(order.price, 0) - order.size
        if remaining > 0:
            sizes[order.price] = remaining
        else:
            sizes.pop(order.price, None)
            # Level emptied: invalidate the cached best if it pointed here.
            if order.side == BID and self._best_bid == order.price:
                self._best_bid = None
            elif order.side == ASK and self._best_ask == order.price:
                self._best_ask = None
        self.orders.pop(order.order_id, None)

    # -- mutations ------------------------------------------------------------
    def add(self, order_id: int, side: int, price: int, size: int, seq: int) -> None:
        self._insert(Order(order_id, side, price, size, seq))

    def cancel(self, order_id: int, size: int) -> None:
        """Reduce a resting order by ``size`` (removing it if fully cancelled)."""
        order = self.orders.get(order_id)
        if order is None:
            return
        if size >= order.size:
            self._remove(order)
            return
        order.size -= size
        sizes = self._sizes(order.side)
        sizes[order.price] -= size

    def modify(self, order_id: int, side: int, price: int, size: int, seq: int) -> None:
        """
        Apply a modify. A pure size *reduction* at the same price keeps queue
        priority; a price change or size increase loses priority (re-queued at back
        with the new sequence), consistent with CME matching rules.
        """
        order = self.orders.get(order_id)
        if order is None:
            self.add(order_id, side, price, size, seq)
            return
        if price == order.price and size <= order.size:
            delta = order.size - size
            order.size = size
            self._sizes(order.side)[order.price] -= delta
            if size == 0:
                self._remove(order)
            return
        self._remove(order)
        self.add(order_id, side, price, size, seq)

    def clear(self) -> None:
        self.orders.clear()
        self.bid_levels.clear()
        self.ask_levels.clear()
        self.bid_sz.clear()
        self.ask_sz.clear()
        self._best_bid = None
        self._best_ask = None

    def step(self, action: int, side: int, price: int, size: int, order_id: int, seq: int) -> None:
        """Dispatch one compact-coded market event to the book (ignores TRADE/FILL/NONE)."""
        if action == ADD:
            self.add(order_id, side, price, size, seq)
        elif action == CANCEL:
            self.cancel(order_id, size)
        elif action == MODIFY:
            self.modify(order_id, side, price, size, seq)
        elif action == CLEAR:
            self.clear()
        # TRADE, FILL, NONE: no book mutation.

    # -- queries --------------------------------------------------------------
    def best_bid(self) -> int | None:
        if self._best_bid is None and self.bid_sz:
            self._best_bid = max(self.bid_sz)
        return self._best_bid

    def best_ask(self) -> int | None:
        if self._best_ask is None and self.ask_sz:
            self._best_ask = min(self.ask_sz)
        return self._best_ask

    def size_at(self, side: int, price: int) -> int:
        return self._sizes(side).get(price, 0)

    def mid(self) -> float | None:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def spread(self) -> int | None:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return ba - bb

    def is_crossed(self) -> bool:
        bb, ba = self.best_bid(), self.best_ask()
        return bb is not None and ba is not None and bb >= ba

    def volume_ahead(self, side: int, price: int, placement_seq: int) -> int:
        """
        Exact FIFO volume resting in front of an order that arrived at ``placement_seq``.

        Iterates the level's insertion-ordered orders (which are in arrival/sequence
        order) and sums the sizes of those that arrived strictly earlier.
        """
        level = self._levels(side).get(price)
        if not level:
            return 0
        total = 0
        for order in level.values():
            if order.seq >= placement_seq:
                break
            total += order.size
        return total

    def depth(self, n: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Return top-``n`` (price, size) levels as (bids desc, asks asc)."""
        bids = sorted(self.bid_sz.items(), key=lambda kv: kv[0], reverse=True)[:n]
        asks = sorted(self.ask_sz.items(), key=lambda kv: kv[0])[:n]
        return bids, asks


def build_book(
    actions: Iterable[int],
    sides: Iterable[int],
    prices: Iterable[int],
    sizes: Iterable[int],
    order_ids: Iterable[int],
    seqs: Iterable[int],
) -> OrderBook:
    """Replay parallel event arrays into a fresh :class:`OrderBook` (convenience helper)."""
    book = OrderBook()
    for a, sd, px, sz, oid, seq in zip(actions, sides, prices, sizes, order_ids, seqs):
        book.step(a, sd, px, sz, oid, seq)
    return book
