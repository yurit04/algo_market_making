"""
Phantom agent orders and queue-position models.

The agent's limit orders are simulated *against* the reconstructed historical book
without impacting it (the same no-impact assumption as Spooner et al. 2018, justified
because the agent's size is tiny relative to market volume). The only non-trivial
question is queue position: when a trade arrives at the agent's price, has enough
volume in front of the agent been consumed for the agent to fill?

Two models answer this differently:

* :class:`ExactFIFOQueue` — uses the order-by-order MBO book to compute the exact
  volume resting ahead of the agent at trade time. This is the fidelity upgrade MBO
  data buys us over the paper.
* :class:`UniformCancelQueue` — reproduces the paper's approximation for aggregated
  (market-by-price) data: executions are known from the trade stream and consume the
  queue from the front, but *cancellations* have unknown position and are assumed
  uniformly distributed, so a cancel of size delta reduces the volume ahead by
  ``delta * ahead / (ahead + behind)``.

Both consume the same event stream via :class:`AgentOrderBook`, so a backtest can
switch models to quantify how much the paper's approximation distorts fills.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from algo_mm.rl.book import ASK, BID, OrderBook


@dataclass
class AgentOrder:
    """A resting agent limit order and its (model-dependent) queue state."""

    side: int
    price: int
    size: int
    placement_seq: int
    remaining: int = 0
    # State used only by the approximate (uniform-cancel) model:
    ahead: float = 0.0

    def __post_init__(self) -> None:
        if self.remaining == 0:
            self.remaining = self.size


@dataclass
class Fill:
    """An execution of (part of) an agent order."""

    side: int          # side the agent order was resting on (BID = we buy, ASK = we sell)
    price: int         # raw fixed-point price
    size: int          # filled quantity (>0)
    seq: int           # sequence of the trade that caused the fill


class QueueModel(ABC):
    """Strategy for tracking an agent order's queue position and computing fills."""

    @abstractmethod
    def on_place(self, book: OrderBook, order: AgentOrder) -> None:
        """Initialise queue state when the order is placed (joins the back of its level)."""

    @abstractmethod
    def on_cancel(self, book: OrderBook, order: AgentOrder, size: int) -> None:
        """A *genuine* cancellation of ``size`` at the order's price (not a fill-cancel)."""

    @abstractmethod
    def on_trade(self, book: OrderBook, order: AgentOrder, trade_size: int) -> int:
        """
        A trade of ``trade_size`` at the order's price with the agent on the passive
        side. Return the quantity that executes against the agent and update state.
        """


class ExactFIFOQueue(QueueModel):
    """Exact queue position from the order-by-order book (recomputed at trade time)."""

    def on_place(self, book: OrderBook, order: AgentOrder) -> None:
        # No per-order state needed; volume ahead is derived from the book on demand.
        pass

    def on_cancel(self, book: OrderBook, order: AgentOrder, size: int) -> None:
        # Cancellations of orders ahead are reflected in the book directly, so the
        # next on_trade recomputation already accounts for them. Nothing to do.
        pass

    def on_trade(self, book: OrderBook, order: AgentOrder, trade_size: int) -> int:
        ahead = book.volume_ahead(order.side, order.price, order.placement_seq)
        executed = max(0, min(trade_size - ahead, order.remaining))
        order.remaining -= executed
        return executed


class UniformCancelQueue(QueueModel):
    """
    Paper-style approximation for aggregated depth: trades consume the queue from the
    front (known); cancellations are assumed uniformly distributed through the queue.
    """

    def on_place(self, book: OrderBook, order: AgentOrder) -> None:
        # Joins the back of the level: everything currently resting is ahead of us.
        order.ahead = float(book.size_at(order.side, order.price))

    def on_cancel(self, book: OrderBook, order: AgentOrder, size: int) -> None:
        # ``book`` is queried BEFORE the cancel is applied, so size_at includes the
        # cancelled volume. behind = total resting (excl. agent) - ahead.
        total = float(book.size_at(order.side, order.price))
        behind = max(0.0, total - order.ahead)
        denom = order.ahead + behind
        if denom <= 0:
            return
        order.ahead = max(0.0, order.ahead - size * (order.ahead / denom))

    def on_trade(self, book: OrderBook, order: AgentOrder, trade_size: int) -> int:
        fillable = max(0.0, trade_size - order.ahead)
        executed = int(min(fillable, order.remaining))
        order.ahead = max(0.0, order.ahead - trade_size)
        order.remaining -= executed
        return executed


class AgentOrderBook:
    """
    Manages the agent's resting orders (at most one bid and one ask) against a live
    :class:`OrderBook`, delegating queue accounting to a :class:`QueueModel`.

    The owning replay loop must call:

    * :meth:`on_book_reduce` *before* applying a genuine (non-fill) cancel/modify to
      the book, so the uniform model can read the pre-cancel level size, and
    * :meth:`on_trade` at each ``TRADE`` event *before* the paired fill-cancels are
      applied, so the exact model sees the pre-trade book.
    """

    def __init__(self, book: OrderBook, queue_model: QueueModel | None = None) -> None:
        self.book = book
        self.model = queue_model or ExactFIFOQueue()
        self.orders: dict[int, AgentOrder] = {}  # side -> order
        self.fills: list[Fill] = []

    def place(self, side: int, price: int, size: int, seq: int) -> AgentOrder:
        """Place (or replace) the agent's order on ``side`` at ``price``/``size``."""
        order = AgentOrder(side=side, price=price, size=size, placement_seq=seq)
        self.model.on_place(self.book, order)
        self.orders[side] = order
        return order

    def cancel(self, side: int) -> None:
        self.orders.pop(side, None)

    def active(self, side: int) -> AgentOrder | None:
        return self.orders.get(side)

    def on_book_reduce(self, side: int, price: int, size: int) -> None:
        """Notify of a genuine cancel/modify-reduction of ``size`` at (side, price)."""
        order = self.orders.get(side)
        if order is not None and order.price == price:
            self.model.on_cancel(self.book, order, size)

    def on_trade(self, aggressor_side: int, price: int, trade_size: int, seq: int) -> int:
        """
        Handle a trade print. The passive (filled) side is the opposite of the
        aggressor; the agent fills only if it rests on that passive side at ``price``.
        Returns the executed quantity.
        """
        passive_side = -aggressor_side
        order = self.orders.get(passive_side)
        if order is None or order.price != price:
            return 0
        executed = self.model.on_trade(self.book, order, trade_size)
        if executed > 0:
            self.fills.append(Fill(side=passive_side, price=price, size=executed, seq=seq))
            if order.remaining <= 0:
                self.orders.pop(passive_side, None)
        return executed
