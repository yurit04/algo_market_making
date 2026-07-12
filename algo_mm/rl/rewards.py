"""
Reward functions from Spooner et al. (2018), §4.2 (Eqs. 3-6).

Per decision interval the agent's executions produce a *spread PnL* measured relative
to the mid-price,

    spread_pnl = sum over fills of  q * (m - p)      for buys  (agent bid / cover)
                                    q * (p - m)      for sells (agent ask / reduce)

which is exactly psi_a + psi_b in the paper (Eq. 3), generalised to also cover the
action-9 market order (whose executions cross the spread and therefore contribute
negative spread PnL). The non-dampened incremental PnL adds a mark-to-market term on
the inventory carried across the interval,

    Psi = spread_pnl + inv * dmid                                        (Eq. 3)

The three studied rewards damp the speculative inventory term to different degrees:

    pnl:        r = Psi                                                  (Eq. 4)
    symmetric:  r = Psi - eta * inv * dmid                              (Eq. 5)
    asymmetric: r = Psi - max(0, eta * inv * dmid)                      (Eq. 6)

Asymmetric damping removes upside from speculation while keeping the downside, which
the paper shows drives the agent toward small, neutral inventories.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardBreakdown:
    """Diagnostic decomposition of a single-step reward."""

    reward: float
    spread_pnl: float      # psi_a + psi_b (spread capture, incl. market-order slippage)
    inventory_pnl: float   # inv * dmid (mark-to-market on carried inventory)
    damping: float         # amount subtracted from the inventory term


def compute_reward(
    spread_pnl: float,
    inventory: float,
    dmid: float,
    *,
    kind: str = "asymmetric",
    eta: float = 0.6,
) -> RewardBreakdown:
    """
    Evaluate one of the paper's reward functions.

    Parameters
    ----------
    spread_pnl
        psi_a + psi_b for the interval (spread capture relative to mid).
    inventory
        Inventory carried across the interval (Inv(t_i) in Eq. 3).
    dmid
        Change in mid-price over the interval (Delta m).
    kind
        "pnl", "symmetric", or "asymmetric".
    eta
        Damping factor applied to the inventory (speculation) term.
    """
    inventory_pnl = inventory * dmid
    base = spread_pnl + inventory_pnl

    if kind == "pnl":
        damping = 0.0
    elif kind == "symmetric":
        damping = eta * inventory_pnl
    elif kind == "asymmetric":
        damping = max(0.0, eta * inventory_pnl)
    else:
        raise ValueError(f"unknown reward kind {kind!r}")

    return RewardBreakdown(
        reward=base - damping,
        spread_pnl=spread_pnl,
        inventory_pnl=inventory_pnl,
        damping=damping,
    )
