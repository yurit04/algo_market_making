"""Tests for the paper's three reward functions (Eqs. 4-6)."""

from __future__ import annotations

from algo_mm.rl.rewards import compute_reward


def test_pnl_is_undamped():
    r = compute_reward(spread_pnl=2.0, inventory=10, dmid=0.5, kind="pnl")
    assert r.spread_pnl == 2.0
    assert r.inventory_pnl == 5.0
    assert r.damping == 0.0
    assert r.reward == 7.0  # 2 + 10*0.5


def test_symmetric_damps_both_directions():
    up = compute_reward(spread_pnl=0.0, inventory=10, dmid=1.0, kind="symmetric", eta=0.5)
    assert up.reward == 10.0 - 0.5 * 10.0  # Psi - eta*inv*dmid
    down = compute_reward(spread_pnl=0.0, inventory=10, dmid=-1.0, kind="symmetric", eta=0.5)
    # Symmetric damping reduces the magnitude of the loss too.
    assert down.reward == -10.0 - 0.5 * (-10.0)
    assert down.reward == -5.0


def test_asymmetric_damps_only_speculative_profit():
    # Favourable inventory move: profit is damped.
    up = compute_reward(spread_pnl=0.0, inventory=10, dmid=1.0, kind="asymmetric", eta=0.6)
    assert up.reward == 10.0 - 0.6 * 10.0
    # Adverse inventory move: loss kept intact (damping = 0).
    down = compute_reward(spread_pnl=0.0, inventory=10, dmid=-1.0, kind="asymmetric", eta=0.6)
    assert down.damping == 0.0
    assert down.reward == -10.0


def test_spread_capture_survives_damping():
    # Pure spread capture with no inventory is unaffected by any reward variant.
    for kind in ("pnl", "symmetric", "asymmetric"):
        r = compute_reward(spread_pnl=3.0, inventory=0, dmid=5.0, kind=kind, eta=0.9)
        assert r.reward == 3.0
