"""Tests for tile coding, the LCTC value function, and eligibility traces."""

from __future__ import annotations

import numpy as np

from algo_mm.rl.config import (
    AGENT_STATE_FEATURES,
    FEATURE_RANGES,
    MARKET_STATE_FEATURES,
)
from algo_mm.rl.env import STATE_FEATURE_NAMES
from algo_mm.rl.tilecoding import IHT, LCTC, tiles


def _lctc(**kw):
    full = STATE_FEATURE_NAMES
    return LCTC(full, (AGENT_STATE_FEATURES, MARKET_STATE_FEATURES, full),
               FEATURE_RANGES, (0.6, 0.1, 0.3), **kw)


def test_iht_consistent_and_bounded():
    iht = IHT(16)
    a = iht.index((1, 2, 3))
    assert iht.index((1, 2, 3)) == a       # stable
    assert iht.index((1, 2, 4)) != a       # distinct
    for i in range(100):
        assert 0 <= iht.index((i, i)) < 16  # bounded even when overfull


def test_tiles_count_and_action_dependence():
    iht = IHT(4096)
    f = np.array([0.3, 0.7])
    t0 = tiles(iht, 8, f, action=0)
    t1 = tiles(iht, 8, f, action=1)
    assert len(t0) == 8
    assert t0 != t1  # action folded into the tiling


def test_q_all_matches_q_per_action():
    v = _lctc(num_tilings=8, tiles_per_dim=4, iht_size=4096)
    state = np.linspace(-0.5, 0.5, len(STATE_FEATURE_NAMES))
    # Put some signal into the weights via a trace update.
    v.accumulate_replacing(v.active_all(state, 2))
    v.update_and_decay(alpha=0.1, delta=1.0, decay=0.9)
    q_all = v.q_all(state, 10)
    for a in range(10):
        assert abs(q_all[a] - v.q(state, a)) < 1e-9


def test_q_from_active_matches_q():
    v = _lctc(num_tilings=8, tiles_per_dim=4, iht_size=4096)
    state = np.zeros(len(STATE_FEATURE_NAMES))
    v.accumulate_replacing(v.active_all(state, 0))
    v.update_and_decay(alpha=0.05, delta=2.0, decay=0.9)
    active = v.active_all(state, 0)
    assert abs(v.q_from_active(active) - v.q(state, 0)) < 1e-9


def test_trace_update_moves_value_toward_target():
    v = _lctc(num_tilings=16, tiles_per_dim=8, iht_size=1 << 16)
    state = np.zeros(len(STATE_FEATURE_NAMES))
    before = v.q(state, 0)
    for _ in range(20):
        v.accumulate_replacing(v.active_all(state, 0))
        delta = 1.0 - v.q(state, 0)
        v.update_and_decay(alpha=0.1, delta=delta, decay=0.9)
    after = v.q(state, 0)
    assert after > before
    assert after < 1.5  # converging toward the target, not diverging


def test_reset_traces_clears():
    v = _lctc(num_tilings=8, tiles_per_dim=4, iht_size=4096)
    state = np.zeros(len(STATE_FEATURE_NAMES))
    v.accumulate_replacing(v.active_all(state, 0))
    v.reset_traces()
    for c in v.codings:
        assert c._tr_n == 0
