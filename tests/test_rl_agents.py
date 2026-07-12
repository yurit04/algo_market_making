"""Tests for the TD control agents (learning, schedules, persistence, all variants)."""

from __future__ import annotations

import numpy as np
import pytest

from algo_mm.rl.agents import TDAgent
from algo_mm.rl.config import AgentConfig
from algo_mm.rl.env import STATE_FEATURE_NAMES

NAMES = STATE_FEATURE_NAMES
IMB = NAMES.index("book_imbalance")


def _state(sig: float) -> np.ndarray:
    s = np.zeros(len(NAMES))
    s[IMB] = sig
    return s


@pytest.mark.parametrize("algorithm", ["sarsa", "q", "expected_sarsa", "r_learning"])
def test_agent_learns_contextual_bandit(algorithm):
    """action == sign(imbalance) is optimal; the agent should learn it for every variant."""
    ag = TDAgent(NAMES, 2, AgentConfig(algorithm=algorithm, epsilon=0.2,
                 learning_rate=0.1, epsilon_decay_episodes=1),
                 rng=np.random.default_rng(0))
    rng = np.random.default_rng(1)
    for _ in range(5000):
        sig = 1.0 if rng.random() < 0.5 else -1.0
        s = _state(sig)
        a, greedy = ag.act(s)
        r = 1.0 if a == (1 if sig > 0 else 0) else 0.0
        ag.start_episode()
        ag.update(s, a, r, s, a, done=True, action_was_greedy=greedy)
    assert ag.greedy_action(_state(1.0)) == 1
    assert ag.greedy_action(_state(-1.0)) == 0


def test_epsilon_decays_to_floor():
    cfg = AgentConfig(epsilon=0.7, epsilon_floor=0.01, epsilon_decay_episodes=10)
    ag = TDAgent(NAMES, 10, cfg)
    assert ag.epsilon == 0.7
    for _ in range(10):
        ag.end_episode()
    assert abs(ag.epsilon - 0.01) < 1e-9


def test_greedy_action_is_deterministic():
    ag = TDAgent(NAMES, 10, AgentConfig(epsilon=1.0))  # fully exploratory policy...
    s = _state(0.5)
    # ...but greedy_action ignores epsilon and is deterministic.
    assert ag.greedy_action(s) == ag.greedy_action(s)


def test_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    ag = TDAgent(NAMES, 10, AgentConfig(algorithm="r_learning"), rng=rng)
    # Train on several distinct states so the IHT has a non-trivial tile->index map.
    for sig in rng.uniform(-1, 1, size=40):
        s = _state(float(sig))
        a, g = ag.act(s)
        ag.start_episode()
        ag.update(s, a, 1.0, s, a, done=False, action_was_greedy=g)
    ag.rho = 0.123
    probe = _state(0.42)
    q_before = ag.value.q_all(probe, 10)

    path = tmp_path / "agent.npz"
    ag.save(path)
    ag2 = TDAgent(NAMES, 10, AgentConfig(algorithm="r_learning"))
    # Perturb ag2's IHT with a *different* state first: if the IHT map were not restored,
    # the tile indices for `probe` would diverge and q would not match.
    ag2.value.q_all(_state(-0.9), 10)
    ag2.load(path)
    assert np.allclose(ag2.value.q_all(probe, 10), q_before)
    assert ag2.rho == 0.123
