"""Tests for session discovery/splitting and a guarded end-to-end train+evaluate run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from algo_mm.rl.config import AgentConfig, EnvConfig, TrainConfig
from algo_mm.rl.evaluate import FixedPolicy, RandomPolicy, evaluate_policy
from algo_mm.rl.train import chronological_split, discover_sessions, train

_CACHE = Path("outputs/rl_cache/mbo")
_HAS_CACHE = _CACHE.is_dir() and any(_CACHE.glob("*/*.parquet"))


def test_discover_and_split_is_chronological(tmp_path):
    base = tmp_path / "mbo"
    for sym, day in [("ESM5", "2025-05-08"), ("ESM5", "2025-05-07"), ("ESU5", "2025-06-20")]:
        d = base / sym
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{day}.parquet").touch()

    paths = discover_sessions(base)
    stems = [p.stem for p in paths]
    assert stems == sorted(stems)  # chronological
    train_p, test_p = chronological_split(paths, test_fraction=0.34)
    assert len(train_p) + len(test_p) == 3
    assert test_p[-1].stem == "2025-06-20"  # latest held out


def test_discover_raises_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_sessions(tmp_path)


@pytest.mark.skipif(not _HAS_CACHE, reason="no preprocessed MBO session cache present")
def test_end_to_end_train_and_evaluate(tmp_path):
    env_cfg = EnvConfig(order_size=1, max_inventory=20, min_inventory=-20,
                        reward="asymmetric", eta=0.6, max_interval_ns=1_000_000_000)
    agent_cfg = AgentConfig(algorithm="sarsa", epsilon=0.5, epsilon_decay_episodes=2)
    train_cfg = TrainConfig(cache_dir=str(_CACHE), episodes=2, max_steps_per_episode=300,
                            test_fraction=0.25, seed=0, checkpoint_every=0,
                            checkpoint_dir=str(tmp_path))
    result = train(env_cfg, agent_cfg, train_cfg, progress=False)

    assert len(result.history) == 2
    assert Path(result.checkpoint).exists()
    assert result.train_sessions and result.test_sessions
    for log in result.history:
        assert np.isfinite(log.total_reward)
        assert log.steps <= 300

    # Benchmarks run and produce finite, comparable metrics.
    test = [Path(p) for p in result.test_sessions[:1]]
    rng = np.random.default_rng(0)
    for factory in (lambda e: result.agent.greedy_action,
                    lambda e: FixedPolicy(3),
                    lambda e: RandomPolicy(e.n_actions, rng)):
        m = evaluate_policy(test, env_cfg, factory, queue_model="exact", max_steps=300)
        assert np.isfinite(m["nd_pnl"])
        assert m["n_sessions"] == 1
