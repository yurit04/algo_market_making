"""
Policies, episode rollouts, and performance metrics (Spooner et al. 2018, §5).

Performance is reported with the paper's normalisation so results are comparable across
instruments and sessions:

* Normalised daily PnL (ND-PnL): session PnL divided by the average market spread —
  i.e. how many market spreads' worth of profit was captured.
* Mean absolute position (MAP): the average |inventory|, a proxy for how speculative
  (inventory-hungry) a strategy is.

Benchmarks mirror the paper: fixed symmetric-spread strategies and a random policy over
the action space (both rely on the environment's inventory clearing at the bounds).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from algo_mm.rl.agent_orders import ExactFIFOQueue, QueueModel, UniformCancelQueue
from algo_mm.rl.config import EnvConfig
from algo_mm.rl.env import MarketMakingEnv
from algo_mm.rl.replay import CachedSession

# A policy maps a state vector to a discrete action id.
Policy = Callable[[np.ndarray], int]


def make_queue(name: str) -> QueueModel:
    if name == "exact":
        return ExactFIFOQueue()
    if name == "uniform":
        return UniformCancelQueue()
    raise ValueError(f"queue_model must be exact|uniform, got {name!r}")


class FixedPolicy:
    """Always quote a fixed symmetric distance: actions 0-4 are theta_a = theta_b = 1..5."""

    def __init__(self, theta: int = 3) -> None:
        if not 1 <= theta <= 5:
            raise ValueError("theta must be in 1..5")
        self.action = theta - 1

    def __call__(self, state: np.ndarray) -> int:
        return self.action


class RandomPolicy:
    def __init__(self, n_actions: int, rng: np.random.Generator | None = None) -> None:
        self.n_actions = n_actions
        self.rng = rng or np.random.default_rng()

    def __call__(self, state: np.ndarray) -> int:
        return int(self.rng.integers(self.n_actions))


@dataclass
class EpisodeResult:
    rewards: np.ndarray
    equity: np.ndarray       # mark-to-market equity curve (index points)
    inventory: np.ndarray
    spread_ticks: np.ndarray
    n_fills: int
    steps: int

    def metrics(self, tick_size: float) -> dict[str, float]:
        avg_spread = float(np.mean(self.spread_ticks)) * tick_size
        final_pnl = float(self.equity[-1]) if len(self.equity) else 0.0
        nd_pnl = final_pnl / avg_spread if avg_spread > 0 else 0.0
        return {
            "nd_pnl": nd_pnl,
            "map": float(np.mean(np.abs(self.inventory))) if len(self.inventory) else 0.0,
            "final_pnl": final_pnl,
            "mean_reward": float(np.mean(self.rewards)) if len(self.rewards) else 0.0,
            "n_fills": self.n_fills,
            "steps": self.steps,
        }


def run_episode(
    env: MarketMakingEnv,
    policy: Policy,
    *,
    max_steps: int | None = None,
) -> EpisodeResult:
    """Roll out ``policy`` (no learning) over one episode and collect diagnostics."""
    state = env.reset()
    rewards: list[float] = []
    equity: list[float] = []
    inventory: list[int] = []
    spreads: list[float] = []
    n_fills = 0
    steps = 0
    spread_idx = 0  # spread_ticks is the first market feature

    while not env.done:
        action = policy(state)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        equity.append(info.equity)
        inventory.append(info.inventory)
        spreads.append(state[spread_idx])
        n_fills += info.n_fills
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break

    return EpisodeResult(
        rewards=np.asarray(rewards),
        equity=np.asarray(equity),
        inventory=np.asarray(inventory),
        spread_ticks=np.asarray(spreads),
        n_fills=n_fills,
        steps=steps,
    )


def evaluate_policy(
    session_paths: list,
    env_cfg: EnvConfig,
    policy_factory: Callable[[MarketMakingEnv], Policy],
    *,
    queue_model: str = "exact",
    max_steps: int | None = None,
) -> dict[str, float]:
    """Average per-session metrics of a policy over a set of cached sessions."""
    per_session = []
    for path in session_paths:
        session = CachedSession.from_parquet(path)
        env = MarketMakingEnv(session, env_cfg, queue_model=make_queue(queue_model))
        result = run_episode(env, policy_factory(env), max_steps=max_steps)
        per_session.append(result.metrics(env_cfg.tick_size))

    keys = ("nd_pnl", "map", "final_pnl", "mean_reward", "n_fills", "steps")
    agg = {k: float(np.mean([m[k] for m in per_session])) for k in keys}
    agg["nd_pnl_std"] = float(np.std([m["nd_pnl"] for m in per_session]))
    agg["n_sessions"] = len(per_session)
    return agg
