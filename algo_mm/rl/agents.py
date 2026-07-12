"""
Temporal-difference control agents with eligibility traces (Spooner et al. 2018, §4.4).

All agents share the LCTC value function and epsilon-greedy exploration; they differ
only in the TD target used to bootstrap:

    SARSA          target = r + gamma * q(s', a')                 (on-policy)
    Q-learning     target = r + gamma * max_a q(s', a)            (off-policy)
    Expected SARSA target = r + gamma * sum_a pi(a|s') q(s', a)   (epsilon-greedy pi)
    R-learning     delta  = r - rho + max_a q(s', a) - q(s, a)    (average reward, no gamma)

Per the paper, every algorithm uses eligibility traces (the ``(lambda)`` variants);
we use replacing traces stored sparsely in the LCTC. The one-step update ordering
follows Sutton & Barto (set current traces, bootstrap, weight update, decay).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from algo_mm.rl.config import (
    AGENT_STATE_FEATURES,
    FEATURE_RANGES,
    MARKET_STATE_FEATURES,
    AgentConfig,
)
from algo_mm.rl.tilecoding import LCTC

_ALGORITHMS = frozenset({"sarsa", "q", "expected_sarsa", "r_learning"})


class TDAgent:
    """Tile-coding TD control agent (SARSA/Q/Expected-SARSA/R-learning) with traces."""

    def __init__(
        self,
        feature_names: tuple[str, ...],
        n_actions: int,
        config: AgentConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.cfg = config or AgentConfig()
        if self.cfg.algorithm not in _ALGORITHMS:
            raise ValueError(f"algorithm must be one of {sorted(_ALGORITHMS)}")
        self.n_actions = n_actions
        self.rng = rng or np.random.default_rng()

        full = tuple(feature_names)
        self.value = LCTC(
            feature_names=full,
            subsets=(AGENT_STATE_FEATURES, MARKET_STATE_FEATURES, full),
            ranges=FEATURE_RANGES,
            lambda_weights=self.cfg.lambda_weights,
            num_tilings=self.cfg.num_tilings,
            tiles_per_dim=self.cfg.tiles_per_dim,
            iht_size=self.cfg.iht_size,
        )
        # Tile-coding step size is scaled by the number of active tilings.
        self.alpha = self.cfg.learning_rate / self.cfg.num_tilings
        self.gamma = self.cfg.gamma
        self.trace_lambda = self.cfg.trace_lambda
        self.epsilon = self.cfg.epsilon
        self.rho = 0.0  # R-learning average-reward estimate
        self._episode = 0
        # Cache of the most recent q_all(state) from act(), reused as the next-state
        # bootstrap in update() to avoid recomputing tiles for the same state.
        self._cache_state: np.ndarray | None = None
        self._cache_q: np.ndarray | None = None

    # -- policy ---------------------------------------------------------------
    def act(self, state: np.ndarray, *, greedy: bool = False) -> tuple[int, bool]:
        """Return (action, was_greedy). ``greedy=True`` forces exploitation (evaluation)."""
        explore = (not greedy) and (self.rng.random() < self.epsilon)
        q = self.value.q_all(state, self.n_actions)
        self._cache_state = state
        self._cache_q = q
        if explore:
            return int(self.rng.integers(self.n_actions)), False
        return int(np.argmax(q)), True

    def _q_all_cached(self, state: np.ndarray) -> np.ndarray:
        if state is self._cache_state and self._cache_q is not None:
            return self._cache_q
        return self.value.q_all(state, self.n_actions)

    def greedy_action(self, state: np.ndarray) -> int:
        return int(np.argmax(self.value.q_all(state, self.n_actions)))

    # -- learning -------------------------------------------------------------
    def start_episode(self) -> None:
        self.value.reset_traces()

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action: int,
        done: bool,
        *,
        action_was_greedy: bool = True,
    ) -> float:
        """One TD(lambda) update. Returns the TD error (for diagnostics)."""
        algo = self.cfg.algorithm
        # Current-state tiles computed once, reused for q(s,a) and the traces.
        active = self.value.active_all(state, action)
        q_sa = self.value.q_from_active(active)

        if done:
            bootstrap = 0.0
        else:
            q_next = self._q_all_cached(next_state)
            if algo == "sarsa":
                bootstrap = float(q_next[next_action])
            elif algo in ("q", "r_learning"):
                bootstrap = float(np.max(q_next))
            else:  # expected_sarsa
                bootstrap = float(np.dot(self._epsilon_greedy_probs(q_next), q_next))

        if algo == "r_learning":
            delta = reward - self.rho - q_sa + bootstrap
        else:
            delta = reward - q_sa + (self.gamma * bootstrap if not done else 0.0)

        # Replacing traces for the current (s, a), then a vectorised weight update + decay.
        self.value.accumulate_replacing(active)
        decay = self.trace_lambda if algo == "r_learning" else self.gamma * self.trace_lambda
        self.value.update_and_decay(self.alpha, delta, decay)

        if algo == "r_learning" and action_was_greedy:
            self.rho += self.cfg.beta * delta

        if done:
            self.value.reset_traces()
        return delta

    def _epsilon_greedy_probs(self, q: np.ndarray) -> np.ndarray:
        n = self.n_actions
        probs = np.full(n, self.epsilon / n, dtype=np.float64)
        probs[int(np.argmax(q))] += 1.0 - self.epsilon
        return probs

    # -- schedules / persistence ---------------------------------------------
    def end_episode(self) -> None:
        self._episode += 1
        frac = min(1.0, self._episode / max(1, self.cfg.epsilon_decay_episodes))
        self.epsilon = self.cfg.epsilon + frac * (self.cfg.epsilon_floor - self.cfg.epsilon)

    def save(self, path: str | Path) -> None:
        """
        Persist the weight vectors, rho and epsilon. Tile indices are computed with a
        deterministic stateless hash, so no tile->index map needs saving — the same
        (state, action) hashes to the same index in any process.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            **{f"w{i}": c.w for i, c in enumerate(self.value.codings)},
            rho=np.array([self.rho]),
            epsilon=np.array([self.epsilon]),
        )

    def load(self, path: str | Path) -> None:
        data = np.load(Path(path))
        for i, c in enumerate(self.value.codings):
            c.w = data[f"w{i}"]
        self.rho = float(data["rho"][0])
        self.epsilon = float(data["epsilon"][0])
