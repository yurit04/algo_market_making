"""
Training loop for the tile-coding TD market-making agent.

An episode is one cached trading session, sampled (with replacement) from a
chronologically earlier training split; the later split is held out for evaluation
(paper §5: all test data occurs after the training data). The same ``train`` function is
called from the CLI and from the notebook, so results are identical either way.

Learning is online SARSA(lambda) / Q(lambda) / Expected-SARSA / R-learning: at each
decision epoch the agent picks an action, the environment returns the paper's reward,
and the agent applies a TD(lambda) update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from algo_mm.rl.agents import TDAgent
from algo_mm.rl.config import AgentConfig, EnvConfig, TrainConfig
from algo_mm.rl.env import STATE_FEATURE_NAMES, MarketMakingEnv
from algo_mm.rl.evaluate import evaluate_policy, make_queue
from algo_mm.rl.replay import CachedSession


@dataclass
class EpisodeLog:
    episode: int
    session: str
    total_reward: float
    mean_reward: float
    final_pnl: float
    mean_abs_position: float
    n_fills: int
    steps: int
    epsilon: float


@dataclass
class TrainResult:
    agent: TDAgent
    history: list[EpisodeLog] = field(default_factory=list)
    train_sessions: list[str] = field(default_factory=list)
    test_sessions: list[str] = field(default_factory=list)
    checkpoint: str | None = None
    history_csv: str | None = None

    def history_frame(self):
        import pandas as pd
        return pd.DataFrame([vars(h) for h in self.history])


def discover_sessions(cache_dir: str | Path, symbol: str | None = None) -> list[Path]:
    """Return cached session parquet paths, sorted chronologically by (date, symbol)."""
    base = Path(cache_dir)
    pattern = f"{symbol}/*.parquet" if symbol else "*/*.parquet"
    paths = list(base.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No cached sessions under {base} (pattern {pattern!r})")
    # File stem is the ISO date; sort by date then symbol for a stable chronology.
    return sorted(paths, key=lambda p: (p.stem, p.parent.name))


def chronological_split(paths: list[Path], test_fraction: float) -> tuple[list[Path], list[Path]]:
    n_test = max(1, int(round(len(paths) * test_fraction)))
    n_train = max(1, len(paths) - n_test)
    return paths[:n_train], paths[n_train:]


def run_training_episode(env: MarketMakingEnv, agent: TDAgent,
                         max_steps: int | None = None) -> dict:
    """One online TD(lambda) episode. Returns episode diagnostics."""
    state = env.reset()
    action, greedy = agent.act(state)
    agent.start_episode()

    total_reward = 0.0
    inventory_abs_sum = 0.0
    n_fills = 0
    steps = 0
    final_equity = 0.0

    while not env.done:
        next_state, reward, done, info = env.step(action)
        if max_steps is not None and steps + 1 >= max_steps:
            done = True
        next_action, next_greedy = agent.act(next_state)
        agent.update(state, action, reward, next_state, next_action, done,
                     action_was_greedy=greedy)
        state, action, greedy = next_state, next_action, next_greedy

        total_reward += reward
        inventory_abs_sum += abs(info.inventory)
        n_fills += info.n_fills
        final_equity = info.equity
        steps += 1
        if done:
            break

    return {
        "total_reward": total_reward,
        "mean_reward": total_reward / steps if steps else 0.0,
        "final_pnl": final_equity,
        "mean_abs_position": inventory_abs_sum / steps if steps else 0.0,
        "n_fills": n_fills,
        "steps": steps,
    }


def train(
    env_cfg: EnvConfig | None = None,
    agent_cfg: AgentConfig | None = None,
    train_cfg: TrainConfig | None = None,
    *,
    progress: bool = True,
) -> TrainResult:
    """Train an agent and return it with its per-episode history and data split."""
    env_cfg = env_cfg or EnvConfig()
    agent_cfg = agent_cfg or AgentConfig()
    train_cfg = train_cfg or TrainConfig()

    paths = discover_sessions(train_cfg.cache_dir, train_cfg.symbol)
    train_paths, test_paths = chronological_split(paths, train_cfg.test_fraction)
    rng = np.random.default_rng(train_cfg.seed)

    agent = TDAgent(STATE_FEATURE_NAMES, MarketMakingEnv.n_actions, agent_cfg, rng=rng)
    result = TrainResult(
        agent=agent,
        train_sessions=[str(p) for p in train_paths],
        test_sessions=[str(p) for p in test_paths],
    )

    ckpt_dir = Path(train_cfg.checkpoint_dir)
    for ep in range(train_cfg.episodes):
        path = train_paths[int(rng.integers(len(train_paths)))]
        session = CachedSession.from_parquet(path)
        env = MarketMakingEnv(session, env_cfg, queue_model=make_queue(train_cfg.queue_model))

        metrics = run_training_episode(env, agent, train_cfg.max_steps_per_episode)
        agent.end_episode()
        log = EpisodeLog(episode=ep, session=path.stem, epsilon=agent.epsilon, **metrics)
        result.history.append(log)

        if progress:
            print(
                f"ep {ep:4d} [{path.parent.name} {path.stem}] "
                f"R={log.total_reward:9.2f} pnl={log.final_pnl:9.2f} "
                f"MAP={log.mean_abs_position:6.1f} fills={log.n_fills:5d} "
                f"eps={log.epsilon:.3f}",
                flush=True,
            )

        if train_cfg.checkpoint_every and (ep + 1) % train_cfg.checkpoint_every == 0:
            path_ck = ckpt_dir / f"{agent_cfg.algorithm}_ep{ep + 1}.npz"
            agent.save(path_ck)
            result.checkpoint = str(path_ck)

    final_ck = ckpt_dir / f"{agent_cfg.algorithm}_final.npz"
    agent.save(final_ck)
    result.checkpoint = str(final_ck)

    # Persist the per-episode history so learning curves can be reloaded later.
    try:
        import pandas as pd
        hist_path = ckpt_dir / f"{agent_cfg.algorithm}_{env_cfg.reward}_history.csv"
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([vars(h) for h in result.history]).to_csv(hist_path, index=False)
        result.history_csv = str(hist_path)
    except Exception:  # pragma: no cover - history persistence is best-effort
        pass
    return result
