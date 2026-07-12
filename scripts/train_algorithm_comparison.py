"""
Train the TD control algorithms studied by Spooner et al. (2018, §4.4 / Table 5) and
evaluate each on the held-out split, for a like-for-like comparison table.

All agents use the same trading regime as the notebook's main agent — the undamped PnL
reward (asymmetric with eta=0), so every algorithm actively market-makes and differences
reflect the learning rule, not the reward. Off-policy methods (Q-learning, R-learning)
are expected to be less stable than on-policy SARSA, per the paper.

Each agent's checkpoint + history is written under ``outputs/rl_models/algos/<algo>/``;
a combined ``algos_summary.csv`` holds train- and test-set metrics. Run standalone:

    python scripts/train_algorithm_comparison.py --episodes 200 --cadence-ms 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from algo_mm.rl.config import AgentConfig, EnvConfig, TrainConfig
from algo_mm.rl.evaluate import evaluate_policy
from algo_mm.rl.train import chronological_split, discover_sessions, train

ALGORITHMS = ["sarsa", "q", "expected_sarsa", "r_learning"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithms", nargs="+", default=ALGORITHMS)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--cadence-ms", type=float, default=1000.0)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--max-inventory", type=int, default=50)
    p.add_argument("--eval-max-steps", type=int, default=4000)
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--cache-dir", default="outputs/rl_cache/mbo")
    p.add_argument("--out", default="outputs/rl_models/algos")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cadence = int(args.cadence_ms * 1_000_000)
    # Undamped PnL trading regime, shared by every algorithm.
    env_cfg = EnvConfig(
        order_size=1, max_inventory=args.max_inventory, min_inventory=-args.max_inventory,
        reward="asymmetric", eta=0.0, min_interval_ns=cadence, max_interval_ns=cadence,
    )
    paths = discover_sessions(args.cache_dir)
    _, test_paths = chronological_split(paths, args.test_fraction)

    summary = []
    for algo in args.algorithms:
        out_dir = Path(args.out) / algo
        train_cfg = TrainConfig(
            cache_dir=args.cache_dir, episodes=args.episodes, max_steps_per_episode=args.max_steps,
            test_fraction=args.test_fraction, checkpoint_dir=str(out_dir),
            checkpoint_every=0, seed=args.seed,
        )
        agent_cfg = AgentConfig(algorithm=algo, epsilon_decay_episodes=args.episodes)
        print(f"\n=== training {algo} ({args.episodes} episodes) ===", flush=True)
        result = train(env_cfg, agent_cfg, train_cfg, progress=True)

        trained = result.agent
        m = evaluate_policy(
            test_paths, env_cfg, lambda e: trained.greedy_action,
            queue_model="exact", max_steps=args.eval_max_steps,
        )
        last = result.history_frame().tail(30)
        summary.append({
            "algorithm": algo,
            "test_nd_pnl": m["nd_pnl"],
            "test_nd_pnl_std": m["nd_pnl_std"],
            "test_map": m["map"],
            "test_fills": m["n_fills"],
            "train_reward_last30": last["total_reward"].mean(),
            "train_map_last30": last["mean_abs_position"].mean(),
            "checkpoint": result.checkpoint,
        })
        print(f"  {algo}: test ND-PnL={m['nd_pnl']:.2f}±{m['nd_pnl_std']:.2f} "
              f"MAP={m['map']:.2f} fills={m['n_fills']:.0f}", flush=True)

    df = pd.DataFrame(summary)
    out = Path(args.out) / "algos_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\n=== algorithm comparison (test-set greedy, train-set last-30 means) ===")
    print(df.drop(columns=["checkpoint"]).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
