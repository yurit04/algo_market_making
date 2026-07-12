"""
Command-line interface for training and evaluating the RL market-making agent.

Examples
--------
Train a SARSA(lambda) agent with asymmetric-dampened reward on the ES cache::

    algo-mm-rl train --algorithm sarsa --reward asymmetric --eta 0.6 \
        --episodes 200 --cadence-ms 500 --cache-dir outputs/rl_cache/mbo

Evaluate a checkpoint against fixed and random benchmarks on the held-out split::

    algo-mm-rl evaluate --checkpoint outputs/rl_models/sarsa_final.npz \
        --cache-dir outputs/rl_cache/mbo

Both subcommands call the same ``train`` / ``evaluate_policy`` functions used by the
notebook, so behaviour is identical across entry points.
"""

from __future__ import annotations

import argparse

import numpy as np

from algo_mm.rl.agents import TDAgent
from algo_mm.rl.config import AgentConfig, EnvConfig, FeatureConfig, TrainConfig
from algo_mm.rl.env import STATE_FEATURE_NAMES, MarketMakingEnv
from algo_mm.rl.evaluate import FixedPolicy, RandomPolicy, evaluate_policy
from algo_mm.rl.train import chronological_split, discover_sessions, train


def _env_cfg(args) -> EnvConfig:
    # A fixed decision cadence: set both the min and max inter-decision interval so the
    # agent re-quotes on a regular grid (keeps episode length — and training cost —
    # bounded, vs. re-quoting on every top-of-book change).
    cadence_ns = int(args.cadence_ms * 1_000_000)
    return EnvConfig(
        tick_size=args.tick_size,
        order_size=args.order_size,
        max_inventory=args.max_inventory,
        min_inventory=-args.max_inventory,
        reward=args.reward,
        eta=args.eta,
        min_interval_ns=cadence_ns,
        max_interval_ns=cadence_ns,
    )


def _add_env_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cache-dir", default="outputs/rl_cache/mbo")
    p.add_argument("--symbol", default=None, help="Restrict to one symbol subdir")
    p.add_argument("--tick-size", type=float, default=0.25)
    p.add_argument("--order-size", type=int, default=1)
    p.add_argument("--max-inventory", type=int, default=50)
    p.add_argument("--reward", default="asymmetric", choices=["pnl", "symmetric", "asymmetric"])
    p.add_argument("--eta", type=float, default=0.6)
    p.add_argument("--cadence-ms", type=float, default=500.0, help="Max re-quote interval (ms)")
    p.add_argument("--queue-model", default="exact", choices=["exact", "uniform"])


def _cmd_train(args) -> None:
    env_cfg = _env_cfg(args)
    agent_cfg = AgentConfig(
        algorithm=args.algorithm,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        trace_lambda=args.trace_lambda,
        epsilon=args.epsilon,
        epsilon_decay_episodes=args.episodes,
    )
    train_cfg = TrainConfig(
        cache_dir=args.cache_dir,
        symbol=args.symbol,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        test_fraction=args.test_fraction,
        queue_model=args.queue_model,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )
    result = train(env_cfg, agent_cfg, train_cfg)
    print(f"\nTrained {args.algorithm} for {args.episodes} episodes.")
    print(f"Checkpoint: {result.checkpoint}")
    print(f"Train sessions: {len(result.train_sessions)}  Test sessions: {len(result.test_sessions)}")


def _cmd_evaluate(args) -> None:
    env_cfg = _env_cfg(args)
    paths = discover_sessions(args.cache_dir, args.symbol)
    _, test_paths = chronological_split(paths, args.test_fraction)
    rng = np.random.default_rng(args.seed)

    def agent_factory(env: MarketMakingEnv):
        agent = TDAgent(STATE_FEATURE_NAMES, MarketMakingEnv.n_actions,
                        AgentConfig(algorithm=args.algorithm))
        agent.load(args.checkpoint)
        return agent.greedy_action

    strategies = {
        f"agent ({args.algorithm})": agent_factory,
        "fixed(theta=1)": lambda env: FixedPolicy(1),
        "fixed(theta=3)": lambda env: FixedPolicy(3),
        "fixed(theta=5)": lambda env: FixedPolicy(5),
        "random": lambda env: RandomPolicy(env.n_actions, rng),
    }

    print(f"Evaluating on {len(test_paths)} held-out sessions "
          f"(queue={args.queue_model}, cadence={args.cadence_ms}ms)\n")
    header = f"{'strategy':<20} {'ND-PnL':>10} {'±std':>8} {'MAP':>8} {'fills':>8} {'PnL(pts)':>10}"
    print(header)
    print("-" * len(header))
    for name, factory in strategies.items():
        m = evaluate_policy(test_paths, env_cfg, factory,
                            queue_model=args.queue_model, max_steps=args.max_steps)
        print(f"{name:<20} {m['nd_pnl']:>10.2f} {m['nd_pnl_std']:>8.2f} "
              f"{m['map']:>8.1f} {m['n_fills']:>8.0f} {m['final_pnl']:>10.2f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="algo-mm-rl", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train", help="Train an agent")
    _add_env_args(pt)
    pt.add_argument("--algorithm", default="sarsa",
                    choices=["sarsa", "q", "expected_sarsa", "r_learning"])
    pt.add_argument("--episodes", type=int, default=200)
    pt.add_argument("--max-steps-per-episode", type=int, default=None,
                    help="Cap decision epochs per episode (default: full session)")
    pt.add_argument("--learning-rate", type=float, default=0.001)
    pt.add_argument("--gamma", type=float, default=0.97)
    pt.add_argument("--trace-lambda", type=float, default=0.96)
    pt.add_argument("--epsilon", type=float, default=0.7)
    pt.add_argument("--test-fraction", type=float, default=0.25)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--checkpoint-dir", default="outputs/rl_models")
    pt.set_defaults(func=_cmd_train)

    pe = sub.add_parser("evaluate", help="Evaluate a checkpoint vs. benchmarks")
    _add_env_args(pe)
    pe.add_argument("--algorithm", default="sarsa",
                    choices=["sarsa", "q", "expected_sarsa", "r_learning"])
    pe.add_argument("--checkpoint", required=True)
    pe.add_argument("--test-fraction", type=float, default=0.25)
    pe.add_argument("--seed", type=int, default=0)
    pe.add_argument("--max-steps", type=int, default=None)
    pe.set_defaults(func=_cmd_evaluate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
