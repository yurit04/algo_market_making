"""
Train SARSA(lambda) agents across the asymmetric-damping factor eta, reproducing the
central experiment of Spooner et al. (2018, Fig. 2): as eta increases, the agent's mean
absolute position (inventory) falls from large speculative holdings toward zero.

Asymmetric reward with eta=0 is exactly the undamped PnL reward, so the sweep spans the
"basic" speculative agent (eta=0) through the strongly inventory-averse agent (eta=0.6).

Each agent's checkpoint and per-episode history are written under
``outputs/rl_models/sweep/eta<eta>/``. Run standalone (it is not imported):

    python scripts/train_damping_sweep.py --episodes 200 --cadence-ms 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from algo_mm.rl.config import AgentConfig, EnvConfig, TrainConfig
from algo_mm.rl.train import train


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--etas", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.6])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--cadence-ms", type=float, default=1000.0)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--max-inventory", type=int, default=50)
    p.add_argument("--cache-dir", default="outputs/rl_cache/mbo")
    p.add_argument("--out", default="outputs/rl_models/sweep")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cadence = int(args.cadence_ms * 1_000_000)
    summary = []
    for eta in args.etas:
        env_cfg = EnvConfig(
            order_size=1, max_inventory=args.max_inventory, min_inventory=-args.max_inventory,
            reward="asymmetric", eta=eta, min_interval_ns=cadence, max_interval_ns=cadence,
        )
        agent_cfg = AgentConfig(algorithm="sarsa", epsilon_decay_episodes=args.episodes)
        out_dir = Path(args.out) / f"eta{eta}"
        train_cfg = TrainConfig(
            cache_dir=args.cache_dir, episodes=args.episodes, max_steps_per_episode=args.max_steps,
            checkpoint_dir=str(out_dir), checkpoint_every=0, seed=args.seed,
        )
        print(f"\n=== training eta={eta} ({args.episodes} episodes) ===", flush=True)
        result = train(env_cfg, agent_cfg, train_cfg, progress=True)
        hist = result.history_frame()
        last = hist.tail(30)
        summary.append({
            "eta": eta,
            "final_reward": last["total_reward"].mean(),
            "final_MAP": last["mean_abs_position"].mean(),
            "final_pnl": last["final_pnl"].mean(),
            "final_fills": last["n_fills"].mean(),
            "checkpoint": result.checkpoint,
        })

    df = pd.DataFrame(summary)
    out = Path(args.out) / "sweep_summary.csv"
    df.to_csv(out, index=False)
    print("\n=== damping sweep summary (last-30-episode means) ===")
    print(df.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
