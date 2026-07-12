"""
Configuration for the market-making environment (Phase 2).

Defaults mirror Spooner et al. (2018) Table 2 and the trading strategy in §4.1 where
they carry over to CME futures. Two deliberate deviations from the paper's equities
setup are called out inline: ``order_size`` and inventory bounds are scaled from
thousands of shares to a handful of futures contracts so the no-market-impact
assumption stays valid on ES.

Learning-algorithm hyperparameters (tile codings, learning rate, traces, ...) belong
to the RL agent and are added in Phase 3; this config covers only the environment and
the trading strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Paper Table 1 action space: nine (ask, bid) quoting-distance pairs; action 9 is a
# market order that clears inventory. Distances are multiples of the spread scale.
ACTION_THETA_ASK: tuple[int, ...] = (1, 2, 3, 4, 5, 1, 3, 2, 5)
ACTION_THETA_BID: tuple[int, ...] = (1, 2, 3, 4, 5, 3, 1, 5, 2)
CLEAR_ACTION_ID: int = 9
N_ACTIONS: int = 10


@dataclass
class FeatureConfig:
    """Lookback parameters for the market-state features (paper §4.3)."""

    imbalance_levels: int = 5          # top-k book levels for queue imbalance
    volatility_lookback: int = 50      # decision steps for mid-return volatility
    rsi_lookback: int = 14             # decision steps for RSI (standard default)
    signed_volume_lookback: int = 50   # decision steps for rolling signed trade volume


# Expected (lo, hi) range of each state feature, used to scale inputs into tile widths.
# Keyed by the names in env.STATE_FEATURE_NAMES. Chosen from typical ES ranges; the
# paper GA-tunes these, so they are the natural thing to sweep later.
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "spread_ticks": (1.0, 8.0),
    "dmid_ticks": (-4.0, 4.0),
    "book_imbalance": (-1.0, 1.0),
    "signed_volume": (-500.0, 500.0),
    "volatility": (0.0, 3.0),
    "rsi": (0.0, 100.0),
    "inventory_norm": (-1.0, 1.0),
    "theta_ask_norm": (0.0, 1.0),
    "theta_bid_norm": (0.0, 1.0),
}

# Feature-name membership of each independent tile coding in the LCTC (paper §4.3).
AGENT_STATE_FEATURES: tuple[str, ...] = ("inventory_norm", "theta_ask_norm", "theta_bid_norm")
MARKET_STATE_FEATURES: tuple[str, ...] = (
    "spread_ticks", "dmid_ticks", "book_imbalance", "signed_volume", "volatility", "rsi",
)


@dataclass
class AgentConfig:
    """Learning-algorithm hyperparameters (paper Table 2 defaults)."""

    algorithm: str = "sarsa"           # sarsa | q | expected_sarsa | r_learning
    num_tilings: int = 32              # M
    tiles_per_dim: int = 8             # resolution of each tiling per feature
    iht_size: int = 1 << 22            # hash size per coding (~4.2M; avoids collisions at 6–9 dims)
    lambda_weights: tuple[float, float, float] = (0.6, 0.1, 0.3)  # (agent, market, full)
    learning_rate: float = 0.001       # alpha (scaled by num_tilings internally)
    beta: float = 0.005                # R-learning average-reward step size
    gamma: float = 0.97                # discount factor
    trace_lambda: float = 0.96         # eligibility-trace decay
    epsilon: float = 0.7               # initial exploration rate
    epsilon_floor: float = 0.0001
    epsilon_decay_episodes: int = 1000  # episodes over which epsilon decays to the floor


@dataclass
class TrainConfig:
    """Training-run parameters (data selection, episode budget, checkpointing)."""

    cache_dir: str = "outputs/rl_cache/mbo"
    symbol: str | None = None          # restrict to one symbol subdir; None = all
    episodes: int = 200
    max_steps_per_episode: int | None = None  # cap decision epochs per episode (None = full session)
    test_fraction: float = 0.25        # last fraction of sessions (chronological) held out
    queue_model: str = "exact"         # exact | uniform
    seed: int = 0
    checkpoint_dir: str = "outputs/rl_models"
    checkpoint_every: int = 50
    eval_every: int = 0                # 0 = no periodic validation during training


@dataclass
class EnvConfig:
    """Environment + trading-strategy parameters."""

    tick_size: float = 0.25            # ES tick (index points)
    order_size: int = 1                # contracts per quote (paper: 1000 shares)
    max_inventory: int = 50            # contracts (paper: 10000 shares)
    min_inventory: int = -50
    clear_alpha: float = 1.0           # action-9 market order clears alpha * inventory

    # Spread scale factor: a moving average of the market half-spread, rounded to a
    # whole number of ticks (>= 1). Quotes sit theta * spread_scale from the mid.
    spread_ma_window: int = 100        # decision steps in the half-spread moving average

    # Decision cadence (event-driven, throttled). A decision is taken at the first
    # book-consistent boundary where top-of-book changed and >= min_interval_ns have
    # elapsed, or unconditionally after max_interval_ns.
    min_interval_ns: int = 0
    max_interval_ns: int = 1_000_000_000  # force a re-quote at least every 1s

    # Reward (paper Eqs. 4/5/6): "pnl" | "symmetric" | "asymmetric"; eta is the damping.
    reward: str = "asymmetric"
    eta: float = 0.6

    features: FeatureConfig = field(default_factory=FeatureConfig)

    def __post_init__(self) -> None:
        if self.reward not in ("pnl", "symmetric", "asymmetric"):
            raise ValueError(f"reward must be pnl|symmetric|asymmetric, got {self.reward!r}")
        if self.max_inventory <= 0 or self.min_inventory >= 0:
            raise ValueError("max_inventory must be > 0 and min_inventory < 0")
