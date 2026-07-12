"""
Reinforcement-learning market making on order-by-order (MBO) data.

Adapts the methodology of Spooner, Fearnley, Savani & Koukorinis, *Market Making via
Reinforcement Learning* (AAMAS 2018) to Databento CME MBO data. Phase 1 provides the
data substrate: L3 order-book reconstruction, exact FIFO queue tracking (plus the
paper's uniform-cancel approximation for comparison), and session replay.
"""

from algo_mm.rl.agent_orders import (
    AgentOrder,
    AgentOrderBook,
    ExactFIFOQueue,
    Fill,
    QueueModel,
    UniformCancelQueue,
)
from algo_mm.rl.agents import TDAgent
from algo_mm.rl.book import ASK, BID, OrderBook, PRICE_SCALE, build_book
from algo_mm.rl.config import AgentConfig, EnvConfig, FeatureConfig, N_ACTIONS, TrainConfig
from algo_mm.rl.env import MarketMakingEnv, STATE_FEATURE_NAMES, StepInfo
from algo_mm.rl.evaluate import (
    FixedPolicy,
    RandomPolicy,
    evaluate_policy,
    make_queue,
    run_episode,
)
from algo_mm.rl.features import MARKET_FEATURE_NAMES, FeatureExtractor
from algo_mm.rl.replay import CachedSession, MarketReplay, Trade, replay_session
from algo_mm.rl.rewards import RewardBreakdown, compute_reward
from algo_mm.rl.tilecoding import IHT, LCTC
from algo_mm.rl.train import TrainResult, discover_sessions, train

__all__ = [
    "AgentOrder",
    "AgentOrderBook",
    "ExactFIFOQueue",
    "Fill",
    "QueueModel",
    "UniformCancelQueue",
    "ASK",
    "BID",
    "OrderBook",
    "PRICE_SCALE",
    "build_book",
    "CachedSession",
    "MarketReplay",
    "Trade",
    "replay_session",
    "EnvConfig",
    "FeatureConfig",
    "N_ACTIONS",
    "MarketMakingEnv",
    "STATE_FEATURE_NAMES",
    "StepInfo",
    "MARKET_FEATURE_NAMES",
    "FeatureExtractor",
    "RewardBreakdown",
    "compute_reward",
    "TDAgent",
    "AgentConfig",
    "TrainConfig",
    "IHT",
    "LCTC",
    "FixedPolicy",
    "RandomPolicy",
    "evaluate_policy",
    "run_episode",
    "make_queue",
    "TrainResult",
    "discover_sessions",
    "train",
]
