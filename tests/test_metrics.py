import numpy as np
import pytest

from algo_mm.metrics.performance import Metrics, compute_metrics


def test_returns_metrics_dataclass():
    pnl = np.random.default_rng(0).standard_normal(101).cumsum()
    m = compute_metrics(pnl)
    assert isinstance(m, Metrics)


def test_max_drawdown_zero_for_monotonic_pnl():
    pnl = np.arange(201, dtype=float)
    m = compute_metrics(pnl)
    assert m.max_drawdown == 0.0


def test_max_drawdown_nonnegative():
    rng = np.random.default_rng(42)
    pnl = rng.standard_normal(201).cumsum()
    m = compute_metrics(pnl)
    assert m.max_drawdown >= 0.0


def test_final_pnl_matches_last_element():
    pnl = np.array([0.0, 1.0, 3.0, 2.0, 5.0])
    m = compute_metrics(pnl)
    assert m.final_pnl == 5.0


def test_sharpe_positive_for_positive_mean_returns():
    rng = np.random.default_rng(7)
    returns = rng.normal(loc=1.0, scale=0.3, size=200)
    pnl = np.concatenate([[0.0], np.cumsum(returns)])
    m = compute_metrics(pnl)
    assert m.sharpe > 0


def test_calmar_positive_when_pnl_grows_with_drawdown():
    # Trending up with a dip → positive calmar
    pnl = np.array([0.0, 1.0, 2.0, 1.5, 3.0, 4.0])
    m = compute_metrics(pnl)
    assert m.calmar > 0
