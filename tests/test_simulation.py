import numpy as np
import pytest

from algo_mm.models.naive import NaiveMarketMaker
from algo_mm.simulation.engine import SimResult, run_simulation
from algo_mm.simulation.monte_carlo import run_monte_carlo


@pytest.fixture
def naive_model():
    return NaiveMarketMaker(half_spread=0.1, lambda_buy=100.0, lambda_sell=100.0)


class TestRunSimulation:
    def test_output_shape(self, naive_model):
        result = run_simulation(naive_model, sigma=2.0, T=1.0, dt=0.01, seed=0)
        n = int(1.0 / 0.01)
        for arr in (result.time, result.midprice, result.bid_quotes, result.ask_quotes,
                    result.inventory, result.pnl):
            assert len(arr) == n

    def test_deterministic_with_seed(self, naive_model):
        r1 = run_simulation(naive_model, sigma=2.0, T=1.0, dt=0.01, seed=42)
        r2 = run_simulation(naive_model, sigma=2.0, T=1.0, dt=0.01, seed=42)
        np.testing.assert_array_equal(r1.pnl, r2.pnl)
        np.testing.assert_array_equal(r1.midprice, r2.midprice)

    def test_different_seeds_differ(self, naive_model):
        r1 = run_simulation(naive_model, sigma=2.0, T=1.0, dt=0.01, seed=1)
        r2 = run_simulation(naive_model, sigma=2.0, T=1.0, dt=0.01, seed=2)
        assert not np.array_equal(r1.pnl, r2.pnl)

    def test_quotes_straddle_mid(self, naive_model):
        result = run_simulation(naive_model, sigma=2.0, T=1.0, dt=0.01, seed=0)
        assert np.all(result.bid_quotes < result.midprice)
        assert np.all(result.ask_quotes > result.midprice)

    def test_returns_sim_result_type(self, naive_model):
        result = run_simulation(naive_model, sigma=2.0, T=1.0, dt=0.01, seed=0)
        assert isinstance(result, SimResult)


class TestMonteCarlo:
    def test_returns_correct_count(self, naive_model):
        results = run_monte_carlo(naive_model, sigma=2.0, n_sims=10, T=0.1, dt=0.01)
        assert len(results) == 10

    def test_metrics_finite(self, naive_model):
        results = run_monte_carlo(naive_model, sigma=2.0, n_sims=20, T=0.1, dt=0.01)
        for m in results:
            assert np.isfinite(m.sharpe)
            assert np.isfinite(m.sortino)
            assert np.isfinite(m.max_drawdown)
