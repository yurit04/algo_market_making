from ..metrics.performance import Metrics, compute_metrics
from ..models.base import MarketMaker
from .engine import SimResult, run_simulation


def run_monte_carlo(
    model: MarketMaker,
    sigma: float,
    n_sims: int = 1000,
    S0: float = 100.0,
    T: float = 1.0,
    dt: float = 0.005,
) -> list[Metrics]:
    """Run n_sims independent simulations and return per-simulation metrics."""
    results = []
    for _ in range(n_sims):
        sim: SimResult = run_simulation(model, sigma, S0=S0, T=T, dt=dt)
        results.append(compute_metrics(sim.pnl))
    return results
