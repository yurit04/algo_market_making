from dataclasses import dataclass

import numpy as np


@dataclass
class Metrics:
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    final_pnl: float


def compute_metrics(pnl: np.ndarray) -> Metrics:
    n_steps = len(pnl) - 1
    returns = np.diff(pnl)

    mu = float(np.mean(returns) * n_steps)
    vol = float(np.std(returns) * np.sqrt(n_steps))
    sharpe = mu / vol if vol != 0 else 0.0

    downside = returns[returns < 0]
    downside_std = float(np.std(downside) * np.sqrt(n_steps)) if len(downside) > 0 else 1e-10
    sortino = mu / downside_std if downside_std != 0 else 0.0

    running_max = np.maximum.accumulate(pnl)
    max_dd = float(np.max(running_max - pnl))
    calmar = mu / max_dd if max_dd != 0 else 0.0

    return Metrics(
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        final_pnl=float(pnl[-1]),
    )
