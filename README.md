# algo_market_making

A Python library and research workspace for simulating and comparing algorithmic market-making strategies. The `algo_mm` package implements several quoting models on a shared interface, runs single-path and Monte Carlo simulations against a Brownian mid-price, and provides performance metrics and plotting utilities. Jupyter notebooks walk through each model in detail, from a naive fixed-spread baseline through inventory-aware quoting, the Avellaneda–Stoikov optimal market maker, and bond market making with a hit-ratio target.

## Directory structure

```
algo_market_making/
├── algo_mm/                    # Installable Python package
│   ├── models/                 # Market-making strategies
│   │   ├── base.py             # Abstract MarketMaker interface
│   │   ├── naive.py            # Fixed symmetric spread
│   │   ├── inventory_adjusted.py
│   │   └── avellaneda_stoikov.py
│   ├── simulation/
│   │   ├── engine.py           # Single-path simulation
│   │   └── monte_carlo.py      # Multi-path batch runs
│   ├── metrics/
│   │   └── performance.py      # Sharpe, Sortino, drawdown, etc.
│   └── visualization/
│       ├── simulation.py       # Time-series plots
│       └── distributions.py    # Monte Carlo distribution plots
├── notebooks/                  # Exploratory analysis and paper reproductions
│   ├── naive_market_making.ipynb
│   ├── inventory_adjusted_market_making.ipynb
│   ├── avellaneda_stoikov.ipynb
│   └── bond_market_making_hit_ratio.ipynb
├── tests/                      # pytest suite
├── pyproject.toml
├── LICENSE
└── README.md
```

## Installation

Requires **Python 3.9+**.

### 1. Clone the repository

```bash
git clone https://github.com/yuriturygin/algo_market_making.git
cd algo_market_making
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install the package

Install the library in editable mode so local changes are picked up immediately:

```bash
pip install -e .
```

For development (tests, Jupyter notebooks):

```bash
pip install -e ".[dev]"
```

### 4. Verify the install

```bash
pytest
```

## Quick example

```python
from algo_mm import AvellanedaStoikovMarketMaker, run_simulation, compute_metrics, plot_simulation

model = AvellanedaStoikovMarketMaker(sigma=2.0, gamma=0.1, kappa=1.5, A=100.0)
result = run_simulation(model, sigma=2.0, seed=42)
print(compute_metrics(result.pnl))
plot_simulation(result)
```

## Notebooks

Open any notebook under `notebooks/` after installing with the `dev` extra:

```bash
jupyter notebook
```

| Notebook | Topic |
|----------|--------|
| `naive_market_making.ipynb` | Fixed-spread baseline with Poisson fills |
| `inventory_adjusted_market_making.ipynb` | Spread skew from inventory |
| `avellaneda_stoikov.ipynb` | Avellaneda–Stoikov (2008) optimal quotes |
| `bond_market_making_hit_ratio.ipynb` | Bond MM with a target hit ratio |

## License

MIT — see [LICENSE](LICENSE).
