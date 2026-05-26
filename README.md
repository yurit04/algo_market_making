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
│   ├── visualization/
│   │   ├── simulation.py       # Time-series plots
│   │   └── distributions.py    # Monte Carlo distribution plots
│   └── data/                   # Local Databento DBN ingestion (optional [data] extra)
│       └── databento/
├── notebooks/                  # Source notebooks (no generated files)
│   ├── naive_market_making.ipynb
│   ├── inventory_adjusted_market_making.ipynb
│   ├── avellaneda_stoikov.ipynb
│   └── bond_market_making_hit_ratio.ipynb
├── references/                 # Source papers (PDFs)
│   └── papers/
│       └── bond-market-making-hit-ratio.pdf
├── outputs/                    # Generated artifacts (gitignored except .gitkeep)
│   └── notebooks/
│       └── bond_market_making_hit_ratio/   # figures from bond notebook
├── tests/                      # pytest suite
├── pyproject.toml
├── LICENSE
└── README.md
```

## Installation

Requires **Python 3.12** and [uv](https://docs.astral.sh/uv/getting-started/installation/).

### 1. Clone the repository

```bash
git clone https://github.com/yuriturygin/algo_market_making.git
cd algo_market_making
```

### 2. Install uv (if needed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Create a virtual environment and install dependencies

Create a local `.venv` with Python 3.12 and install the package in editable mode (includes dev tools for tests and notebooks):

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Activate the environment when working in a shell (optional — `uv run` works without activation):

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 4. Verify the install

```bash
uv run pytest
```

## Quick example

```python
from algo_mm import AvellanedaStoikovMarketMaker, run_simulation, compute_metrics, plot_simulation

model = AvellanedaStoikovMarketMaker(sigma=2.0, gamma=0.1, kappa=1.5, A=100.0)
result = run_simulation(model, sigma=2.0, seed=42)
print(compute_metrics(result.pnl))
plot_simulation(result)
```

## Databento CME data (local)

If you keep [Databento](https://databento.com) batch downloads under `~/Documents/Databento/CME-Futures`, install the `data` extra and use `algo_mm.data`:

```bash
uv pip install -e ".[data]"
```

```python
from algo_mm.data import CMEDataCatalog, load, iter_load

catalog = CMEDataCatalog()  # or CMEDataCatalog("/path/to/CME-Futures")
print(catalog.list_schemas())
print(catalog.describe())

parents = list_parent_symbols("ohlcv-1m")  # e.g. ES.FUT, NQ.FUT, ...
contracts = list_contracts("ohlcv-1m", parent="ES.FUT")  # all ES outrights in symbology

df = load("ohlcv-1m", symbols="ES.FUT", start="2024-06-01", end="2024-06-02")

# MBO / MBP shards: one DataFrame per .dbn file
for chunk in iter_load("mbo", start="2025-06-08", end="2025-06-09", symbols="ES.FUT"):
    ...
```

Set `DATABENTO_DATA_ROOT` to override the default path. See `notebooks/databento_cme_ingestion.ipynb`.

| Local folder | Databento schema |
|--------------|------------------|
| `MBO` | `mbo` |
| `MBO-10` | `mbp-10` |
| `TTBO` | `tbbo` |
| `OHLCV-1s` | `ohlcv-1s` |
| `OHLCV-1m` / `1h` / `1d` | `ohlcv-1m` / `1h` / `1d` |

## Notebooks

Open any notebook under `notebooks/` after installing with the `dev` extra:

```bash
uv run jupyter notebook
```

| Notebook | Topic |
|----------|--------|
| `naive_market_making.ipynb` | Fixed-spread baseline with Poisson fills |
| `inventory_adjusted_market_making.ipynb` | Spread skew from inventory |
| `avellaneda_stoikov.ipynb` | Avellaneda–Stoikov (2008) optimal quotes |
| `bond_market_making_hit_ratio.ipynb` | Bond MM with a target hit ratio |
| `databento_cme_ingestion.ipynb` | Load local Databento CME DBN batch files |

## License

MIT — see [LICENSE](LICENSE).
