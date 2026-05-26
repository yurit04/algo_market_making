"""Tests for CME futures contract enrichment."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from algo_mm.data import list_contracts, list_parent_symbols
from algo_mm.data.databento.contracts import (
    cme_quarterly_expiration,
    enrich_futures_contracts,
    parse_outright_symbol,
)

SYMOLOGY = Path.home() / "Documents/Databento/CME-Futures/OHLCV-1m/GLBX-20250608-ESWGBFBE48/symbology.json"


def test_parse_esm0() -> None:
    p = parse_outright_symbol("ESM0", ref=date(2010, 6, 6))
    assert p is not None
    assert p.root == "ES"
    assert p.month == 6
    assert p.year == 2010


def test_expiration_june_2010() -> None:
    exp = cme_quarterly_expiration(6, 2010)
    assert exp.date() == date(2010, 6, 18)


def test_list_parent_symbols_ohlcv() -> None:
    parents = list_parent_symbols("ohlcv-1m")
    assert isinstance(parents, list)
    assert "ES.FUT" in parents
    assert len(parents) >= 40


def test_list_contracts_es_parent() -> None:
    if not SYMOLOGY.parent.is_dir():
        return
    df = list_contracts("ohlcv-1m", parent="ES.FUT", start="2010-06-01", end="2010-07-01")
    assert "symbol" in df.columns
    assert (df["symbol"].str.startswith("ES") | df["is_spread"]).all()
    assert "ESM0" in df["symbol"].values or "ESU0" in df["symbol"].values


def test_contract_rank_front_and_next() -> None:
    ts = pd.Timestamp("2010-06-06 22:00:00", tz="UTC")
    df = pd.DataFrame(
        {
            "instrument_id": [6640, 26714],
            "symbol": ["ESM0", "ESU0"],
            "close": [1058.75, 1063.25],
        },
        index=[ts, ts],
    )
    if not SYMOLOGY.is_file():
        return
    out = enrich_futures_contracts(df, symbology_path=SYMOLOGY)
    ranks = dict(zip(out["symbol"], out["contract_rank"]))
    assert ranks["ESM0"] == 0
    assert ranks["ESU0"] == 1
    assert out.loc[out["symbol"] == "ESM0", "contract_label"].iloc[0] == "front"
    assert out.loc[out["symbol"] == "ESU0", "contract_label"].iloc[0] == "next"
    assert out["activation"].notna().all()
    assert out["expiration"].notna().all()
