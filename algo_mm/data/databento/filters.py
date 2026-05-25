"""Post-load filters for symbols and time ranges."""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

import pandas as pd


def parse_time(value: str | date | datetime | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.tz_localize("UTC") if value.tz is None else value.tz_convert("UTC")
    if isinstance(value, datetime):
        ts = pd.Timestamp(value)
        return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    if isinstance(value, date):
        return pd.Timestamp(value, tz="UTC")
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return pd.Timestamp(datetime.strptime(text, "%Y%m%d"), tz="UTC")
    return pd.to_datetime(text, utc=True)


def symbol_matches(symbol: str, query: str) -> bool:
    """Match a raw symbol against a query (exact or parent e.g. ES.FUT)."""
    if query.endswith(".FUT"):
        stem = query[: -len(".FUT")]
        return symbol.startswith(stem)
    return symbol == query


def filter_dataframe(
    df: pd.DataFrame,
    *,
    symbols: str | Iterable[str] | None = None,
    start: str | date | datetime | pd.Timestamp | None = None,
    end: str | date | datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    out = df
    if symbols is not None:
        queries = [symbols] if isinstance(symbols, str) else list(symbols)
        if "symbol" not in out.columns:
            raise ValueError(
                "Cannot filter by symbols: DataFrame has no 'symbol' column. "
                "Load with map_symbols=True (default)."
            )
        mask = False
        for query in queries:
            mask = mask | out["symbol"].astype(str).map(
                lambda s, q=query: symbol_matches(s, q),
            )
        out = out.loc[mask]

    start_ts = parse_time(start)
    end_ts = parse_time(end)
    if start_ts is not None or end_ts is not None:
        if isinstance(out.index, pd.DatetimeIndex):
            idx = out.index
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            else:
                idx = idx.tz_convert("UTC")
            if start_ts is not None:
                out = out.loc[idx >= start_ts]
            if end_ts is not None:
                out = out.loc[idx < end_ts]
        else:
            raise ValueError("Cannot filter by time: index is not a DatetimeIndex.")

    return out
