"""List contracts available in local batch downloads for a schema."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from algo_mm.data.databento.contracts import parse_outright_symbol
from algo_mm.data.databento.filters import parse_time, symbol_matches
from algo_mm.data.databento.symbology import load_symbology_result

if TYPE_CHECKING:
    from algo_mm.data.databento.catalog import CMEDataCatalog


def _interval_overlaps(d0: date, d1: date, q_start: date | None, q_end: date | None) -> bool:
    if q_start is not None and d1 < q_start:
        return False
    if q_end is not None and d0 >= q_end:
        return False
    return True


def contracts_for_schema(
    catalog: CMEDataCatalog,
    schema: str,
    *,
    job_id: str | None = None,
    parent: str | list[str] | None = None,
    start: str | date | None = None,
    end: str | date | None = None,
) -> pd.DataFrame:
    """
    List every contract (raw symbol) available for a schema in local batch data.

    Reads ``symbology.json`` from each batch job folder. Each row is one
    (symbol, instrument_id, active interval) tuple.

    Parameters
    ----------
    catalog
        Indexed local data catalog.
    schema
        Databento schema id (e.g. ``"ohlcv-1m"``, ``"mbo"``).
    job_id
        Restrict to a single batch folder.
    parent
        Filter to contracts under parent symbol(s), e.g. ``"ES.FUT"`` or ``["ES.FUT", "NQ.FUT"]``.
    start, end
        Only intervals overlapping ``[start, end)`` (UTC calendar dates).

    Returns
    -------
    pd.DataFrame
        Columns: ``schema``, ``job_id``, ``symbol``, ``instrument_id``, ``d0``, ``d1``,
        ``root``, ``is_spread``, ``parent_symbols`` (batch query parents for that job).
    """
    parents = None
    if parent is not None:
        parents = [parent] if isinstance(parent, str) else list(parent)

    start_ts = parse_time(start)
    end_ts = parse_time(end)
    q_start = start_ts.date() if start_ts is not None else None
    q_end = end_ts.date() if end_ts is not None else None

    rows: list[dict] = []
    for job in catalog.batches(schema):
        if job_id is not None and job.job_id != job_id:
            continue
        sym_path = job.symbology_path
        if not sym_path.is_file():
            continue
        result = load_symbology_result(sym_path)
        job_parents = tuple(job.symbols)
        for raw_symbol, intervals in sorted(result.items()):
            if parents is not None and not any(symbol_matches(raw_symbol, p) for p in parents):
                continue
            parsed = parse_outright_symbol(raw_symbol)
            is_spread = parsed is None or parsed.is_spread
            root = None if is_spread else parsed.root
            for interval in intervals:
                d0 = date.fromisoformat(interval["d0"])
                d1 = date.fromisoformat(interval["d1"])
                if not _interval_overlaps(d0, d1, q_start, q_end):
                    continue
                rows.append(
                    {
                        "schema": job.schema,
                        "job_id": job.job_id,
                        "symbol": raw_symbol,
                        "instrument_id": int(interval["s"]),
                        "d0": d0,
                        "d1": d1,
                        "root": root,
                        "is_spread": is_spread,
                        "parent_symbols": job_parents,
                    }
                )

    if not rows:
        msg = f"No contracts found for schema={schema!r}"
        if job_id:
            msg += f" job_id={job_id!r}"
        if parents:
            msg += f" parent={parents!r}"
        raise FileNotFoundError(msg)

    df = pd.DataFrame(rows)
    return df.sort_values(["symbol", "d0", "job_id"]).reset_index(drop=True)


def parent_symbols_for_schema(
    catalog: CMEDataCatalog,
    schema: str,
    *,
    job_id: str | None = None,
    as_dataframe: bool = False,
) -> list[str] | pd.DataFrame:
    """
    List parent symbols (e.g. ``ES.FUT``) available for a schema in local batch downloads.

    These are the ``symbols`` from each job's ``metadata.json`` batch query
    (``stype_in=parent``), not inferred from symbology.

    Parameters
    ----------
    as_dataframe
        If True, return one row per (job_id, parent_symbol) instead of a sorted unique list.
    """
    rows: list[dict] = []
    seen: set[str] = set()
    for job in catalog.batches(schema):
        if job_id is not None and job.job_id != job_id:
            continue
        for parent in job.symbols:
            seen.add(parent)
            if as_dataframe:
                rows.append(
                    {
                        "schema": job.schema,
                        "job_id": job.job_id,
                        "parent_symbol": parent,
                        "job_start": job.start,
                        "job_end": job.end,
                    }
                )

    if not seen:
        raise FileNotFoundError(f"No parent symbols for schema={schema!r}")

    if as_dataframe:
        return pd.DataFrame(rows).sort_values(["job_id", "parent_symbol"]).reset_index(drop=True)
    return sorted(seen)


def list_contracts(
    schema: str,
    *,
    root: str | Path | None = None,
    job_id: str | None = None,
    parent: str | list[str] | None = None,
    start: str | date | None = None,
    end: str | date | None = None,
    catalog: CMEDataCatalog | None = None,
) -> pd.DataFrame:
    """List contracts for ``schema`` (convenience wrapper around :class:`CMEDataCatalog`)."""
    from algo_mm.data.databento.catalog import CMEDataCatalog

    cat = catalog or CMEDataCatalog(root)
    return contracts_for_schema(
        cat,
        schema,
        job_id=job_id,
        parent=parent,
        start=start,
        end=end,
    )


def list_parent_symbols(
    schema: str,
    *,
    root: str | Path | None = None,
    job_id: str | None = None,
    as_dataframe: bool = False,
    catalog: CMEDataCatalog | None = None,
) -> list[str] | pd.DataFrame:
    """List parent symbols for ``schema`` (convenience wrapper around :class:`CMEDataCatalog`)."""
    from algo_mm.data.databento.catalog import CMEDataCatalog

    cat = catalog or CMEDataCatalog(root)
    return parent_symbols_for_schema(
        cat,
        schema,
        job_id=job_id,
        as_dataframe=as_dataframe,
    )
