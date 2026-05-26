"""Load local DBN batch files into pandas via the official Databento client."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import pandas as pd

from algo_mm.data.databento.catalog import CMEDataCatalog
from algo_mm.data.databento.filters import filter_dataframe, parse_time
from algo_mm.data.databento.paths import normalize_schema, resolve_data_root
from algo_mm.data.databento.contracts import enrich_futures_contracts
from algo_mm.data.databento.symbology import (
    build_symbol_lookup,
    instrument_ids_for_symbols,
)

if TYPE_CHECKING:
    import databento as db

_DEFAULT_CATALOG: CMEDataCatalog | None = None
# Above this size, scan records and filter in one pass instead of materializing the full file.
_STREAM_THRESHOLD_BYTES = 400 * 1024 * 1024
_PRICE_COLUMNS = frozenset({"open", "high", "low", "close", "price"})


def _require_databento() -> type:
    try:
        import databento as db
    except ImportError as exc:
        raise ImportError(
            "Reading DBN files requires the 'databento' package. "
            "Install with: uv pip install -e '.[data]'"
        ) from exc
    return db


def open_dbn(path: str | Path, *, inject_symbology: bool = True) -> "db.DBNStore":
    """
    Open a local ``.dbn`` file as a :class:`databento.DBNStore`.

    Batch downloads ship ``symbology.json`` beside the DBN shards; it is merged
    automatically so ``map_symbols=True`` resolves raw CME symbols correctly.
    """
    db = _require_databento()
    path = Path(path)
    store = db.DBNStore.from_file(path)
    if inject_symbology:
        sym_path = path.parent / "symbology.json"
        if sym_path.is_file():
            with open(sym_path) as f:
                store.insert_symbology_json(json.load(f))
    return store


def _price_type(pretty_px: bool) -> str:
    """Map our ``pretty_px`` flag to Databento ``price_type`` (v0.78+)."""
    return "float" if pretty_px else "fixed"


def apply_pretty_px(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Databento fixed-point prices (int × 1e-9) to floats.

    Raw DBN values like ``5302000000000`` are ~5302.00, not epoch timestamps.
    """
    try:
        from databento_dbn import FIXED_PRICE_SCALE, UNDEF_PRICE
    except ImportError:
        FIXED_PRICE_SCALE = 1e-9
        UNDEF_PRICE = 9223372036854775807

    out = df.copy()
    for col in _PRICE_COLUMNS:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].replace(UNDEF_PRICE, np.nan) / FIXED_PRICE_SCALE
    return out


def load(
    schema: str,
    *,
    symbols: str | list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    root: str | Path | None = None,
    job_id: str | None = None,
    map_symbols: bool = True,
    pretty_px: bool = True,
    contract_metadata: bool = False,
    streaming: bool | None = None,
    catalog: CMEDataCatalog | None = None,
) -> pd.DataFrame:
    """
    Load records for a schema into one DataFrame.

    Parameters
    ----------
    schema
        Databento schema id (e.g. ``"ohlcv-1m"``, ``"mbo"``, ``"mbp-10"``, ``"tbbo"``).
    symbols
        Parent symbols (``"ES.FUT"``) or raw symbols (``"ESM5"``). Omit to load all.
    start, end
        UTC range filter on the event timestamp index (``end`` exclusive).
    root
        Override data root (default: ``DATABENTO_DATA_ROOT`` or ~/Documents/Databento/CME-Futures).
    job_id
        Restrict to a single batch folder (e.g. ``"GLBX-20250608-ESWGBFBE48"``).
    map_symbols
        Add a ``symbol`` column via symbology (recommended).
    pretty_px
        If True (default), scale ``open``/``high``/``low``/``close`` from fixed-point
        integers to floats (divide by 1e-9 per Databento encoding).
    streaming
        If True, stream/filter large DBN shards in one pass (less memory).
        Default: on when any shard exceeds ~400MB and a time or symbol filter is set.
    contract_metadata
        If True, add ``activation``, ``expiration``, ``contract_rank`` (0=front),
        and ``contract_label`` (front/next/...) using symbology + CME expiry rules.

    Examples
    --------
    >>> from algo_mm.data import load
    >>> bars = load("ohlcv-1m", symbols="ES.FUT", start="2024-06-01", end="2024-06-02")
    """
    cat = catalog or _get_catalog(root)
    paths = cat.resolve_files(
        normalize_schema(schema),
        start=start,
        end=end,
        job_id=job_id,
    )
    total_bytes = sum(p.stat().st_size for p in paths)
    use_stream = _use_streaming(streaming, total_bytes, symbols, start, end)
    if total_bytes > _STREAM_THRESHOLD_BYTES and not use_stream:
        warnings.warn(
            f"Loading {total_bytes / 1e9:.1f} GB of DBN without streaming; "
            "pass symbols/start/end or streaming=True for a filtered scan, "
            "or ingest_parquet() first.",
            stacklevel=2,
        )
    frames = [
        _read_dbn_file(
            p,
            map_symbols=map_symbols,
            pretty_px=pretty_px,
            symbols=symbols,
            start=start,
            end=end,
            streaming=use_stream,
        )
        for p in paths
    ]
    df = pd.concat(frames, axis=0) if len(frames) > 1 else frames[0]
    if not use_stream:
        df = filter_dataframe(df, symbols=symbols, start=start, end=end)
    if contract_metadata:
        sym_path = paths[0].parent / "symbology.json"
        df = enrich_futures_contracts(
            df,
            symbology_path=sym_path if sym_path.is_file() else None,
        )
    return df


def iter_load(
    schema: str,
    *,
    symbols: str | list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    root: str | Path | None = None,
    job_id: str | None = None,
    map_symbols: bool = True,
    pretty_px: bool = True,
    catalog: CMEDataCatalog | None = None,
) -> Iterator[pd.DataFrame]:
    """
    Yield one DataFrame per matching ``.dbn`` file (for large MBO / MBP shards).

    Apply the same ``symbols`` / ``start`` / ``end`` filters to each chunk.
    """
    cat = catalog or _get_catalog(root)
    for path in cat.resolve_files(
        normalize_schema(schema),
        start=start,
        end=end,
        job_id=job_id,
    ):
        use_stream = _use_streaming(None, path.stat().st_size, symbols, start, end)
        df = _read_dbn_file(
            path,
            map_symbols=map_symbols,
            pretty_px=pretty_px,
            symbols=symbols,
            start=start,
            end=end,
            streaming=use_stream,
        )
        if not use_stream:
            df = filter_dataframe(df, symbols=symbols, start=start, end=end)
        yield df


def ingest_parquet(
    schema: str,
    cache_dir: str | Path,
    *,
    start: str | None = None,
    end: str | None = None,
    root: str | Path | None = None,
    job_id: str | None = None,
    force: bool = False,
    map_symbols: bool = True,
    catalog: CMEDataCatalog | None = None,
) -> list[Path]:
    """
    Convert matching DBN shards to Parquet under ``cache_dir/<schema>/``.

    Returns paths of written (or existing) parquet files. Use for faster repeat loads.
    """
    cache_dir = Path(cache_dir)
    schema_key = normalize_schema(schema)
    out_dir = cache_dir / schema_key
    out_dir.mkdir(parents=True, exist_ok=True)

    cat = catalog or _get_catalog(root)
    written: list[Path] = []
    for dbn_path in cat.resolve_files(schema_key, start=start, end=end, job_id=job_id):
        out_path = out_dir / f"{dbn_path.stem}.parquet"
        if out_path.exists() and not force:
            written.append(out_path)
            continue
        store = open_dbn(dbn_path)
        store.to_parquet(
            out_path,
            map_symbols=map_symbols,
            price_type="float",
            mode="w",
        )
        written.append(out_path)
    return written


def load_parquet(
    schema: str,
    cache_dir: str | Path,
    *,
    symbols: str | list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load previously ingested Parquet shards for a schema."""
    schema_key = normalize_schema(schema)
    folder = Path(cache_dir) / schema_key
    if not folder.is_dir():
        raise FileNotFoundError(f"No parquet cache at {folder}")
    start_ts = parse_time(start)
    end_ts = parse_time(end)
    frames: list[pd.DataFrame] = []
    for path in sorted(folder.glob("*.parquet")):
        df = pd.read_parquet(path)
        if start_ts is not None or end_ts is not None:
            df = filter_dataframe(df, start=start, end=end)
        if symbols is not None:
            df = filter_dataframe(df, symbols=symbols)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No parquet files in {folder}")
    return pd.concat(frames, axis=0) if len(frames) > 1 else frames[0]


def _use_streaming(
    streaming: bool | None,
    nbytes: int,
    symbols: str | list[str] | None,
    start: str | None,
    end: str | None,
) -> bool:
    if streaming is not None:
        return streaming
    if nbytes < _STREAM_THRESHOLD_BYTES:
        return False
    return symbols is not None or start is not None or end is not None


def _read_dbn_file(
    path: Path,
    *,
    map_symbols: bool,
    pretty_px: bool = True,
    symbols: str | list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    streaming: bool = False,
) -> pd.DataFrame:
    if streaming:
        return _stream_filtered_dbn(
            path,
            symbols=symbols,
            start=start,
            end=end,
            map_symbols=map_symbols,
            pretty_px=pretty_px,
        )
    return open_dbn(path).to_df(
        map_symbols=map_symbols,
        price_type=_price_type(pretty_px),
    )


def _stream_filtered_dbn(
    path: Path,
    *,
    symbols: str | list[str] | None,
    start: str | None,
    end: str | None,
    map_symbols: bool,
    pretty_px: bool = True,
) -> pd.DataFrame:
    store = open_dbn(path)
    start_ts = parse_time(start)
    end_ts = parse_time(end)
    start_ns = int(start_ts.value) if start_ts is not None else None
    end_ns = int(end_ts.value) if end_ts is not None else None

    queries: list[str] | None = None
    id_filter: set[int] | None = None
    sym_path = path.parent / "symbology.json"
    symbol_lookup: dict[tuple[int, Any], str] | None = None
    if symbols is not None:
        queries = [symbols] if isinstance(symbols, str) else list(symbols)
        if sym_path.is_file():
            id_filter = instrument_ids_for_symbols(sym_path, queries)
            if not id_filter:
                raise ValueError(f"No instrument ids matched symbols={symbols!r}")
            symbol_lookup = build_symbol_lookup(sym_path, queries)

    rows: list[dict[str, Any]] = []
    timestamps: list[int] = []
    for record in store:
        if id_filter is not None and record.instrument_id not in id_filter:
            continue
        ts = int(record.ts_event)
        if start_ns is not None and ts < start_ns:
            continue
        if end_ns is not None and ts >= end_ns:
            continue
        fields = getattr(type(record), "_ordered_fields", None)
        if fields is None:
            fields = [f for f in dir(record) if not f.startswith("_") and not f.startswith("pretty_")]
        row = {name: getattr(record, name) for name in fields if name != "ts_event"}
        rows.append(row)
        timestamps.append(ts)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    ts = pd.to_datetime(timestamps, unit="ns", utc=True)
    df = df.set_index(ts)
    if map_symbols and symbol_lookup is not None:
        df["symbol"] = [
            symbol_lookup.get((iid, t.date()))
            for iid, t in zip(df["instrument_id"], df.index)
        ]
    elif map_symbols and "instrument_id" in df.columns:
        dates = df.index.date.to_numpy().astype("datetime64[D]")
        df["symbol"] = store._instrument_map.resolve_many(  # noqa: SLF001
            df["instrument_id"].to_numpy(),
            dates,
        )
    if pretty_px:
        df = apply_pretty_px(df)
    return df


def _get_catalog(root: str | Path | None) -> CMEDataCatalog:
    global _DEFAULT_CATALOG
    if root is not None:
        return CMEDataCatalog(root)
    if _DEFAULT_CATALOG is None:
        _DEFAULT_CATALOG = CMEDataCatalog(resolve_data_root(None))
    return _DEFAULT_CATALOG
