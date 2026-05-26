"""CME futures contract metadata: parse symbols, expiry, and roll rank."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
import pandas as pd

from algo_mm.data.databento.symbology import load_symbology_result

MONTH_CODE_TO_NUM: dict[str, int] = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}

MONTH_CODES = frozenset(MONTH_CODE_TO_NUM)

# Outright symbols: root + month code + 1–2 digit year (e.g. ESM0, ESZ24).
_OUTRIGHT_RE = re.compile(
    r"^(?P<root>[A-Z][A-Z0-9]*)(?P<month>[FGHJKMNQUVXZ])(?P<year>\d{1,2})$",
)

_RANK_LABELS: dict[int, str] = {
    0: "front",
    1: "next",
    2: "third",
    3: "fourth",
}


@dataclass(frozen=True)
class ParsedContract:
    root: str
    month: int
    year: int
    month_code: str
    is_spread: bool = False


def parse_outright_symbol(symbol: str, *, ref: date | None = None) -> ParsedContract | None:
    """Parse a CME-style outright (e.g. ``ESM0``, ``NQZ24``). Returns None for spreads."""
    if "-" in symbol or "." in symbol:
        return ParsedContract("", 0, 0, "", is_spread=True)

    match = _OUTRIGHT_RE.match(symbol.strip().upper())
    if not match:
        return None

    month_code = match.group("month")
    year_digits = match.group("year")
    ref = ref or date.today()
    if len(year_digits) == 2:
        year = 2000 + int(year_digits) if int(year_digits) < 80 else 1900 + int(year_digits)
    else:
        year = _expand_single_digit_year(int(year_digits), ref)

    return ParsedContract(
        root=match.group("root"),
        month=MONTH_CODE_TO_NUM[month_code],
        year=year,
        month_code=month_code,
    )


def _expand_single_digit_year(digit: int, ref: date) -> int:
    decade = (ref.year // 10) * 10
    year = decade + digit
    if year < ref.year - 5:
        year += 10
    elif year > ref.year + 5:
        year -= 10
    return year


def cme_quarterly_expiration(contract_month: int, contract_year: int) -> datetime:
    """
    Last trade date for standard CME quarterly equity index futures (ES, NQ, etc.).

    Third Friday of the contract month, end of Globex session (~21:00 UTC same day).
    """
    fridays = [
        d
        for d in calendar.Calendar().itermonthdates(contract_year, contract_month)
        if d.weekday() == calendar.FRIDAY and d.month == contract_month
    ]
    expiry_date = fridays[2]
    return datetime.combine(expiry_date, time(21, 0), tzinfo=timezone.utc)


def build_instrument_calendar(
    symbology_path: Path | None,
    symbols: list[str],
) -> pd.DataFrame:
    """
    Per (instrument_id, raw_symbol): activation and expiration timestamps.

    Activation uses symbology ``d0`` when available; otherwise ~1 year before expiry.
    Expiration uses CME quarterly third-Friday rule from the parsed symbol.
    """
    rows: list[dict] = []
    symbology = load_symbology_result(symbology_path) if symbology_path and symbology_path.is_file() else {}

    for raw_symbol in symbols:
        parsed = parse_outright_symbol(raw_symbol)
        if parsed is None or parsed.is_spread:
            continue

        intervals = symbology.get(raw_symbol, [])
        if intervals:
            for interval in intervals:
                d0 = date.fromisoformat(interval["d0"])
                parsed_row = parse_outright_symbol(raw_symbol, ref=d0) or parsed
                expiration = cme_quarterly_expiration(parsed_row.month, parsed_row.year)
                rows.append(
                    {
                        "instrument_id": int(interval["s"]),
                        "symbol": raw_symbol,
                        "root": parsed_row.root,
                        "activation": pd.Timestamp(d0, tz="UTC"),
                        "expiration": expiration,
                    }
                )
        else:
            expiration = cme_quarterly_expiration(parsed.month, parsed.year)
            activation = pd.Timestamp(
                date(parsed.year, parsed.month, 1) - pd.Timedelta(days=365),
                tz="UTC",
            )
            rows.append(
                {
                    "instrument_id": None,
                    "symbol": raw_symbol,
                    "root": parsed.root,
                    "activation": activation,
                    "expiration": expiration,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["instrument_id", "symbol", "root", "activation", "expiration"])
    return pd.DataFrame(rows)


def enrich_futures_contracts(
    df: pd.DataFrame,
    *,
    symbology_path: str | Path | None = None,
    rank_by_root: bool = True,
) -> pd.DataFrame:
    """
    Add contract lifecycle and roll-rank columns to a DataFrame with ``symbol`` and timestamps.

    New columns
    -----------
    activation : datetime64[ns, UTC]
        When the contract mapping starts (symbology ``d0``) or estimated listing.
    expiration : datetime64[ns, UTC]
        Estimated last trade (CME third Friday of contract month).
    contract_rank : int
        0 = front (nearest expiration among active outrights at that timestamp), 1 = next, ...
    contract_label : str
        ``front``, ``next``, ``third``, ``fourth``, or ``deferred_N``.
    is_spread : bool
        True for multi-leg / non-outright symbols.
    """
    if "symbol" not in df.columns:
        raise ValueError("enrich_futures_contracts requires a 'symbol' column (load with map_symbols=True).")

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "ts_event" in out.columns:
            out = out.set_index(pd.to_datetime(out["ts_event"], utc=True))
        else:
            raise ValueError("DataFrame needs a DatetimeIndex or a 'ts_event' column.")

    out = out.reset_index(names="ts_event")
    symbology_path = Path(symbology_path) if symbology_path else None
    unique_symbols = sorted(out["symbol"].astype(str).unique())
    calendar = build_instrument_calendar(symbology_path, unique_symbols)

    if calendar.empty:
        out["activation"] = pd.NaT
        out["expiration"] = pd.NaT
        out["contract_rank"] = pd.NA
        out["contract_label"] = None
        out["is_spread"] = out["symbol"].astype(str).str.contains("-")
        return out.set_index("ts_event")

    # Merge activation/expiration on symbol + instrument_id when id is known.
    cal_by_id = calendar.dropna(subset=["instrument_id"]).copy()
    cal_by_id["instrument_id"] = cal_by_id["instrument_id"].astype("int64")

    if "instrument_id" in out.columns and not cal_by_id.empty:
        merged = out.merge(
            cal_by_id[["instrument_id", "symbol", "activation", "expiration", "root"]],
            on=["instrument_id", "symbol"],
            how="left",
        )
    else:
        merged = out.copy()
        merged["activation"] = pd.NaT
        merged["expiration"] = pd.NaT
        merged["root"] = None

    # Fallback: symbol-only merge for rows without instrument_id match.
    missing = merged["expiration"].isna()
    if missing.any():
        cal_sym = calendar.groupby("symbol", as_index=False).first()
        sym_map = cal_sym.set_index("symbol")[["activation", "expiration", "root"]]
        for col in ("activation", "expiration", "root"):
            merged.loc[missing, col] = merged.loc[missing, "symbol"].map(sym_map[col])

    merged["is_spread"] = merged["symbol"].astype(str).str.contains("-") | merged["root"].isna()

    merged["contract_rank"] = pd.Series(pd.NA, index=merged.index, dtype="Int64")
    merged["contract_label"] = pd.Series(pd.NA, index=merged.index, dtype=object)

    for _ts, group in merged.groupby("ts_event", sort=False):
        outright = group[~group["is_spread"] & group["expiration"].notna()].copy()
        ts_values = pd.to_datetime(outright["ts_event"], utc=True)
        activation = pd.to_datetime(outright["activation"], utc=True)
        expiration = pd.to_datetime(outright["expiration"], utc=True)
        act_ok = activation.isna() | (ts_values >= activation)
        act_ok = act_ok.fillna(True)
        exp_ok = ts_values < expiration
        active = outright[act_ok & exp_ok]
        order = active.sort_values("expiration")["symbol"].unique()
        rank_map = {sym: i for i, sym in enumerate(order)}
        for idx, row in group.iterrows():
            if row["is_spread"]:
                merged.at[idx, "contract_label"] = "spread"
                continue
            sym = row["symbol"]
            if sym in rank_map:
                r = rank_map[sym]
                merged.at[idx, "contract_rank"] = r
                merged.at[idx, "contract_label"] = _RANK_LABELS.get(r, f"deferred_{r}")
            else:
                merged.at[idx, "contract_label"] = "inactive"

    return merged.set_index("ts_event")
