"""
Preprocess raw Databento MBO DBN files into compact per-session caches.

Raw daily MBO shards are large (multi-GB, hundreds of millions of records across all
instruments). For RL training we replay a single instrument over a single session
thousands of times, so we extract once into a compact per-session Parquet file holding
only the columns needed to reconstruct the book and trade stream:

    ts_event, action (code), side (+1/-1/0), price (raw int), size, order_id,
    sequence, flags

Front-month selection
---------------------
CME equity-index futures roll on volume ~8 days before expiry, so the tradeable
"front" contract is the *most active* outright on a given session, not simply the
nearest expiry. :func:`most_active_instrument` picks it by trade count; callers may
also pin an explicit raw symbol (e.g. ``ESM5``).

CLI
---
    python -m algo_mm.data.preprocess_mbo --parent ES.FUT \
        --start 2025-05-07 --end 2025-06-28 \
        --session 13:30 20:00 --out outputs/rl_cache
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from algo_mm.data.databento.catalog import CMEDataCatalog
from algo_mm.data.databento.loader import open_dbn
from algo_mm.data.databento.paths import parse_dbn_filename
from algo_mm.data.databento.symbology import load_symbology_result
from algo_mm.rl.book import (
    UNDEF_PRICE,
    action_from_char,
    side_from_char,
)

# TRADE action code, used to rank instruments by activity.
_TRADE_CODE = action_from_char("T")

DEFAULT_ROOT = "/Users/yuriturygin/Documents/market_data/Databento/CME-Futures"


@dataclass(frozen=True)
class SessionWindow:
    """A daily UTC session window, e.g. RTH 13:30-20:00 for ES during EDT."""

    start: time
    end: time

    def bounds_ns(self, day: date) -> tuple[int, int]:
        lo = datetime.combine(day, self.start, tzinfo=timezone.utc)
        hi = datetime.combine(day, self.end, tzinfo=timezone.utc)
        return int(pd.Timestamp(lo).value), int(pd.Timestamp(hi).value)


def _action_char(action) -> str:
    return action.value if hasattr(action, "value") else str(action)


def _side_char(side) -> str:
    return side.value if hasattr(side, "value") else str(side)


def most_active_instrument(
    dbn_path: Path,
    instrument_ids: set[int],
    *,
    lo_ns: int,
    hi_ns: int,
) -> int | None:
    """Return the instrument id (from ``instrument_ids``) with the most trades in-window."""
    store = open_dbn(dbn_path)
    counts: Counter[int] = Counter()
    for rec in store:
        iid = rec.instrument_id
        if iid not in instrument_ids:
            continue
        ts = int(rec.ts_event)
        if ts < lo_ns or ts >= hi_ns:
            continue
        if _action_char(rec.action) == "T":
            counts[iid] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def extract_session(
    dbn_path: Path,
    instrument_id: int,
    *,
    lo_ns: int,
    hi_ns: int,
) -> pd.DataFrame:
    """Scan one DBN shard and return the compact event frame for one instrument/window."""
    store = open_dbn(dbn_path)
    ts_event: list[int] = []
    action: list[int] = []
    side: list[int] = []
    price: list[int] = []
    size: list[int] = []
    order_id: list[int] = []
    sequence: list[int] = []
    flags: list[int] = []

    for rec in store:
        if rec.instrument_id != instrument_id:
            continue
        ts = int(rec.ts_event)
        if ts < lo_ns:
            continue
        if ts >= hi_ns:
            break
        px = int(rec.price)
        ts_event.append(ts)
        action.append(action_from_char(_action_char(rec.action)))
        side.append(side_from_char(_side_char(rec.side)))
        price.append(px if px != UNDEF_PRICE else UNDEF_PRICE)
        size.append(int(rec.size))
        order_id.append(int(rec.order_id))
        sequence.append(int(rec.sequence))
        flags.append(int(rec.flags) & 0xFF)

    return pd.DataFrame(
        {
            "ts_event": np.asarray(ts_event, dtype=np.int64),
            "action": np.asarray(action, dtype=np.int8),
            "side": np.asarray(side, dtype=np.int8),
            "price": np.asarray(price, dtype=np.int64),
            "size": np.asarray(size, dtype=np.int64),
            "order_id": np.asarray(order_id, dtype=np.int64),
            "sequence": np.asarray(sequence, dtype=np.int64),
            "flags": np.asarray(flags, dtype=np.uint8),
        }
    )


def _instrument_ids_for_root(symbology: dict, root: str, day: date) -> dict[str, int]:
    """Map active outright symbols of ``root`` to instrument ids on ``day`` (skips spreads)."""
    out: dict[str, int] = {}
    iso = day.isoformat()
    for symbol, intervals in symbology.items():
        if "-" in symbol or "." in symbol or not symbol.startswith(root):
            continue
        # Outright of this root: root immediately followed by a month code + year.
        rest = symbol[len(root):]
        if not rest or not rest[0].isalpha():
            continue
        for iv in intervals:
            if iv["d0"] <= iso < iv["d1"]:
                out[symbol] = int(iv["s"])
                break
    return out


def preprocess(
    parent: str,
    start: str,
    end: str,
    out_dir: str | Path,
    *,
    session: SessionWindow,
    root: str | Path | None = None,
    symbol: str | None = None,
    force: bool = False,
) -> list[Path]:
    """
    Extract compact session caches for the ``parent`` root's front-month over a date range.

    For each daily DBN shard in [start, end): resolve the root's outright instrument ids
    from symbology, pick the most-active one (unless ``symbol`` pins a specific outright),
    extract the session window, and write ``<out_dir>/<schema>/<symbol>/<date>.parquet``.
    """
    root = str(root or DEFAULT_ROOT)
    root_sym = parent.split(".")[0]
    catalog = CMEDataCatalog(root)
    out_base = Path(out_dir) / "mbo"

    written: list[Path] = []
    for job in catalog.batches("mbo"):
        symbology = load_symbology_result(job.symbology_path) if job.symbology_path.is_file() else {}
        for info in job.dbn_files:
            day = info.start
            if not (date.fromisoformat(start) <= day < date.fromisoformat(end)):
                continue
            lo_ns, hi_ns = session.bounds_ns(day)
            id_map = _instrument_ids_for_root(symbology, root_sym, day)
            if not id_map:
                continue

            if symbol is not None:
                if symbol not in id_map:
                    continue
                iid, chosen = id_map[symbol], symbol
            else:
                iid = most_active_instrument(
                    info.path, set(id_map.values()), lo_ns=lo_ns, hi_ns=hi_ns
                )
                if iid is None:
                    continue
                inv = {v: k for k, v in id_map.items()}
                chosen = inv.get(iid, str(iid))

            out_path = out_base / chosen / f"{day.isoformat()}.parquet"
            if out_path.exists() and not force:
                written.append(out_path)
                continue

            df = extract_session(info.path, iid, lo_ns=lo_ns, hi_ns=hi_ns)
            if df.empty:
                continue
            df.attrs["symbol"] = chosen
            df.attrs["date"] = day.isoformat()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_path, index=False)
            written.append(out_path)
    return written


def _parse_hhmm(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Cache Databento MBO sessions for RL market making.")
    p.add_argument("--parent", default="ES.FUT", help="Parent symbol, e.g. ES.FUT")
    p.add_argument("--symbol", default=None, help="Pin a specific outright (e.g. ESM5); else most-active")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD (exclusive)")
    p.add_argument("--session", nargs=2, default=["13:30", "20:00"], metavar=("START", "END"),
                   help="UTC session window HH:MM HH:MM (default RTH 13:30 20:00)")
    p.add_argument("--out", default="outputs/rl_cache", help="Output cache directory")
    p.add_argument("--root", default=None, help="Databento data root override")
    p.add_argument("--force", action="store_true", help="Overwrite existing caches")
    args = p.parse_args(argv)

    window = SessionWindow(_parse_hhmm(args.session[0]), _parse_hhmm(args.session[1]))
    paths = preprocess(
        args.parent, args.start, args.end, args.out,
        session=window, root=args.root, symbol=args.symbol, force=args.force,
    )
    print(f"Wrote/kept {len(paths)} session cache file(s) under {Path(args.out) / 'mbo'}")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
