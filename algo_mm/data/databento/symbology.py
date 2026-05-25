"""Resolve parent symbols to instrument ids using batch symbology.json."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from algo_mm.data.databento.filters import symbol_matches


def load_symbology_result(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data.get("result", data)


def instrument_ids_for_symbols(symbology_path: Path, symbols: list[str]) -> set[int]:
    """Map parent/raw symbol queries to numeric instrument ids."""
    result = load_symbology_result(symbology_path)
    ids: set[int] = set()
    for raw_symbol, intervals in result.items():
        for query in symbols:
            if symbol_matches(raw_symbol, query):
                for interval in intervals:
                    ids.add(int(interval["s"]))
                break
    return ids


def build_symbol_lookup(
    symbology_path: Path,
    symbols: list[str] | None = None,
) -> dict[tuple[int, date], str]:
    """
    Map (instrument_id, calendar date) -> raw symbol using batch symbology.json intervals.
    """
    result = load_symbology_result(symbology_path)
    lookup: dict[tuple[int, date], str] = {}
    for raw_symbol, intervals in result.items():
        if symbols is not None and not any(symbol_matches(raw_symbol, q) for q in symbols):
            continue
        for interval in intervals:
            iid = int(interval["s"])
            d0 = date.fromisoformat(interval["d0"])
            d1 = date.fromisoformat(interval["d1"])
            day = d0
            while day <= d1:
                lookup[(iid, day)] = raw_symbol
                day += timedelta(days=1)
    return lookup
