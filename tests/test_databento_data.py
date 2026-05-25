"""Tests for local Databento catalog (no full DBN decode)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from algo_mm.data.databento.catalog import CMEDataCatalog, _file_overlaps
from algo_mm.data.databento.filters import symbol_matches
from algo_mm.data.databento.paths import (
    DbFileInfo,
    normalize_schema,
    parse_dbn_filename,
)

DATABENTO_ROOT = Path.home() / "Documents" / "Databento" / "CME-Futures"


def test_normalize_schema_aliases() -> None:
    assert normalize_schema("MBO-10") == "mbp-10"
    assert normalize_schema("TTBO") == "tbbo"
    assert normalize_schema("ohlcv-1s-v2") == "ohlcv-1s"


def test_parse_dbn_filename_daily() -> None:
    info = parse_dbn_filename(Path("glbx-mdp3-20250608.mbo.dbn"))
    assert info is not None
    assert info.schema == "mbo"
    assert info.start == date(2025, 6, 8)
    assert info.end == date(2025, 6, 8)


def test_parse_dbn_filename_range_and_part() -> None:
    info = parse_dbn_filename(
        Path("glbx-mdp3-20100606-20250627.ohlcv-1s.0008.dbn"),
    )
    assert info is not None
    assert info.start == date(2010, 6, 6)
    assert info.end == date(2025, 6, 27)
    assert info.part == 8


def test_symbol_matches_parent() -> None:
    assert symbol_matches("ESM5", "ES.FUT")
    assert not symbol_matches("NQM5", "ES.FUT")


def test_file_overlap() -> None:
    info = DbFileInfo(
        path=Path("x.dbn"),
        schema="mbo",
        start=date(2025, 6, 8),
        end=date(2025, 6, 8),
        part=None,
    )
    assert _file_overlaps(info, date(2025, 6, 8), date(2025, 6, 9))
    assert not _file_overlaps(info, date(2025, 6, 9), None)


@pytest.mark.skipif(not DATABENTO_ROOT.is_dir(), reason="local Databento data not present")
class TestLiveCatalog:
    def test_list_schemas(self) -> None:
        cat = CMEDataCatalog(DATABENTO_ROOT)
        schemas = cat.list_schemas()
        assert "ohlcv-1m" in schemas
        assert "mbo" in schemas

    def test_resolve_daily_mbo(self) -> None:
        cat = CMEDataCatalog(DATABENTO_ROOT)
        paths = cat.resolve_files("mbo", start="2025-06-08", end="2025-06-09")
        assert any("20250608" in p.name for p in paths)
