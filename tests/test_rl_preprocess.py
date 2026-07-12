"""
Real-data smoke test for the MBO preprocessing + replay pipeline.

Skipped unless a preprocessed session cache is present (it is gitignored and produced
by ``python -m algo_mm.data.preprocess_mbo``). Validates the invariants established
against the raw feed: the reconstructed book is never crossed at F_LAST boundaries and
the top of book is economically sane.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from algo_mm.rl import ASK, BID, CachedSession, OrderBook, PRICE_SCALE, replay_session

_CACHE = Path("outputs/rl_cache/mbo")


def _first_cache() -> Path | None:
    if not _CACHE.is_dir():
        return None
    files = sorted(_CACHE.glob("*/*.parquet"))
    return files[0] if files else None


@pytest.mark.skipif(_first_cache() is None, reason="no preprocessed MBO session cache present")
def test_replay_real_session_invariants():
    path = _first_cache()
    full = CachedSession.from_parquet(path)
    # Cap to the first chunk for a fast test; enough to exercise real book dynamics.
    n = min(len(full), 300_000)
    sess = CachedSession(
        ts_event=full.ts_event[:n], action=full.action[:n], side=full.side[:n],
        price=full.price[:n], size=full.size[:n], order_id=full.order_id[:n],
        sequence=full.sequence[:n], flags=full.flags[:n],
        symbol=full.symbol, date=full.date,
    )

    crossed = {"n": 0, "checks": 0}

    def snapshot(i, ts, book):
        crossed["checks"] += 1
        if book.is_crossed():
            crossed["n"] += 1

    book = OrderBook()
    replay_session(sess, book=book, on_snapshot=snapshot)

    assert crossed["checks"] > 0
    assert crossed["n"] == 0  # never crossed at F_LAST boundaries

    bid, ask = book.best_bid(), book.best_ask()
    assert bid is not None and ask is not None
    assert bid < ask
    # ES trades in the thousands; a sane mid rules out price-scale/decoding regressions.
    assert 1_000 < book.mid() * PRICE_SCALE < 100_000
