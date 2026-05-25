"""Path conventions for local Databento batch downloads."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

DEFAULT_DATA_ROOT = Path.home() / "Documents" / "Databento" / "CME-Futures"

# Folder names on disk -> Databento schema ids (metadata.json "schema" field).
FOLDER_TO_SCHEMA: dict[str, str] = {
    "MBO": "mbo",
    "MBO-10": "mbp-10",
    "TTBO": "tbbo",
    "OHLCV-1s": "ohlcv-1s",
    "OHLCV-1m": "ohlcv-1m",
    "OHLCV-1h": "ohlcv-1h",
    "OHLCV-1d": "ohlcv-1d",
}

SCHEMA_TO_FOLDERS: dict[str, list[str]] = {}
for folder, schema in FOLDER_TO_SCHEMA.items():
    SCHEMA_TO_FOLDERS.setdefault(schema, []).append(folder)

DBN_FILENAME_RE = re.compile(
    r"^glbx-mdp3-(?P<start>\d{8})(?:-(?P<end>\d{8}))?\.(?P<schema>[\w-]+)(?:\.(?P<part>\d+))?\.dbn$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DbFileInfo:
    path: Path
    schema: str
    start: date
    end: date
    part: int | None


def resolve_data_root(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root).expanduser().resolve()
    env = os.environ.get("DATABENTO_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_DATA_ROOT.resolve()


def normalize_schema(schema: str) -> str:
    key = schema.strip().lower()
    aliases = {
        "mbo10": "mbp-10",
        "mbp10": "mbp-10",
        "mbo-10": "mbp-10",
        "tbbo": "tbbo",
        "ttbo": "tbbo",
        "ohlcv1m": "ohlcv-1m",
        "ohlcv-1s-v2": "ohlcv-1s",
    }
    return aliases.get(key, key)


def schema_folders(root: Path, schema: str) -> list[Path]:
    normalized = normalize_schema(schema)
    folders = SCHEMA_TO_FOLDERS.get(normalized, [])
    if not folders:
        # Allow direct folder name match (e.g. user passes "OHLCV-1m").
        candidate = root / schema
        if candidate.is_dir():
            return [candidate]
        candidate = root / schema.upper()
        if candidate.is_dir():
            return [candidate]
        raise FileNotFoundError(
            f"No data folder for schema {schema!r} under {root}. "
            f"Known schemas: {sorted(SCHEMA_TO_FOLDERS)}",
        )
    return [root / name for name in folders if (root / name).is_dir()]


def parse_dbn_filename(path: Path) -> DbFileInfo | None:
    match = DBN_FILENAME_RE.match(path.name)
    if not match:
        return None
    start = _parse_ymd(match.group("start"))
    end_s = match.group("end")
    end = _parse_ymd(end_s) if end_s else start
    return DbFileInfo(
        path=path,
        schema=match.group("schema").lower(),
        start=start,
        end=end,
        part=int(match.group("part")) if match.group("part") else None,
    )


def ns_to_datetime(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def _parse_ymd(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()
