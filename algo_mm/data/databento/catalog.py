"""Index and query local Databento CME batch download folders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterator

import pandas as pd

from algo_mm.data.databento.filters import parse_time
from algo_mm.data.databento.paths import (
    DbFileInfo,
    FOLDER_TO_SCHEMA,
    normalize_schema,
    ns_to_datetime,
    parse_dbn_filename,
    resolve_data_root,
    schema_folders,
)


@dataclass(frozen=True)
class BatchJob:
    """One Databento batch download directory (job_id folder)."""

    job_id: str
    schema: str
    dataset: str
    symbols: tuple[str, ...]
    start: pd.Timestamp
    end: pd.Timestamp
    path: Path
    split_duration: str | None
    dbn_files: tuple[DbFileInfo, ...]

    @property
    def symbology_path(self) -> Path:
        return self.path / "symbology.json"

    @property
    def metadata_path(self) -> Path:
        return self.path / "metadata.json"


class CMEDataCatalog:
    """
    Discover and load CME Globex (GLBX.MDP3) data from local Databento batch downloads.

    Default root: ``~/Documents/Databento/CME-Futures`` or ``DATABENTO_DATA_ROOT``.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = resolve_data_root(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Data root does not exist: {self.root}")
        self._jobs: dict[str, list[BatchJob]] | None = None

    def _scan(self) -> dict[str, list[BatchJob]]:
        if self._jobs is not None:
            return self._jobs

        by_schema: dict[str, list[BatchJob]] = {}
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            schema = FOLDER_TO_SCHEMA.get(folder.name)
            if schema is None:
                continue
            for job_dir in sorted(folder.iterdir()):
                if not job_dir.is_dir():
                    continue
                job = _load_batch_job(job_dir, schema)
                if job is not None:
                    by_schema.setdefault(schema, []).append(job)

        self._jobs = by_schema
        return by_schema

    def list_schemas(self) -> list[str]:
        return sorted(self._scan().keys())

    def batches(self, schema: str) -> list[BatchJob]:
        key = normalize_schema(schema)
        jobs = self._scan().get(key, [])
        if not jobs:
            available = ", ".join(self.list_schemas()) or "(none)"
            raise KeyError(f"Schema {schema!r} not found. Available: {available}")
        return list(jobs)

    def describe(self, schema: str | None = None) -> pd.DataFrame:
        """Summary table of indexed batch jobs (one row per job folder)."""
        rows: list[dict] = []
        schemas = [normalize_schema(schema)] if schema else self.list_schemas()
        for sch in schemas:
            for job in self.batches(sch):
                rows.append(
                    {
                        "schema": job.schema,
                        "job_id": job.job_id,
                        "dataset": job.dataset,
                        "symbols": ", ".join(job.symbols[:8])
                        + ("..." if len(job.symbols) > 8 else ""),
                        "start": job.start,
                        "end": job.end,
                        "n_files": len(job.dbn_files),
                        "split": job.split_duration or "none",
                        "path": str(job.path),
                    }
                )
        return pd.DataFrame(rows)

    def resolve_files(
        self,
        schema: str,
        *,
        start: str | date | None = None,
        end: str | date | None = None,
        job_id: str | None = None,
    ) -> list[Path]:
        """Return ``.dbn`` paths whose calendar span overlaps ``[start, end)``."""
        start_ts = parse_time(start)
        end_ts = parse_time(end)
        q_start = start_ts.date() if start_ts is not None else None
        q_end = end_ts.date() if end_ts is not None else None

        paths: list[Path] = []
        for job in self.batches(schema):
            if job_id is not None and job.job_id != job_id:
                continue
            for info in job.dbn_files:
                if _file_overlaps(info, q_start, q_end):
                    paths.append(info.path)
        paths.sort()
        if not paths:
            raise FileNotFoundError(
                f"No DBN files for schema={schema!r} in range "
                f"start={start!r} end={end!r} under {self.root}",
            )
        return paths

    def iter_jobs(self, schema: str) -> Iterator[BatchJob]:
        yield from self.batches(schema)

    def list_contracts(
        self,
        schema: str,
        *,
        job_id: str | None = None,
        parent: str | list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """
        List every contract (raw symbol) in local batch data for this schema.

        See :func:`algo_mm.data.list_contracts` for parameter details.
        """
        from algo_mm.data.databento.contracts_catalog import contracts_for_schema

        return contracts_for_schema(
            self,
            schema,
            job_id=job_id,
            parent=parent,
            start=start,
            end=end,
        )

    def list_parent_symbols(
        self,
        schema: str,
        *,
        job_id: str | None = None,
        as_dataframe: bool = False,
    ) -> list[str] | pd.DataFrame:
        """
        List parent symbols (e.g. ``ES.FUT``) requested in local batch jobs for this schema.

        See :func:`algo_mm.data.list_parent_symbols`.
        """
        from algo_mm.data.databento.contracts_catalog import parent_symbols_for_schema

        return parent_symbols_for_schema(
            self,
            schema,
            job_id=job_id,
            as_dataframe=as_dataframe,
        )


def _load_batch_job(job_dir: Path, schema: str) -> BatchJob | None:
    meta_path = job_dir / "metadata.json"
    if not meta_path.is_file():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    query = meta.get("query", {})
    custom = meta.get("customizations", {})
    dbn_files = tuple(
        sorted(
            (info for p in job_dir.glob("*.dbn") if (info := parse_dbn_filename(p))),
            key=lambda x: (x.start, x.part or 0, x.path.name),
        )
    )
    if not dbn_files:
        return None
    return BatchJob(
        job_id=meta.get("job_id", job_dir.name),
        schema=query.get("schema", schema),
        dataset=query.get("dataset", "GLBX.MDP3"),
        symbols=tuple(query.get("symbols", ())),
        start=pd.Timestamp(ns_to_datetime(query["start"])),
        end=pd.Timestamp(ns_to_datetime(query["end"])),
        path=job_dir,
        split_duration=custom.get("split_duration"),
        dbn_files=dbn_files,
    )


def _file_overlaps(
    info: DbFileInfo,
    query_start: date | None,
    query_end: date | None,
) -> bool:
    if query_start is None and query_end is None:
        return True
    f_start, f_end = info.start, info.end
    if query_start is not None and f_end < query_start:
        return False
    if query_end is not None and f_start >= query_end:
        return False
    return True
