"""SQLite persistence for job metadata.

Deliberately plain ``sqlite3``: no ORM, no migration framework. Schema
versioning is a single ``user_version`` pragma so upgrades stay inspectable.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Iterable

from .states import InferenceMode, JobStatus, RunMode, assert_transition

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id                  TEXT PRIMARY KEY,
    title               TEXT NOT NULL DEFAULT '',
    note                TEXT NOT NULL DEFAULT '',
    entry_name          TEXT NOT NULL,
    status              TEXT NOT NULL,
    run_mode            TEXT NOT NULL,
    inference_mode      TEXT NOT NULL,
    gpu_index           INTEGER,
    stage               TEXT NOT NULL DEFAULT '',
    created_at          REAL NOT NULL,
    started_at          REAL,
    finished_at         REAL,
    exit_code           INTEGER,
    error_summary       TEXT NOT NULL DEFAULT '',
    resolution          REAL NOT NULL,
    contour_level       REAL NOT NULL,
    total_seq_len       INTEGER NOT NULL DEFAULT 0,
    map_filename        TEXT NOT NULL DEFAULT '',
    overwrite           INTEGER NOT NULL DEFAULT 0,
    pid                 INTEGER,
    sequences_json      TEXT NOT NULL DEFAULT '[]',
    command_json        TEXT NOT NULL DEFAULT '[]',
    submitted_by        TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
"""


@dataclass
class Job:
    id: str
    title: str
    note: str
    entry_name: str
    status: JobStatus
    run_mode: RunMode
    inference_mode: InferenceMode
    gpu_index: int | None
    stage: str
    created_at: float
    started_at: float | None
    finished_at: float | None
    exit_code: int | None
    error_summary: str
    resolution: float
    contour_level: float
    total_seq_len: int
    map_filename: str
    overwrite: bool
    pid: int | None
    submitted_by: str = ""
    sequences: list[dict[str, Any]] = field(default_factory=list)
    command: list[str] = field(default_factory=list)

    @property
    def display_title(self) -> str:
        return self.title or self.entry_name or self.id[:8]

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, end - self.started_at)


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        title=row["title"],
        note=row["note"],
        entry_name=row["entry_name"],
        status=JobStatus(row["status"]),
        run_mode=RunMode(row["run_mode"]),
        inference_mode=InferenceMode(row["inference_mode"]),
        gpu_index=row["gpu_index"],
        stage=row["stage"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        exit_code=row["exit_code"],
        error_summary=row["error_summary"],
        resolution=row["resolution"],
        contour_level=row["contour_level"],
        total_seq_len=row["total_seq_len"],
        map_filename=row["map_filename"],
        overwrite=bool(row["overwrite"]),
        pid=row["pid"],
        submitted_by=row["submitted_by"] if "submitted_by" in row.keys() else "",
        sequences=json.loads(row["sequences_json"] or "[]"),
        command=json.loads(row["command_json"] or "[]"),
    )


class JobStore:
    """Thread-safe job metadata store.

    A single connection guarded by a lock is more than sufficient: this server
    handles one operator and a handful of GPUs, and WAL mode keeps the web
    request path from blocking behind worker writes.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self.migrate()

    # -- schema ---------------------------------------------------------
    def migrate(self) -> None:
        """Create or upgrade the schema.

        Migrations are additive only, so an existing job history is never
        rewritten or lost when the server is upgraded.
        """
        with self._lock:
            self._conn.executescript(_SCHEMA)
            current = self._conn.execute("PRAGMA user_version").fetchone()[0]

            if current < 2:
                existing = {
                    row["name"]
                    for row in self._conn.execute("PRAGMA table_info(jobs)").fetchall()
                }
                if "submitted_by" not in existing:
                    self._conn.execute(
                        "ALTER TABLE jobs ADD COLUMN submitted_by TEXT NOT NULL DEFAULT ''"
                    )

            if current < SCHEMA_VERSION:
                self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- writes ---------------------------------------------------------
    def create(
        self,
        *,
        entry_name: str,
        title: str,
        note: str,
        run_mode: RunMode,
        inference_mode: InferenceMode,
        gpu_index: int | None,
        resolution: float,
        contour_level: float,
        total_seq_len: int,
        map_filename: str,
        overwrite: bool,
        sequences: list[dict[str, Any]],
        submitted_by: str = "",
        job_id: str | None = None,
    ) -> Job:
        job_id = job_id or str(uuid.uuid4())
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO jobs (
                    id, title, note, entry_name, status, run_mode, inference_mode,
                    gpu_index, stage, created_at, resolution, contour_level,
                    total_seq_len, map_filename, overwrite, sequences_json,
                    submitted_by
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    job_id,
                    title,
                    note,
                    entry_name,
                    JobStatus.QUEUED.value,
                    run_mode.value,
                    inference_mode.value,
                    gpu_index,
                    "",
                    now,
                    resolution,
                    contour_level,
                    total_seq_len,
                    map_filename,
                    int(overwrite),
                    json.dumps(sequences),
                    submitted_by,
                ),
            )
            self._conn.commit()
        return self.get(job_id)  # type: ignore[return-value]

    def _update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._conn.execute(
                f"UPDATE jobs SET {assignments} WHERE id=?",
                (*fields.values(), job_id),
            )
            self._conn.commit()

    def set_status(
        self,
        job_id: str,
        target: JobStatus,
        *,
        exit_code: int | None = None,
        error_summary: str | None = None,
        enforce: bool = True,
    ) -> None:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if enforce:
            assert_transition(job.status, target)

        fields: dict[str, Any] = {"status": target.value}
        if target is JobStatus.RUNNING:
            fields["started_at"] = time.time()
        if target.is_terminal:
            fields["finished_at"] = time.time()
            fields["pid"] = None
        if exit_code is not None:
            fields["exit_code"] = exit_code
        if error_summary is not None:
            fields["error_summary"] = error_summary
        self._update(job_id, **fields)

    def set_stage(self, job_id: str, stage: str) -> None:
        self._update(job_id, stage=stage)

    def set_pid(self, job_id: str, pid: int | None) -> None:
        self._update(job_id, pid=pid)

    def set_gpu(self, job_id: str, gpu_index: int | None) -> None:
        self._update(job_id, gpu_index=gpu_index)

    def set_command(self, job_id: str, command: list[str]) -> None:
        self._update(job_id, command_json=json.dumps(command))

    # -- reads ----------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        return _row_to_job(row) if row else None

    def list(self, limit: int = 200, status: JobStatus | None = None) -> list[Job]:
        query = "SELECT * FROM jobs"
        params: list[Any] = []
        if status is not None:
            query += " WHERE status=?"
            params.append(status.value)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [_row_to_job(r) for r in rows]

    def queued_jobs(self) -> list[Job]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE status=? ORDER BY created_at ASC",
                (JobStatus.QUEUED.value,),
            ).fetchall()
        return [_row_to_job(r) for r in rows]

    def jobs_with_status(self, statuses: Iterable[JobStatus]) -> list[Job]:
        values = [s.value for s in statuses]
        if not values:
            return []
        placeholders = ",".join("?" * len(values))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM jobs WHERE status IN ({placeholders}) "
                "ORDER BY created_at ASC",
                values,
            ).fetchall()
        return [_row_to_job(r) for r in rows]

    def counts_by_status(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    # -- crash recovery -------------------------------------------------
    def mark_orphans_interrupted(self) -> list[str]:
        """Flag jobs that were RUNNING when the server stopped.

        Called once at startup. Their subprocesses are gone with the previous
        process group, so their outputs are partial; they become INTERRUPTED,
        never COMPLETED, and the operator can rerun them explicitly.
        """
        affected: list[str] = []
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM jobs WHERE status=?", (JobStatus.RUNNING.value,)
            ).fetchall()
            affected = [r["id"] for r in rows]
            if affected:
                self._conn.execute(
                    "UPDATE jobs SET status=?, finished_at=?, pid=NULL, "
                    "error_summary=? WHERE status=?",
                    (
                        JobStatus.INTERRUPTED.value,
                        time.time(),
                        "Server stopped while this job was running; "
                        "its outputs are incomplete.",
                        JobStatus.RUNNING.value,
                    ),
                )
                self._conn.commit()
        return affected
