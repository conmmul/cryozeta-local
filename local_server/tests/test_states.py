"""Job-state transitions, persistence and crash recovery."""

from __future__ import annotations

import pytest

from app.db import JobStore
from app.states import (
    InferenceMode,
    JobStatus,
    RunMode,
    TransitionError,
    assert_transition,
    can_transition,
    stages_for,
)


def make_job(store: JobStore, **overrides):
    defaults = dict(
        entry_name="test",
        title="Test job",
        note="",
        run_mode=RunMode.STANDARD,
        inference_mode=InferenceMode.COMBINED,
        gpu_index=None,
        resolution=3.0,
        contour_level=0.3,
        total_seq_len=100,
        map_filename="emd.map",
        overwrite=False,
        sequences=[],
    )
    defaults.update(overrides)
    return store.create(**defaults)


class TestTransitions:
    @pytest.mark.parametrize(
        "current,target",
        [
            (JobStatus.QUEUED, JobStatus.RUNNING),
            (JobStatus.QUEUED, JobStatus.CANCELLED),
            (JobStatus.RUNNING, JobStatus.COMPLETED),
            (JobStatus.RUNNING, JobStatus.FAILED),
            (JobStatus.RUNNING, JobStatus.CANCELLED),
            (JobStatus.RUNNING, JobStatus.INTERRUPTED),
        ],
    )
    def test_allowed(self, current, target):
        assert can_transition(current, target)

    @pytest.mark.parametrize(
        "current,target",
        [
            (JobStatus.COMPLETED, JobStatus.RUNNING),
            (JobStatus.FAILED, JobStatus.COMPLETED),
            (JobStatus.CANCELLED, JobStatus.RUNNING),
            (JobStatus.QUEUED, JobStatus.COMPLETED),
            # The key invariant: an interrupted job must never become completed.
            (JobStatus.INTERRUPTED, JobStatus.COMPLETED),
        ],
    )
    def test_forbidden(self, current, target):
        assert not can_transition(current, target)
        with pytest.raises(TransitionError):
            assert_transition(current, target)

    def test_terminal_classification(self):
        assert JobStatus.COMPLETED.is_terminal
        assert JobStatus.INTERRUPTED.is_terminal
        assert not JobStatus.QUEUED.is_terminal
        assert JobStatus.RUNNING.is_active


class TestStages:
    def test_combined_runs_all_four(self):
        stages = stages_for(RunMode.STANDARD, InferenceMode.COMBINED)
        assert [k for k, _ in stages] == [
            "detection", "cryozeta", "cryozeta-interpolate", "combine",
        ]

    def test_cryozeta_only_skips_interpolate_and_combine(self):
        stages = stages_for(RunMode.STANDARD, InferenceMode.CRYOZETA)
        assert [k for k, _ in stages] == ["detection", "cryozeta"]

    def test_interpolate_only(self):
        stages = stages_for(RunMode.STANDARD, InferenceMode.CRYOZETA_INTERPOLATE)
        assert [k for k, _ in stages] == ["detection", "cryozeta-interpolate"]

    def test_large_mode_stages(self):
        stages = stages_for(RunMode.LARGE, InferenceMode.COMBINED)
        assert [k for k, _ in stages] == [
            "detection", "cycle-predict", "combine-stages",
        ]


class TestPersistence:
    def test_job_round_trips(self, store: JobStore):
        job = make_job(store, title="Ribosome")
        fetched = store.get(job.id)
        assert fetched.title == "Ribosome"
        assert fetched.status is JobStatus.QUEUED

    def test_survives_reopen(self, settings, store: JobStore):
        job = make_job(store)
        store.close()

        reopened = JobStore(settings.db_path)
        try:
            assert reopened.get(job.id) is not None
        finally:
            reopened.close()

    def test_illegal_transition_blocked(self, store: JobStore):
        job = make_job(store)
        with pytest.raises(TransitionError):
            store.set_status(job.id, JobStatus.COMPLETED)

    def test_timestamps_recorded(self, store: JobStore):
        job = make_job(store)
        store.set_status(job.id, JobStatus.RUNNING)
        assert store.get(job.id).started_at is not None
        store.set_status(job.id, JobStatus.COMPLETED, exit_code=0)
        finished = store.get(job.id)
        assert finished.finished_at is not None
        assert finished.exit_code == 0
        assert finished.duration_seconds is not None

    def test_queued_jobs_ordered_oldest_first(self, store: JobStore):
        first = make_job(store, title="first")
        second = make_job(store, title="second")
        assert [j.id for j in store.queued_jobs()] == [first.id, second.id]


class TestCrashRecovery:
    def test_running_jobs_become_interrupted(self, store: JobStore):
        job = make_job(store)
        store.set_status(job.id, JobStatus.RUNNING)

        affected = store.mark_orphans_interrupted()

        assert job.id in affected
        recovered = store.get(job.id)
        assert recovered.status is JobStatus.INTERRUPTED
        assert recovered.pid is None
        assert "incomplete" in recovered.error_summary

    def test_queued_jobs_are_untouched(self, store: JobStore):
        job = make_job(store)
        store.mark_orphans_interrupted()
        assert store.get(job.id).status is JobStatus.QUEUED

    def test_completed_jobs_are_untouched(self, store: JobStore):
        job = make_job(store)
        store.set_status(job.id, JobStatus.RUNNING)
        store.set_status(job.id, JobStatus.COMPLETED, exit_code=0)
        store.mark_orphans_interrupted()
        assert store.get(job.id).status is JobStatus.COMPLETED


class TestSchemaMigration:
    """Upgrading must never discard an existing job history."""

    def _make_v1_database(self, path):
        import sqlite3

        conn = sqlite3.connect(str(path))
        conn.executescript(
            """
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '',
                note TEXT NOT NULL DEFAULT '', entry_name TEXT NOT NULL,
                status TEXT NOT NULL, run_mode TEXT NOT NULL,
                inference_mode TEXT NOT NULL, gpu_index INTEGER,
                stage TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL,
                started_at REAL, finished_at REAL, exit_code INTEGER,
                error_summary TEXT NOT NULL DEFAULT '', resolution REAL NOT NULL,
                contour_level REAL NOT NULL, total_seq_len INTEGER NOT NULL DEFAULT 0,
                map_filename TEXT NOT NULL DEFAULT '', overwrite INTEGER NOT NULL DEFAULT 0,
                pid INTEGER, sequences_json TEXT NOT NULL DEFAULT '[]',
                command_json TEXT NOT NULL DEFAULT '[]'
            );
            PRAGMA user_version=1;
            """
        )
        conn.execute(
            "INSERT INTO jobs (id, entry_name, status, run_mode, inference_mode, "
            "created_at, resolution, contour_level, title) "
            "VALUES ('old-job', 'legacy', 'completed', 'standard', 'combined', "
            "1000.0, 3.2, 0.5, 'Legacy job')"
        )
        conn.commit()
        conn.close()

    def test_v1_database_upgrades_and_keeps_rows(self, tmp_path):
        db_path = tmp_path / "old.sqlite3"
        self._make_v1_database(db_path)

        store = JobStore(db_path)
        try:
            job = store.get("old-job")
            assert job is not None
            assert job.title == "Legacy job"
            assert job.status is JobStatus.COMPLETED
            # The new column exists and defaults to blank for old rows.
            assert job.submitted_by == ""
        finally:
            store.close()

    def test_migration_is_idempotent(self, tmp_path):
        db_path = tmp_path / "old.sqlite3"
        self._make_v1_database(db_path)
        for _ in range(3):
            store = JobStore(db_path)
            store.close()
        store = JobStore(db_path)
        try:
            assert store.get("old-job") is not None
        finally:
            store.close()

    def test_new_jobs_record_the_submitter(self, store: JobStore):
        job = make_job(store, submitted_by="alice@lab.edu")
        assert store.get(job.id).submitted_by == "alice@lab.edu"
