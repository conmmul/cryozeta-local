"""Local background execution: one worker thread per configured GPU.

Scheduling rules:

* a worker owns exactly one GPU and runs at most one job at a time, so two
  jobs can never share a GPU
* separate GPUs run jobs concurrently
* a job may pin itself to a GPU, or leave it unset and take the first free one
* the child is started in its own session (``start_new_session=True``) so that
  cancellation can signal the entire process group -- CryoZeta spawns pixi,
  which spawns Python, which may spawn dataloader workers
"""

from __future__ import annotations

import errno
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

from .config import Settings
from .db import Job, JobStore
from .paths import job_paths
from .runner import (
    RunnerError,
    build_command,
    build_environment,
    detect_stage,
    summarize_failure,
)
from .states import InferenceMode, JobStatus, RunMode


@dataclass
class RunningProcess:
    job_id: str
    popen: subprocess.Popen
    cancelled: bool = False


class JobManager:
    """Owns the worker threads and the map of in-flight processes."""

    def __init__(
        self,
        settings: Settings,
        store: JobStore,
        repo: Path | None,
        gpu_indices: list[int],
        *,
        pixi_env: str | None = None,
        poll_interval: float = 1.0,
        command_builder: Callable[..., list[str]] | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.repo = repo
        self.gpu_indices = gpu_indices or [0]
        self.pixi_env = pixi_env
        self.poll_interval = poll_interval
        self._build_command = command_builder or build_command

        self._claim_lock = threading.Lock()
        self._processes: dict[str, RunningProcess] = {}
        self._processes_lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        self.store.mark_orphans_interrupted()
        for gpu in self.gpu_indices:
            thread = threading.Thread(
                target=self._worker_loop, args=(gpu,), name=f"gpu-{gpu}", daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def shutdown(self, timeout: float = 10.0) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=timeout)

    # -- claiming -------------------------------------------------------
    def _claim(self, gpu: int) -> Job | None:
        """Atomically take the oldest eligible queued job for this GPU."""
        with self._claim_lock:
            for job in self.store.queued_jobs():
                if job.gpu_index is not None and job.gpu_index != gpu:
                    continue
                # Re-read under the lock: a cancel may have landed meanwhile.
                current = self.store.get(job.id)
                if current is None or current.status is not JobStatus.QUEUED:
                    continue
                self.store.set_gpu(job.id, gpu)
                self.store.set_status(job.id, JobStatus.RUNNING)
                return self.store.get(job.id)
        return None

    def _worker_loop(self, gpu: int) -> None:
        while not self._stop.is_set():
            job = self._claim(gpu)
            if job is None:
                self._stop.wait(self.poll_interval)
                continue
            try:
                self._execute(job, gpu)
            except Exception as exc:  # a worker thread must never die
                self.store.set_status(
                    job.id,
                    JobStatus.FAILED,
                    exit_code=None,
                    error_summary=f"Internal scheduler error: {exc}",
                    enforce=False,
                )

    # -- execution ------------------------------------------------------
    def _execute(self, job: Job, gpu: int) -> None:
        paths = job_paths(self.settings.jobs_dir, job.id)
        paths.ensure()

        if self.repo is None:
            self.store.set_status(
                job.id,
                JobStatus.FAILED,
                error_summary=(
                    "No CryoZeta repository was found on this machine. Set "
                    "CRYOZETA_WEB_REPO to its location."
                ),
                enforce=False,
            )
            return

        try:
            cmd = self._build_command(
                repo=self.repo,
                paths=paths,
                run_mode=RunMode(job.run_mode),
                inference_mode=InferenceMode(job.inference_mode),
                gpu_index=gpu,
                pixi_env=self.pixi_env,
                overwrite=job.overwrite,
            )
        except RunnerError as exc:
            self.store.set_status(
                job.id, JobStatus.FAILED, error_summary=str(exc), enforce=False
            )
            return

        self.store.set_command(job.id, cmd)
        env = build_environment(gpu_index=gpu)

        with open(paths.log_file, "a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n{'=' * 72}\n")
            log.write(f"CryoZeta local server -- job {job.id}\n")
            log.write(f"started : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"gpu     : {gpu}\n")
            log.write(f"command : {' '.join(cmd)}\n")
            log.write(f"{'=' * 72}\n")
            log.flush()

            try:
                proc = subprocess.Popen(  # noqa: S603 - argv array, shell=False
                    cmd,
                    cwd=str(self.repo),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    shell=False,
                    start_new_session=True,
                )
            except OSError as exc:
                log.write(f"\nfailed to start process: {exc}\n")
                self.store.set_status(
                    job.id,
                    JobStatus.FAILED,
                    error_summary=f"Could not start CryoZeta: {exc}",
                    enforce=False,
                )
                return

            record = RunningProcess(job_id=job.id, popen=proc)
            with self._processes_lock:
                self._processes[job.id] = record
            self.store.set_pid(job.id, proc.pid)

            current_stage = ""
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if (stage := detect_stage(line)) and stage != current_stage:
                    current_stage = stage
                    self.store.set_stage(job.id, stage)

            exit_code = proc.wait()
            log.write(f"\n--- process exited with code {exit_code} ---\n")

        with self._processes_lock:
            self._processes.pop(job.id, None)

        self._finalise(job.id, exit_code, record.cancelled, paths.log_file)

    def _finalise(
        self, job_id: str, exit_code: int, cancelled: bool, log_file: Path
    ) -> None:
        if cancelled:
            self.store.set_status(
                job_id,
                JobStatus.CANCELLED,
                exit_code=exit_code,
                error_summary="Cancelled by the operator.",
                enforce=False,
            )
            return

        if exit_code == 0:
            self.store.set_status(
                job_id, JobStatus.COMPLETED, exit_code=0, error_summary="", enforce=False
            )
            self.store.set_stage(job_id, "done")
            return

        try:
            log_text = log_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            log_text = ""
        self.store.set_status(
            job_id,
            JobStatus.FAILED,
            exit_code=exit_code,
            error_summary=summarize_failure(log_text, exit_code),
            enforce=False,
        )

    # -- cancellation ---------------------------------------------------
    def cancel(self, job_id: str) -> bool:
        """Cancel a queued or running job.

        Running jobs get SIGTERM to the whole process group, a grace period,
        then SIGKILL. Killing the group matters: terminating only the bash
        wrapper would orphan the Python process still holding the GPU.
        """
        job = self.store.get(job_id)
        if job is None:
            return False

        if job.status is JobStatus.QUEUED:
            self.store.set_status(
                job_id,
                JobStatus.CANCELLED,
                error_summary="Cancelled before it started.",
                enforce=False,
            )
            return True

        if job.status is not JobStatus.RUNNING:
            return False

        with self._processes_lock:
            record = self._processes.get(job_id)

        if record is None:
            # Running per the database but not owned by this process: it
            # belongs to a previous server lifetime.
            self.store.set_status(
                job_id,
                JobStatus.INTERRUPTED,
                error_summary="No live process for this job in the current server.",
                enforce=False,
            )
            return True

        record.cancelled = True
        _terminate_group(record.popen, self.settings.cancel_grace_seconds)
        return True

    def running_job_ids(self) -> set[str]:
        with self._processes_lock:
            return set(self._processes)


def _terminate_group(proc: subprocess.Popen, grace_seconds: int) -> None:
    """SIGTERM the process group, escalating to SIGKILL only if needed."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return

    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError as exc:
        if exc.errno != errno.ESRCH:
            raise
        return

    deadline = time.monotonic() + max(1, grace_seconds)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.25)

    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError as exc:
        if exc.errno != errno.ESRCH:
            raise
