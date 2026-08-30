"""Job status and pipeline-stage vocabulary, plus legal transitions."""

from __future__ import annotations

from enum import Enum


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    # Set at startup for jobs that were RUNNING when the server died. They are
    # deliberately never reported as completed, because their outputs are
    # partial and untrustworthy.
    INTERRUPTED = "interrupted"

    @property
    def is_terminal(self) -> bool:
        return self in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.INTERRUPTED,
        }

    @property
    def is_active(self) -> bool:
        return self in {JobStatus.QUEUED, JobStatus.RUNNING}

    @property
    def badge(self) -> str:
        return {
            JobStatus.QUEUED: "queued",
            JobStatus.RUNNING: "running",
            JobStatus.COMPLETED: "ok",
            JobStatus.FAILED: "error",
            JobStatus.CANCELLED: "muted",
            JobStatus.INTERRUPTED: "warn",
        }[self]


# Transitions the scheduler is permitted to make. Anything else is a bug.
_ALLOWED: dict[JobStatus, set[JobStatus]] = {
    JobStatus.QUEUED: {JobStatus.RUNNING, JobStatus.CANCELLED, JobStatus.FAILED},
    JobStatus.RUNNING: {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.INTERRUPTED,
    },
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELLED: set(),
    JobStatus.INTERRUPTED: set(),
}


class TransitionError(RuntimeError):
    pass


def can_transition(current: JobStatus, target: JobStatus) -> bool:
    return target in _ALLOWED[current]


def assert_transition(current: JobStatus, target: JobStatus) -> None:
    if not can_transition(current, target):
        raise TransitionError(f"illegal job transition: {current.value} -> {target.value}")


class RunMode(str, Enum):
    STANDARD = "standard"
    LARGE = "large"

    @property
    def label(self) -> str:
        return "Standard" if self is RunMode.STANDARD else "Large / cycle"


class InferenceMode(str, Enum):
    """``-m/--mode`` accepted by inference_demo.sh."""

    COMBINED = "combined"
    CRYOZETA = "cryozeta"
    CRYOZETA_INTERPOLATE = "cryozeta-interpolate"


# Ordered pipeline stages, used to drive the run-status tracker in the UI.
STANDARD_STAGES = [
    ("detection", "Atom detection"),
    ("cryozeta", "CryoZeta inference"),
    ("cryozeta-interpolate", "CryoZeta-Interpolate"),
    ("combine", "Combine + rank"),
]

LARGE_STAGES = [
    ("detection", "Atom detection"),
    ("cycle-predict", "Cycle prediction"),
    ("combine-stages", "Combine stages"),
]


def stages_for(mode: RunMode, inference_mode: InferenceMode) -> list[tuple[str, str]]:
    """Stages actually executed for a given configuration."""
    if mode is RunMode.LARGE:
        return list(LARGE_STAGES)

    stages = [STANDARD_STAGES[0]]
    if inference_mode in (InferenceMode.COMBINED, InferenceMode.CRYOZETA):
        stages.append(STANDARD_STAGES[1])
    if inference_mode in (InferenceMode.COMBINED, InferenceMode.CRYOZETA_INTERPOLATE):
        stages.append(STANDARD_STAGES[2])
    if inference_mode is InferenceMode.COMBINED:
        stages.append(STANDARD_STAGES[3])
    return stages
