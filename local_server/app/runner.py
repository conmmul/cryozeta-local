"""Construction of the CryoZeta invocation.

Commands are built as argument *arrays* and executed with ``shell=False``, so
no user-supplied value is ever parsed by a shell. The flags below are taken
verbatim from the current demo scripts:

``inference_demo.sh``
    ``-e/--env  -g/--gpu  -i/--input-json  -o/--output-dir  -m/--mode
    --checkpoint  --interp-checkpoint  --overwrite``

``large_inference_demo.sh``
    ``-e/--env  -g/--gpu  -x/--example  -r/--registration  -i/--input-json
    -o/--output-dir  --checkpoint  --detection-checkpoint``

Note that the large script has **no** ``--overwrite`` flag: it always passes
``--overwrite`` to the detection step itself. The web UI therefore only offers
the overwrite control for standard jobs.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .paths import JobPaths
from .states import InferenceMode, RunMode

STANDARD_SCRIPT = "inference_demo.sh"
LARGE_SCRIPT = "large_inference_demo.sh"

# Registration methods accepted by large_inference_demo.sh -r.
REGISTRATION_METHODS = ("auto", "teaser", "svd", "vesper")


class RunnerError(RuntimeError):
    pass


def build_command(
    *,
    repo: Path,
    paths: JobPaths,
    run_mode: RunMode,
    inference_mode: InferenceMode,
    gpu_index: int,
    pixi_env: str | None = None,
    overwrite: bool = False,
    registration: str = "auto",
) -> list[str]:
    """Return the argv array for this job."""
    repo = Path(repo)
    script = repo / (LARGE_SCRIPT if run_mode is RunMode.LARGE else STANDARD_SCRIPT)
    if not script.is_file():
        raise RunnerError(f"CryoZeta script not found: {script}")

    # The demo scripts use bash arrays and BASH_SOURCE, so they must be run
    # with bash even though their usage text says "sh".
    cmd: list[str] = [
        "bash",
        str(script),
        "-i",
        str(paths.input_json),
        "-o",
        str(paths.output_dir),
        "-g",
        str(int(gpu_index)),
    ]

    if pixi_env:
        cmd += ["-e", pixi_env]

    if run_mode is RunMode.LARGE:
        if registration not in REGISTRATION_METHODS:
            raise RunnerError(f"unknown registration method: {registration!r}")
        # Our generated JSON always holds exactly one entry, so index 0 is
        # correct. (It is also the script's default, but being explicit
        # documents the intent and survives an upstream default change.)
        cmd += ["-x", "0", "-r", registration]
    else:
        cmd += ["-m", inference_mode.value]
        if overwrite:
            cmd.append("--overwrite")

    return cmd


def build_environment(
    *, gpu_index: int, extra_cache_dir: Path | None = None
) -> dict[str, str]:
    """Environment for the subprocess.

    ``CUDA_VISIBLE_DEVICES`` is set here as a second layer of GPU isolation:
    the demo scripts already set it per-command, but pinning it for the whole
    process group means nothing the pipeline spawns can wander onto another
    GPU that a concurrent job owns.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_index))
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Unbuffered Python output so the live log tail is actually live.
    env["PYTHONUNBUFFERED"] = "1"
    if extra_cache_dir is not None:
        # Documented escape hatch for read-only / shared installs.
        env["CRYOZETA_TORCH_EXTENSIONS_DIR"] = str(extra_cache_dir)
    return env


# --------------------------------------------------------------------------
# Log interpretation
# --------------------------------------------------------------------------
_STAGE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"cryozeta-detection|Running detection", re.I), "detection"),
    (re.compile(r"cryozeta-cycle-predict|cycle prediction", re.I), "cycle-predict"),
    (re.compile(r"cryozeta-combine-stages|Combining stages", re.I), "combine-stages"),
    (re.compile(r"use_interpolation\s+true", re.I), "cryozeta-interpolate"),
    (re.compile(r"use_interpolation\s+false", re.I), "cryozeta"),
    (re.compile(r"cryozeta-combine\b", re.I), "combine"),
]


def detect_stage(line: str) -> str | None:
    """Best-effort mapping from a log line to a pipeline stage."""
    for pattern, stage in _STAGE_PATTERNS:
        if pattern.search(line):
            return stage
    return None


# Ordered most-specific first: the first match wins.
_ERROR_HINTS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"CUDA out of memory|torch\.cuda\.OutOfMemoryError", re.I),
        "The GPU ran out of memory. Try a GPU with more VRAM, reduce the "
        "complex size, or use large/cycle mode.",
    ),
    (
        re.compile(r"No pairing-MSA of", re.I),
        "A required pairing MSA (uniref100_hits.a3m) was missing. Complexes "
        "with two or more distinct protein sequences need it for every "
        "protein chain.",
    ),
    (
        re.compile(r"precomputed MSA path .* does not exists", re.I),
        "A precomputed MSA directory referenced in the input JSON no longer "
        "exists on disk.",
    ),
    (
        re.compile(r"contour_level must be provided", re.I),
        "The contour level was missing from the generated input JSON.",
    ),
    (
        re.compile(r"checkpoint not found|Detection checkpoint not found", re.I),
        "A model checkpoint is missing. Run 'pixi run download-assets' in the "
        "CryoZeta repository.",
    ),
    (
        re.compile(r"libteaser|TEASER", re.I),
        "TEASER++ appears to be missing or unbuilt. Run 'pixi run build-teaser'.",
    ),
    (
        re.compile(r"No space left on device", re.I),
        "The disk filled up. Free space, or point TMPDIR and "
        "CRYOZETA_TORCH_EXTENSIONS_DIR at a larger volume.",
    ),
    (
        re.compile(r"ninja: build stopped|error: command .*(gcc|g\+\+|nvcc)", re.I),
        "A CUDA/C++ extension failed to compile. Check that the selected Pixi "
        "environment matches your driver's CUDA version.",
    ),
    (
        re.compile(r"CUDA driver version is insufficient|no kernel image", re.I),
        "The CUDA build does not match this GPU or driver. Try a different "
        "Pixi environment (cu11 / default / cu13).",
    ),
    (
        re.compile(r"pixi: command not found|could not find pixi", re.I),
        "The pixi binary was not found. Install pixi or set its location.",
    ),
    (
        re.compile(r"environment .* not found|--frozen", re.I),
        "The Pixi environment is not installed. Run 'pixi install' for the "
        "environment this job selected.",
    ),
]


def summarize_failure(log_text: str, exit_code: int | None) -> str:
    """Produce a short, human-readable explanation of a failed run."""
    if not log_text:
        return f"Job failed with exit code {exit_code}." if exit_code else "Job failed."

    tail = log_text[-20000:]
    for pattern, hint in _ERROR_HINTS:
        if pattern.search(tail):
            return hint

    # Fall back to the last non-empty line that looks like an error.
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    for line in reversed(lines[-40:]):
        if re.search(r"error|exception|traceback|failed", line, re.I):
            return line[:300]
    if lines:
        return f"Exit code {exit_code}. Last output: {lines[-1][:250]}"
    return f"Job failed with exit code {exit_code}."
