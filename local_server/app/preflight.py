"""Read-only environment preflight.

Answers, for whatever machine it is run on: is there a usable GPU, is Pixi
installed, which CUDA environment should be used, and is CryoZeta actually set
up. Nothing here mutates the system.
"""

from __future__ import annotations

import concurrent.futures
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings, get_settings
from .discovery import (
    PIXI_INSTALL_COMMAND,
    InstallState,
    NvidiaInfo,
    find_cryozeta_repo,
    find_pixi,
    inspect_install,
    query_nvidia_cached,
    select_pixi_env,
)

# CryoZeta README: "CUDA-capable GPU with 32 GB memory or more".
MIN_VRAM_GIB = 32.0

OK = "ok"
WARN = "warn"
FAIL = "fail"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    remedy: str = ""

    @property
    def symbol(self) -> str:
        return {OK: "PASS", WARN: "WARN", FAIL: "FAIL"}[self.status]


@dataclass
class PreflightReport:
    checks: list[Check] = field(default_factory=list)
    nvidia: NvidiaInfo | None = None
    install: InstallState | None = None
    repo: Path | None = None
    pixi_path: Path | None = None
    recommended_env: str = "default"

    @property
    def ok(self) -> bool:
        return not any(c.status == FAIL for c in self.checks)

    @property
    def gpu_indices(self) -> list[int]:
        if not self.nvidia:
            return []
        return [g.index for g in self.nvidia.gpus]

    def add(self, name: str, status: str, detail: str, remedy: str = "") -> None:
        self.checks.append(Check(name, status, detail, remedy))


def _with_timeout(fn, seconds: float, default=None):
    """Run ``fn`` in a worker thread and give up after ``seconds``.

    Guards against calls that can block indefinitely rather than fail --
    notably ``shutil.disk_usage`` on a stale NFS mount, which is a realistic
    condition on a shared lab workstation and would otherwise hang the page.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(fn)
        return future.result(timeout=seconds)
    except concurrent.futures.TimeoutError:
        return default
    finally:
        # Do not wait: if the call is wedged, the thread is unreclaimable.
        executor.shutdown(wait=False)


def _guard(report: PreflightReport, name: str, fn, default=None):
    """Run one probe in isolation.

    The System page exists to diagnose broken machines, so it must not itself
    break on one. A probe that raises (hung NFS mount, permission error, an
    nvidia-smi that returns something unparseable) becomes a FAIL row and the
    rest of the report still renders.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - deliberately broad
        report.add(
            name,
            FAIL,
            f"this check could not run: {type(exc).__name__}: {exc}",
            "The remaining checks below are still valid.",
        )
        return default


def run_preflight(settings: Settings | None = None) -> PreflightReport:
    settings = settings or get_settings()
    report = PreflightReport()

    # -- Platform -------------------------------------------------------
    import platform

    system = platform.system()
    if system != "Linux":
        report.add(
            "Operating system",
            FAIL,
            f"{system} {platform.release()} ({platform.machine()})",
            "CryoZeta supports Linux only: its Pixi workspace declares "
            "platforms = [\"linux-64\"] and requires glibc >= 2.31.",
        )
    else:
        report.add("Operating system", OK, f"Linux {platform.release()} ({platform.machine()})")

    # -- GPU ------------------------------------------------------------
    nvidia = _guard(report, "NVIDIA GPU", query_nvidia_cached) or NvidiaInfo(
        available=False, error="GPU probe failed"
    )
    report.nvidia = nvidia
    if not nvidia.available:
        report.add(
            "NVIDIA GPU",
            FAIL,
            nvidia.error or "no NVIDIA GPU detected",
            "CryoZeta requires a CUDA-capable NVIDIA GPU. Verify with nvidia-smi.",
        )
    else:
        for gpu in nvidia.gpus:
            status = OK if gpu.memory_total_gib >= MIN_VRAM_GIB else WARN
            report.add(
                f"GPU {gpu.index}",
                status,
                f"{gpu.name}, {gpu.memory_total_gib:.1f} GiB VRAM, "
                f"compute capability {gpu.compute_cap}",
                ""
                if status == OK
                else f"CryoZeta documents a {MIN_VRAM_GIB:.0f} GB minimum; "
                "large complexes will run out of memory.",
            )
        report.add(
            "NVIDIA driver",
            OK,
            f"driver {nvidia.driver_version}, supports CUDA up to "
            f"{nvidia.driver_cuda_version}",
        )

    report.recommended_env = select_pixi_env(nvidia)
    report.add(
        "CUDA environment",
        OK if nvidia.available else WARN,
        f"recommended Pixi environment: {report.recommended_env}",
        ""
        if nvidia.available
        else "Defaulted because no GPU was detected; verify on the target host.",
    )

    # -- Pixi -----------------------------------------------------------
    pixi = _guard(report, "Pixi", find_pixi)
    report.pixi_path = pixi
    if pixi is None:
        report.add(
            "Pixi",
            FAIL,
            "pixi not found on PATH or in ~/.pixi/bin",
            f"Install it with: {PIXI_INSTALL_COMMAND}",
        )
    else:
        report.add("Pixi", OK, str(pixi))

    # -- CryoZeta repository --------------------------------------------
    repo = _guard(
        report, "CryoZeta repository", lambda: find_cryozeta_repo(settings.cryozeta_repo)
    )
    report.repo = repo
    if repo is None:
        report.add(
            "CryoZeta repository",
            FAIL,
            "no CryoZeta checkout found",
            "Clone it, or set CRYOZETA_WEB_REPO=/path/to/CryoZeta.",
        )
        report.install = InstallState(repo=None)
        return report

    report.add("CryoZeta repository", OK, str(repo))

    install = _guard(
        report, "CryoZeta installation", lambda: inspect_install(repo)
    ) or InstallState(repo=repo)
    report.install = install

    if install.assets_ok:
        report.add("Model weights", OK, f"all checkpoints present in {install.assets_dir}")
    else:
        missing = ", ".join(install.missing_checkpoints) or "assets/ directory"
        report.add(
            "Model weights",
            FAIL,
            f"missing: {missing}",
            "Run 'pixi run download-assets' in the CryoZeta repository.",
        )

    report.add(
        "Bundled example",
        OK if install.has_examples else WARN,
        "assets/examples/example.json present"
        if install.has_examples
        else "assets/examples/example.json not found",
        "" if install.has_examples else "Downloaded by 'pixi run download-assets'.",
    )

    report.add(
        "TEASER++",
        OK if install.teaser_built else FAIL,
        "libteaser.so present" if install.teaser_built else "libteaser.so not built",
        "" if install.teaser_built else "Run 'pixi run build-teaser'.",
    )

    if install.pixi_envs:
        have = report.recommended_env in install.pixi_envs
        report.add(
            "Pixi environments",
            OK if have else WARN,
            f"installed: {', '.join(install.pixi_envs)}",
            ""
            if have
            else f"The recommended environment '{report.recommended_env}' is not "
            f"installed. Run: pixi install -e {report.recommended_env}",
        )
    else:
        report.add(
            "Pixi environments",
            FAIL,
            "no environments installed under .pixi/envs",
            "Run 'pixi run setup' in the CryoZeta repository.",
        )

    # -- Disk -----------------------------------------------------------
    # Read-only: never create the data root here. Walk up to the nearest
    # existing ancestor so the free-space figure is still meaningful.
    probe = settings.data_root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = _with_timeout(lambda: shutil.disk_usage(probe), seconds=5.0)
    if usage is None:
        report.add(
            "Data root free space",
            WARN,
            f"could not stat {probe} within 5s",
            "The filesystem may be unresponsive (for example a stale NFS mount).",
        )
    else:
        free_gib = usage.free / (1024**3)
        suffix = "" if probe == settings.data_root else f" (volume holding {probe})"
        report.add(
            "Data root free space",
            OK if free_gib >= 50 else WARN,
            f"{free_gib:.1f} GiB free for {settings.data_root}{suffix}",
            "" if free_gib >= 50 else "Jobs can produce many GB; consider more space.",
        )

    return report


def format_report(report: PreflightReport) -> str:
    lines = ["CryoZeta local server -- preflight", "=" * 60]
    for check in report.checks:
        lines.append(f"[{check.symbol}] {check.name}: {check.detail}")
        if check.remedy:
            lines.append(f"        -> {check.remedy}")
    lines.append("=" * 60)
    lines.append(
        "Result: READY" if report.ok else "Result: NOT READY (see FAIL items above)"
    )
    return "\n".join(lines)
