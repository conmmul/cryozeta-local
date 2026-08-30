"""Runtime discovery of the host environment.

This module is what makes the server portable: the CryoZeta checkout, the Pixi
binary, the CUDA environment and the GPU inventory are all detected on the
machine the server happens to be running on. No host-specific paths anywhere.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

# Files that together identify a CryoZeta checkout.
_REPO_MARKERS = ("inference_demo.sh", "large_inference_demo.sh", "pyproject.toml")


def looks_like_cryozeta_repo(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not all((path / marker).is_file() for marker in _REPO_MARKERS):
        return False
    try:
        text = (path / "pyproject.toml").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return 'name = "cryozeta"' in text


def find_cryozeta_repo(explicit: Path | None = None) -> Path | None:
    """Locate a CryoZeta checkout, most-specific candidate first."""
    candidates: list[Path] = []

    if explicit:
        candidates.append(Path(explicit).expanduser())
    if env := os.environ.get("CRYOZETA_REPO"):
        candidates.append(Path(env).expanduser())
    if env := os.environ.get("PIXI_PROJECT_ROOT"):
        candidates.append(Path(env).expanduser())

    # Walk up from this file: local_server/app -> local_server -> repo root,
    # checking the conventional submodule location at each level.
    here = Path(__file__).resolve()
    for parent in list(here.parents)[:5]:
        candidates.append(parent / "external" / "CryoZeta")
        candidates.append(parent / "CryoZeta")
        candidates.append(parent)

    candidates.extend(
        [
            Path.cwd() / "external" / "CryoZeta",
            Path.cwd() / "CryoZeta",
            Path.cwd(),
            Path.home() / "CryoZeta",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if looks_like_cryozeta_repo(resolved):
            return resolved
    return None


def find_pixi() -> Path | None:
    """Find the pixi binary, including its default non-PATH install location."""
    if found := shutil.which("pixi"):
        return Path(found)
    for candidate in (
        Path.home() / ".pixi" / "bin" / "pixi",
        Path("/usr/local/bin/pixi"),
        Path("/opt/pixi/bin/pixi"),
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


PIXI_INSTALL_COMMAND = "curl -fsSL https://pixi.sh/install.sh | bash"


@dataclass
class GpuInfo:
    index: int
    name: str
    memory_total_mib: int
    compute_cap: str

    @property
    def memory_total_gib(self) -> float:
        return self.memory_total_mib / 1024.0

    @property
    def compute_cap_major(self) -> int:
        try:
            return int(self.compute_cap.split(".")[0])
        except (ValueError, IndexError):
            return 0


@dataclass
class NvidiaInfo:
    available: bool
    driver_version: str | None = None
    driver_cuda_version: str | None = None
    gpus: list[GpuInfo] = field(default_factory=list)
    error: str | None = None

    @property
    def driver_cuda_major(self) -> int:
        if not self.driver_cuda_version:
            return 0
        try:
            return int(str(self.driver_cuda_version).split(".")[0])
        except ValueError:
            return 0


def _run(cmd: list[str], timeout: int = 15) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, shell=False
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"{cmd[0]}: timed out"


def query_nvidia() -> NvidiaInfo:
    """Read the GPU inventory via nvidia-smi. Never raises."""
    if shutil.which("nvidia-smi") is None:
        return NvidiaInfo(available=False, error="nvidia-smi not found on PATH")

    code, out, err = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if code != 0:
        return NvidiaInfo(available=False, error=(err or out).strip() or "nvidia-smi failed")

    gpus: list[GpuInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    memory_total_mib=int(float(parts[2])),
                    compute_cap=parts[3],
                )
            )
        except ValueError:
            continue

    # Driver version and the maximum CUDA version it supports.
    _, banner, _ = _run(["nvidia-smi"])
    driver_cuda = None
    if match := re.search(r"CUDA Version:\s*([0-9.]+)", banner):
        driver_cuda = match.group(1)
    driver_version = None
    if match := re.search(r"Driver Version:\s*([0-9.]+)", banner):
        driver_version = match.group(1)

    return NvidiaInfo(
        available=bool(gpus),
        driver_version=driver_version,
        driver_cuda_version=driver_cuda,
        gpus=gpus,
        error=None if gpus else "nvidia-smi reported no GPUs",
    )


# nvidia-smi is a subprocess, so it is cached briefly rather than run on every
# page render. It is NOT cached for the process lifetime: a driver that is not
# ready when the server boots (common when started by systemd at login) would
# otherwise leave the UI permanently reporting "no GPU".
_NVIDIA_CACHE: tuple[float, NvidiaInfo] | None = None
NVIDIA_CACHE_SECONDS = 15.0


def query_nvidia_cached(ttl: float = NVIDIA_CACHE_SECONDS) -> NvidiaInfo:
    """Return GPU info, re-probing at most every ``ttl`` seconds."""
    global _NVIDIA_CACHE
    now = time.monotonic()
    if _NVIDIA_CACHE is not None:
        cached_at, info = _NVIDIA_CACHE
        # Never serve a stale "no GPU" result: if the last probe failed, retry
        # immediately so a late-starting driver is picked up.
        if info.available and (now - cached_at) < ttl:
            return info
        if not info.available and (now - cached_at) < 2.0:
            return info
    info = query_nvidia()
    _NVIDIA_CACHE = (now, info)
    return info


def select_pixi_env(info: NvidiaInfo) -> str:
    """Mirror ``detect_pixi_env`` from CryoZeta's demo scripts exactly.

    Architecture preference: cc >= 10 -> cu13, cc >= 8 -> default (cu12),
    otherwise cu11; each additionally gated on the driver-supported CUDA major.
    """
    if not info.available or not info.gpus:
        return "default"
    driver_major = info.driver_cuda_major
    cc_major = max(g.compute_cap_major for g in info.gpus)
    if not driver_major or not cc_major:
        return "default"
    if cc_major >= 10 and driver_major >= 13:
        return "cu13"
    if cc_major >= 8 and driver_major >= 12:
        return "default"
    if driver_major >= 11:
        return "cu11"
    return "default"


# --------------------------------------------------------------------------
# CryoZeta installation state
# --------------------------------------------------------------------------
CHECKPOINTS = (
    "cryozeta-detection-v0.0.1.safetensors",
    "cryozeta-v0.0.1.safetensors",
    "cryozeta-interpolate-v0.0.1.safetensors",
)


@dataclass
class InstallState:
    repo: Path | None
    assets_dir: Path | None = None
    missing_checkpoints: list[str] = field(default_factory=list)
    has_examples: bool = False
    teaser_built: bool = False
    pixi_envs: list[str] = field(default_factory=list)

    @property
    def assets_ok(self) -> bool:
        return self.assets_dir is not None and not self.missing_checkpoints


def inspect_install(repo: Path | None, assets_override: str | None = None) -> InstallState:
    """Report which parts of the CryoZeta setup are present."""
    if repo is None:
        return InstallState(repo=None)

    assets_dir = Path(assets_override).expanduser() if assets_override else repo / "assets"
    state = InstallState(repo=repo, assets_dir=assets_dir if assets_dir.is_dir() else None)

    for name in CHECKPOINTS:
        if not (assets_dir / name).is_file():
            state.missing_checkpoints.append(name)

    state.has_examples = (assets_dir / "examples" / "example.json").is_file()

    teaser_lib = repo / "externals" / "TEASER-plusplus" / "build" / "libteaser.so"
    state.teaser_built = teaser_lib.is_file()

    envs_dir = repo / ".pixi" / "envs"
    if envs_dir.is_dir():
        state.pixi_envs = sorted(p.name for p in envs_dir.iterdir() if p.is_dir())

    return state
