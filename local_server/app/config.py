"""Configuration for the CryoZeta local server.

Every value is environment-overridable so that one checkout runs unmodified on
any workstation. Nothing in this module may be specific to a single host: paths
are discovered at runtime (see :mod:`app.discovery`), never baked in.

Environment variables all use the ``CRYOZETA_WEB_`` prefix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ENV_PREFIX = "CRYOZETA_WEB_"


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(ENV_PREFIX + name, default)


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_PREFIX}{name} must be an integer, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


MIB = 1024 * 1024
GIB = 1024 * MIB


@dataclass
class Settings:
    """Runtime settings.

    Defaults are chosen so that ``./start_local_server.sh`` works with no
    configuration at all on a machine where CryoZeta is already set up.
    """

    # --- Networking -------------------------------------------------------
    # Bound to loopback on purpose. Exposing the app on a LAN requires the
    # operator to *also* set CRYOZETA_WEB_ALLOW_LAN=1, which is checked in
    # main.py. There are no accounts and no authentication of any kind.
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))
    allow_lan: bool = field(default_factory=lambda: _env_bool("ALLOW_LAN", False))

    # --- Storage ----------------------------------------------------------
    data_root: Path = field(
        default_factory=lambda: Path(
            _env("DATA_ROOT", str(Path.home() / "cryozeta-web-data"))
        ).expanduser()
    )

    # --- CryoZeta location / execution -----------------------------------
    # None means "discover at runtime". Set CRYOZETA_WEB_REPO to pin it.
    cryozeta_repo: Path | None = field(
        default_factory=lambda: Path(p).expanduser() if (p := _env("REPO")) else None
    )
    # None means "let the demo scripts auto-detect from nvidia-smi".
    pixi_env: str | None = field(default_factory=lambda: _env("PIXI_ENV") or None)
    # Comma-separated GPU indices. None means "detect via nvidia-smi".
    gpus: str | None = field(default_factory=lambda: _env("GPUS") or None)

    # --- Limits (all configurable) ---------------------------------------
    max_upload_bytes: int = field(
        default_factory=lambda: _env_int("MAX_UPLOAD_MB", 4096) * MIB
    )
    max_decompressed_bytes: int = field(
        default_factory=lambda: _env_int("MAX_DECOMPRESSED_MB", 16384) * MIB
    )
    max_msa_archive_bytes: int = field(
        default_factory=lambda: _env_int("MAX_MSA_ARCHIVE_MB", 2048) * MIB
    )
    max_archive_members: int = field(
        default_factory=lambda: _env_int("MAX_ARCHIVE_MEMBERS", 10000)
    )
    # Guards against decompression bombs: refuse members whose declared
    # expansion ratio exceeds this.
    max_compression_ratio: int = field(
        default_factory=lambda: _env_int("MAX_COMPRESSION_RATIO", 200)
    )

    # --- Validation ranges ------------------------------------------------
    min_resolution: float = 0.5
    max_resolution: float = 30.0
    max_copies_per_sequence: int = 128
    max_total_sequence_length: int = field(
        default_factory=lambda: _env_int("MAX_TOTAL_SEQ_LEN", 20000)
    )

    # Large-complex recommendation threshold. Sourced from the CryoZeta
    # README ("For modeling large complexes (>2800 residues or nucleotides),
    # we recommend using the large-complex inference mode"). Overridable
    # because upstream may revise it.
    large_complex_threshold: int = field(
        default_factory=lambda: _env_int("LARGE_THRESHOLD", 2800)
    )

    # --- Process management ----------------------------------------------
    # Seconds to wait after SIGTERM to the process group before SIGKILL.
    cancel_grace_seconds: int = field(
        default_factory=lambda: _env_int("CANCEL_GRACE_SECONDS", 20)
    )

    # --- Multi-user / tailnet publishing ----------------------------------
    # When enabled, identity headers injected by `tailscale serve` are trusted,
    # but only for requests arriving from loopback (i.e. from the local proxy).
    # Off by default: an unconditionally trusted header is spoofable.
    trust_tailscale_headers: bool = field(
        default_factory=lambda: _env_bool("TRUST_TAILSCALE_HEADERS", False)
    )

    # --- Optional external MSA backend -----------------------------------
    # Off by default and must be explicitly enabled: it transmits sequences to
    # a third-party server, which contradicts local-only operation.
    allow_remote_msa: bool = field(
        default_factory=lambda: _env_bool("ALLOW_REMOTE_MSA", False)
    )

    # --- Paths derived from data_root ------------------------------------
    @property
    def jobs_dir(self) -> Path:
        return self.data_root / "jobs"

    @property
    def db_path(self) -> Path:
        return self.data_root / "cryozeta.sqlite3"

    @property
    def msa_library_dir(self) -> Path:
        """Shared, deduplicated MSA directories keyed by sequence hash."""
        return self.data_root / "msa_library"

    @property
    def run_dir(self) -> Path:
        return self.data_root / "run"

    def ensure_dirs(self) -> None:
        for path in (
            self.data_root,
            self.jobs_dir,
            self.msa_library_dir,
            self.run_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def is_loopback(self) -> bool:
        return self.host in {"127.0.0.1", "localhost", "::1"}


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings
