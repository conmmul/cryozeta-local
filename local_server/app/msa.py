"""MSA sourcing and validation.

CryoZeta does not bundle MSA generation, so the first working version accepts
*precomputed* alignments only: either a ZIP upload or a directory that already
exists on the server. The :class:`MSAProvider` interface exists so a
ColabFold/MMseqs2 backend can be dropped in later without touching the job
pipeline.

Required filenames are taken from ``src/cryozeta/data/msa_featurizer.py``
rather than from prose documentation, because the code is stricter than the
README:

* protein, always: ``mmseqs_other_hits.a3m``
* protein, only when the complex has >= 2 *distinct* protein sequences:
  ``uniref100_hits.a3m`` (enforced by a bare ``assert``)
* RNA: ``rnacentral.a3m`` (RNA is always treated as homomer/monomer, so it
  never needs a pairing MSA)
* DNA: no MSA at all
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .security import SecurityError, safe_extract_zip
from .sequences import SeqType

PROTEIN_NON_PAIRING = "mmseqs_other_hits.a3m"
PROTEIN_PAIRING = "uniref100_hits.a3m"
RNA_NON_PAIRING = "rnacentral.a3m"

DEFAULT_PAIRING_DB = "uniref100"


class MSAError(ValueError):
    """Raised when required MSA files are missing or unusable."""


def required_filenames(seq_type: SeqType, needs_pairing: bool) -> list[str]:
    """Filenames CryoZeta will actually open for this chain type."""
    if seq_type is SeqType.DNA:
        return []
    if seq_type is SeqType.RNA:
        return [RNA_NON_PAIRING]
    names = [PROTEIN_NON_PAIRING]
    if needs_pairing:
        names.append(PROTEIN_PAIRING)
    return names


def _looks_like_a3m(path: Path) -> bool:
    """Cheap sanity check that a file is a non-empty alignment."""
    try:
        if path.stat().st_size == 0:
            return False
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.strip():
                    return line.lstrip().startswith(">")
    except OSError:
        return False
    return False


@dataclass
class MSAValidation:
    directory: Path
    seq_type: SeqType
    needs_pairing: bool
    missing: list[str]
    malformed: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.malformed

    def message(self) -> str:
        parts = []
        if self.missing:
            parts.append(f"missing required file(s): {', '.join(self.missing)}")
        if self.malformed:
            parts.append(
                f"file(s) present but not valid non-empty A3M: {', '.join(self.malformed)}"
            )
        return "; ".join(parts)


def find_msa_root(directory: Path, seq_type: SeqType, needs_pairing: bool) -> Path:
    """Locate the directory that actually holds the .a3m files.

    People commonly zip a folder, producing ``archive/msa/uniref100_hits.a3m``.
    Rather than reject that, descend into a unique subdirectory chain until the
    expected files appear.
    """
    required = required_filenames(seq_type, needs_pairing)
    if not required:
        return directory

    primary = required[0]
    if (directory / primary).is_file():
        return directory

    matches = sorted(directory.rglob(primary))
    # Ignore macOS resource forks that ZIP archives frequently carry.
    matches = [m for m in matches if "__MACOSX" not in m.parts]
    if len(matches) >= 1:
        return matches[0].parent
    return directory


def validate_msa_dir(
    directory: Path, seq_type: SeqType, needs_pairing: bool
) -> MSAValidation:
    """Check a directory against CryoZeta's real expectations."""
    missing: list[str] = []
    malformed: list[str] = []

    if not directory.is_dir():
        return MSAValidation(
            directory=directory,
            seq_type=seq_type,
            needs_pairing=needs_pairing,
            missing=required_filenames(seq_type, needs_pairing) or ["<directory>"],
            malformed=[],
        )

    for name in required_filenames(seq_type, needs_pairing):
        path = directory / name
        if not path.is_file():
            missing.append(name)
        elif not _looks_like_a3m(path):
            malformed.append(name)

    return MSAValidation(
        directory=directory,
        seq_type=seq_type,
        needs_pairing=needs_pairing,
        missing=missing,
        malformed=malformed,
    )


# --------------------------------------------------------------------------
# Provider interface
# --------------------------------------------------------------------------
class MSAProvider(ABC):
    """Supplies a directory of precomputed alignments for one chain.

    Implementations must return a directory that passes :func:`validate_msa_dir`
    or raise :class:`MSAError`.
    """

    name: str = "abstract"

    @abstractmethod
    def provide(
        self,
        *,
        sequence: str,
        seq_type: SeqType,
        needs_pairing: bool,
        destination: Path,
    ) -> Path:
        """Materialise alignments for ``sequence`` and return their directory."""

    def _finalise(
        self, directory: Path, seq_type: SeqType, needs_pairing: bool
    ) -> Path:
        root = find_msa_root(directory, seq_type, needs_pairing)
        result = validate_msa_dir(root, seq_type, needs_pairing)
        if not result.ok:
            raise MSAError(
                f"MSA directory for this {seq_type.label} chain is unusable: "
                f"{result.message()}"
            )
        return root


class UploadedArchiveProvider(MSAProvider):
    """Extracts a user-uploaded ZIP archive of precomputed alignments."""

    name = "upload"

    def __init__(
        self,
        archive_path: Path,
        *,
        max_total_bytes: int,
        max_members: int,
        max_ratio: int,
    ) -> None:
        self.archive_path = archive_path
        self.max_total_bytes = max_total_bytes
        self.max_members = max_members
        self.max_ratio = max_ratio

    def provide(
        self,
        *,
        sequence: str,
        seq_type: SeqType,
        needs_pairing: bool,
        destination: Path,
    ) -> Path:
        try:
            safe_extract_zip(
                self.archive_path,
                destination,
                max_total_bytes=self.max_total_bytes,
                max_members=self.max_members,
                max_ratio=self.max_ratio,
            )
        except SecurityError as exc:
            raise MSAError(f"rejected MSA archive: {exc}") from exc
        except Exception as exc:  # zipfile.BadZipFile and friends
            raise MSAError(f"could not read MSA archive: {exc}") from exc
        return self._finalise(destination, seq_type, needs_pairing)


class LocalDirectoryProvider(MSAProvider):
    """Uses a directory that already exists on the server.

    The contents are copied into the job directory so that results remain
    reproducible even if the original directory is later modified or deleted.
    """

    name = "local-directory"

    def __init__(self, source: Path, *, copy: bool = True) -> None:
        self.source = Path(source).expanduser()
        self.copy = copy

    def provide(
        self,
        *,
        sequence: str,
        seq_type: SeqType,
        needs_pairing: bool,
        destination: Path,
    ) -> Path:
        if not self.source.is_dir():
            raise MSAError(f"MSA directory does not exist on the server: {self.source}")

        root = find_msa_root(self.source, seq_type, needs_pairing)
        # Validate before copying so a wrong path fails fast and cheaply.
        result = validate_msa_dir(root, seq_type, needs_pairing)
        if not result.ok:
            raise MSAError(
                f"MSA directory for this {seq_type.label} chain is unusable: "
                f"{result.message()}"
            )

        if not self.copy:
            return root

        destination.mkdir(parents=True, exist_ok=True)
        for name in required_filenames(seq_type, needs_pairing):
            shutil.copy2(root / name, destination / name)
        return self._finalise(destination, seq_type, needs_pairing)


class RemoteColabFoldProvider(MSAProvider):
    """Placeholder for a future ColabFold/MMseqs2 MSA backend.

    Deliberately not implemented. Generating alignments through ColabFold's
    public endpoint transmits the query sequence to a third party, which
    conflicts with this server's local-only guarantee, so it must remain an
    explicit opt-in rather than a silent fallback.
    """

    name = "colabfold-remote"

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def provide(
        self,
        *,
        sequence: str,
        seq_type: SeqType,
        needs_pairing: bool,
        destination: Path,
    ) -> Path:
        raise MSAError(
            "Automatic MSA generation is not available. CryoZeta does not bundle "
            "an MSA pipeline, and this server does not contact external services. "
            "Supply precomputed alignments instead."
        )
