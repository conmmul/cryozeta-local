"""Filesystem and archive safety helpers.

Everything a user can influence -- filenames, archive members, job titles --
passes through this module before it touches the filesystem or a subprocess.
"""

from __future__ import annotations

import re
import tarfile
import unicodedata
import zipfile
from pathlib import Path


class SecurityError(Exception):
    """Raised when input would escape its sandbox or exhaust resources."""


# --------------------------------------------------------------------------
# Path containment
# --------------------------------------------------------------------------
def is_within(base: Path, target: Path) -> bool:
    """True if ``target`` resolves to a location inside ``base``.

    Both sides are fully resolved first, so symlinks that point outside the
    base directory are correctly rejected.
    """
    try:
        base_r = base.resolve()
        target_r = target.resolve()
    except OSError:
        return False
    return base_r == target_r or base_r in target_r.parents


def resolve_within(base: Path, relative: str) -> Path:
    """Join ``relative`` onto ``base``, refusing anything that escapes it.

    Used for serving job artifacts by relative path from the results page.
    """
    if not relative:
        raise SecurityError("empty path")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise SecurityError(f"absolute paths are not permitted: {relative!r}")
    if any(part == ".." for part in candidate.parts):
        raise SecurityError(f"parent traversal is not permitted: {relative!r}")
    target = base / candidate
    if not is_within(base, target):
        raise SecurityError(f"path escapes its job directory: {relative!r}")
    return target


# --------------------------------------------------------------------------
# Display names
# --------------------------------------------------------------------------
_UNSAFE_DISPLAY = re.compile(r"[\x00-\x1f\x7f]")


def sanitize_display_name(raw: str, max_length: int = 200) -> str:
    """Clean a user-supplied title/note for storage and display.

    This produces a *display* string only. It is never used to build a
    filesystem path -- job directories are always UUIDs.
    """
    if raw is None:
        return ""
    text = unicodedata.normalize("NFC", str(raw))
    text = _UNSAFE_DISPLAY.sub("", text)
    text = " ".join(text.split())
    return text[:max_length]


_ENTRY_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")


def safe_entry_name(raw: str, fallback: str) -> str:
    """Derive the CryoZeta ``name`` field, which *does* become a directory.

    CryoZeta creates ``<dump_dir>/<name>/`` from this value, so it is
    restricted to a conservative character set and never allowed to be empty,
    a dot-name, or overlong.
    """
    cleaned = _ENTRY_NAME_RE.sub("_", (raw or "").strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        cleaned = fallback
    return cleaned[:64]


# --------------------------------------------------------------------------
# Archive extraction
# --------------------------------------------------------------------------
def _check_member_name(name: str) -> Path:
    if not name or name in {".", "/"}:
        raise SecurityError(f"invalid archive member name: {name!r}")
    # Normalise Windows separators that zip files may legitimately contain.
    normalised = name.replace("\\", "/")
    candidate = Path(normalised)
    if candidate.is_absolute() or normalised.startswith("/"):
        raise SecurityError(f"archive member is an absolute path: {name!r}")
    if re.match(r"^[A-Za-z]:", normalised):
        raise SecurityError(f"archive member has a drive letter: {name!r}")
    if any(part == ".." for part in candidate.parts):
        raise SecurityError(f"archive member escapes the archive root: {name!r}")
    return candidate


def safe_extract_zip(
    archive_path: Path,
    dest_dir: Path,
    *,
    max_total_bytes: int,
    max_members: int,
    max_ratio: int,
) -> list[Path]:
    """Extract a ZIP archive, refusing traversal, symlinks and bombs.

    Returns the list of extracted file paths (directories excluded).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    total = 0

    with zipfile.ZipFile(archive_path) as zf:
        infos = zf.infolist()
        if len(infos) > max_members:
            raise SecurityError(
                f"archive contains {len(infos)} members, limit is {max_members}"
            )

        for info in infos:
            rel = _check_member_name(info.filename)

            # Reject symlinks and any non-regular entry. The high 16 bits of
            # external_attr hold the Unix mode for zips written on POSIX.
            mode = info.external_attr >> 16
            if mode and not (mode & 0o170000) in (0o100000, 0o040000, 0):
                raise SecurityError(
                    f"archive member is not a regular file or directory: {info.filename!r}"
                )

            if info.is_dir():
                (dest_dir / rel).mkdir(parents=True, exist_ok=True)
                continue

            total += info.file_size
            if total > max_total_bytes:
                raise SecurityError(
                    f"archive expands to more than {max_total_bytes} bytes"
                )
            if info.compress_size > 0:
                ratio = info.file_size / info.compress_size
                if ratio > max_ratio:
                    raise SecurityError(
                        f"archive member {info.filename!r} has a suspicious "
                        f"compression ratio ({ratio:.0f}:1)"
                    )

            target = dest_dir / rel
            if not is_within(dest_dir, target.parent if target.parent.exists() else dest_dir):
                raise SecurityError(f"archive member escapes destination: {info.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(info) as src, open(target, "wb") as dst:
                remaining = info.file_size
                while remaining > 0:
                    chunk = src.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    dst.write(chunk)
                    remaining -= len(chunk)
            extracted.append(target)

    return extracted


def assert_safe_tar(archive_path: Path) -> None:
    """Reject tar archives containing links or traversal.

    Provided for completeness; the UI accepts ZIP, but operators sometimes
    have ``.tar.gz`` MSA bundles and may extend the form.
    """
    with tarfile.open(archive_path) as tf:
        for member in tf.getmembers():
            _check_member_name(member.name)
            if member.issym() or member.islnk():
                raise SecurityError(f"archive contains a link: {member.name!r}")
