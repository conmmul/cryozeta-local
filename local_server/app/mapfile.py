"""Validation of uploaded cryo-EM density maps.

Checks are ordered cheapest-first so an obviously wrong upload is rejected
before anything is decompressed.
"""

from __future__ import annotations

import gzip
import shutil
from dataclasses import dataclass
from pathlib import Path

ALLOWED_SUFFIXES = (".mrc", ".map", ".mrc.gz", ".map.gz")

# MRC/CCP4 files carry their format tag at byte offset 208.
_MRC_STAMP_OFFSET = 208
_MRC_HEADER_BYTES = 1024


class MapError(ValueError):
    """Raised when an uploaded map is unusable."""


@dataclass
class MapInfo:
    path: Path
    original_name: str
    is_gzipped: bool
    size_bytes: int
    shape: tuple[int, int, int] | None = None
    voxel_size: tuple[float, float, float] | None = None


def classify_suffix(filename: str) -> str:
    """Return the matched allowed suffix, or raise."""
    lowered = filename.lower()
    # Longest first so ".map.gz" wins over ".gz"/".map".
    for suffix in sorted(ALLOWED_SUFFIXES, key=len, reverse=True):
        if lowered.endswith(suffix):
            return suffix
    raise MapError(
        f"unsupported map file type: {filename!r}. "
        f"Accepted extensions: {', '.join(ALLOWED_SUFFIXES)}"
    )


def is_gzipped_name(filename: str) -> bool:
    return filename.lower().endswith(".gz")


def sniff_gzip(path: Path) -> bool:
    """Detect gzip by magic number rather than trusting the filename."""
    try:
        with open(path, "rb") as fh:
            return fh.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def decompress_if_needed(
    path: Path, destination: Path, *, max_decompressed_bytes: int
) -> Path:
    """Gunzip ``path`` to ``destination``, enforcing a decompressed size cap.

    The cap is enforced while streaming, so a decompression bomb is stopped
    partway rather than after filling the disk.
    """
    if not sniff_gzip(path):
        return path

    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with gzip.open(path, "rb") as src, open(destination, "wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_decompressed_bytes:
                    dst.close()
                    destination.unlink(missing_ok=True)
                    raise MapError(
                        f"map expands beyond the configured limit of "
                        f"{max_decompressed_bytes} bytes"
                    )
                dst.write(chunk)
    except MapError:
        raise
    except (OSError, EOFError) as exc:
        destination.unlink(missing_ok=True)
        raise MapError(f"could not decompress map: {exc}") from exc
    return destination


def check_readable(path: Path) -> MapInfo:
    """Confirm the file really is an MRC/CCP4 map.

    Uses ``mrcfile`` when available (it ships in CryoZeta's own dependency
    set); otherwise falls back to checking the MAP stamp in the header.
    """
    gz = sniff_gzip(path)
    size = path.stat().st_size
    info = MapInfo(
        path=path, original_name=path.name, is_gzipped=gz, size_bytes=size
    )

    try:
        import mrcfile  # noqa: PLC0415 -- optional dependency, probed lazily
    except ImportError:
        _check_stamp(path, gz)
        return info

    try:
        with mrcfile.open(str(path), permissive=True, header_only=False) as mrc:
            if mrc.data is None:
                raise MapError("map contains no density data")
            info.shape = tuple(int(x) for x in mrc.data.shape)
            try:
                vs = mrc.voxel_size
                info.voxel_size = (float(vs.x), float(vs.y), float(vs.z))
            except (AttributeError, TypeError):
                info.voxel_size = None
    except MapError:
        raise
    except Exception as exc:
        raise MapError(f"file is not a readable MRC/CCP4 map: {exc}") from exc
    return info


def _check_stamp(path: Path, gz: bool) -> None:
    opener = gzip.open if gz else open
    try:
        with opener(path, "rb") as fh:  # type: ignore[operator]
            header = fh.read(_MRC_HEADER_BYTES)
    except (OSError, EOFError) as exc:
        raise MapError(f"could not read map header: {exc}") from exc

    if len(header) < _MRC_STAMP_OFFSET + 4:
        raise MapError("file is too small to be an MRC/CCP4 map")
    stamp = header[_MRC_STAMP_OFFSET : _MRC_STAMP_OFFSET + 4]
    if stamp != b"MAP ":
        raise MapError(
            "file does not carry the MRC/CCP4 'MAP ' header stamp; "
            "it is probably not a density map"
        )


def store_upload(
    src_stream, destination: Path, *, max_bytes: int
) -> int:
    """Stream an upload to disk, aborting if it exceeds ``max_bytes``."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(destination, "wb") as dst:
        while True:
            chunk = src_stream.read(1 << 20)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                dst.close()
                destination.unlink(missing_ok=True)
                raise MapError(
                    f"upload exceeds the configured limit of {max_bytes} bytes"
                )
            dst.write(chunk)
    return written


def copy_into(path: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return destination
