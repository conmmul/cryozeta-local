"""Discovery and classification of CryoZeta output files.

The layout is documented in CryoZeta's README and produced by
``runner/dumper.py``. Standard jobs:

    <dump>/<name>/CryoZeta-Detection/<name>_timing.txt, <name>.pt, *.pdb
    <dump>/<name>/CryoZeta/seed_<seed>/predictions/<name>_sample_N.cif
    <dump>/<name>/CryoZeta/seed_<seed>/predictions/<name>_summary_confidence_sample_N.json
    <dump>/<name>/CryoZeta/saved_data/scores.csv
    <dump>/<name>/CryoZeta-Interpolate/...   (same shape)
    <dump>/<name>/CryoZeta-Final/<name>_sample_{0..N}.cif   <- primary output

Large/cycle jobs additionally produce ``<dump>/combined.cif``.

Rather than hard-coding every filename, files are globbed and classified, so an
upstream rename degrades to "appears under Other files" instead of an empty
results page.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

FINAL_DIR = "CryoZeta-Final"
DETECTION_DIR = "CryoZeta-Detection"


@dataclass
class ResultFile:
    path: Path
    relative: str
    size_bytes: int
    category: str
    rank: int | None = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def size_human(self) -> str:
        size = float(self.size_bytes)
        for unit in ("B", "KiB", "MiB", "GiB"):
            if size < 1024 or unit == "GiB":
                return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GiB"


@dataclass
class ResultSet:
    root: Path
    final_models: list[ResultFile] = field(default_factory=list)
    confidence: list[ResultFile] = field(default_factory=list)
    scores: list[ResultFile] = field(default_factory=list)
    timing: list[ResultFile] = field(default_factory=list)
    intermediate_models: list[ResultFile] = field(default_factory=list)
    other: list[ResultFile] = field(default_factory=list)

    @property
    def has_any(self) -> bool:
        return bool(
            self.final_models
            or self.confidence
            or self.scores
            or self.timing
            or self.intermediate_models
            or self.other
        )

    @property
    def primary_model(self) -> ResultFile | None:
        return self.final_models[0] if self.final_models else None

    def all_files(self) -> list[ResultFile]:
        return [
            *self.final_models,
            *self.confidence,
            *self.scores,
            *self.timing,
            *self.intermediate_models,
            *self.other,
        ]


def _rank_from_name(name: str) -> int | None:
    """Extract the sample index from ``..._sample_3.cif``."""
    stem = Path(name).stem
    if "_sample_" not in stem:
        return None
    tail = stem.rsplit("_sample_", 1)[-1]
    return int(tail) if tail.isdigit() else None


def collect_results(output_dir: Path) -> ResultSet:
    """Walk a job's output directory and classify what is there."""
    output_dir = Path(output_dir)
    results = ResultSet(root=output_dir)
    if not output_dir.is_dir():
        return results

    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            relative = str(path.relative_to(output_dir))
            size = path.stat().st_size
        except (OSError, ValueError):
            continue

        parts = path.parts
        suffix = path.suffix.lower()
        item = ResultFile(
            path=path, relative=relative, size_bytes=size, category="other"
        )

        if suffix == ".cif":
            # Final ranked models, plus large-mode's combined.cif.
            if FINAL_DIR in parts or path.name == "combined.cif":
                item.category = "final"
                item.rank = _rank_from_name(path.name)
                results.final_models.append(item)
            else:
                item.category = "intermediate"
                item.rank = _rank_from_name(path.name)
                results.intermediate_models.append(item)
        elif suffix == ".json" and "confidence" in path.name.lower():
            item.category = "confidence"
            item.rank = _rank_from_name(path.name)
            results.confidence.append(item)
        elif path.name == "scores.csv" or (suffix == ".csv" and "score" in path.name.lower()):
            item.category = "scores"
            results.scores.append(item)
        elif "timing" in path.name.lower():
            item.category = "timing"
            results.timing.append(item)
        else:
            results.other.append(item)

    # Rank-ordered, with combined.cif first for large jobs.
    results.final_models.sort(
        key=lambda f: (f.name != "combined.cif", f.rank if f.rank is not None else 999)
    )
    results.confidence.sort(key=lambda f: (f.rank if f.rank is not None else 999))
    return results


def build_job_zip(job_root: Path, arcname_prefix: str) -> io.BytesIO:
    """Package an entire job directory (inputs, spec, logs, outputs) in memory.

    Symlinks are skipped rather than followed, so a crafted MSA directory
    cannot pull unrelated files into the download.
    """
    buffer = io.BytesIO()
    job_root = Path(job_root)
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(job_root.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                relative = path.relative_to(job_root)
            except ValueError:
                continue
            zf.write(path, arcname=str(Path(arcname_prefix) / relative))
    buffer.seek(0)
    return buffer


def read_log_tail(log_file: Path, max_bytes: int = 200_000) -> str:
    """Return the tail of a log file without loading a huge file into memory."""
    try:
        size = log_file.stat().st_size
    except OSError:
        return ""
    try:
        with open(log_file, "rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                fh.readline()  # discard the partial first line
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
