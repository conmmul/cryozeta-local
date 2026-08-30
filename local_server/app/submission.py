"""Turn a validated new-job form into a job on disk.

Kept free of FastAPI types so the whole path is unit-testable: callers hand in
plain values and already-staged temporary files.

Order of operations matters. Everything cheap and rejectable is validated
*before* a job directory is created, so a bad submission leaves nothing behind.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .db import Job, JobStore
from .jobspec import (
    ChainSpec,
    JobSpec,
    SpecError,
    recommend_large_mode,
    validate_contour_level,
    validate_resolution,
    write_input_json,
)
from .mapfile import MapError, classify_suffix, check_readable, decompress_if_needed
from .msa import (
    LocalDirectoryProvider,
    MSAError,
    MSAProvider,
    UploadedArchiveProvider,
    required_filenames,
)
from .paths import JobPaths, job_paths
from .security import safe_entry_name, sanitize_display_name
from .sequences import (
    SequenceEntry,
    SequenceError,
    SeqType,
    is_homomer_or_monomer,
    normalize_sequence,
    total_sequence_length,
    validate_count,
)
from .states import InferenceMode, RunMode


class ValidationError(ValueError):
    """Collects every problem with a submission so the form can show them all."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass
class SequenceInput:
    """One raw row from the sequence repeater."""

    seq_type: SeqType
    sequence: str
    count: int = 1
    msa_archive: Path | None = None
    msa_directory: str | None = None

    def has_msa_source(self) -> bool:
        return self.msa_archive is not None or bool(
            self.msa_directory and self.msa_directory.strip()
        )


@dataclass
class SubmissionRequest:
    map_source: Path
    map_filename: str
    resolution: float | str
    contour_level: float | str
    sequences: list[SequenceInput]
    title: str = ""
    note: str = ""
    gpu_index: int | None = None
    run_mode: RunMode = RunMode.STANDARD
    inference_mode: InferenceMode = InferenceMode.COMBINED
    overwrite: bool = False
    submitted_by: str = ""
    model_seeds: list[int] = field(default_factory=list)


@dataclass
class PreparedJob:
    job: Job
    paths: JobPaths
    spec: JobSpec
    warnings: list[str] = field(default_factory=list)


def _normalise_sequences(
    inputs: list[SequenceInput], settings: Settings, errors: list[str]
) -> list[tuple[SequenceEntry, SequenceInput]]:
    """Clean, validate and de-duplicate the submitted chains."""
    pairs: list[tuple[SequenceEntry, SequenceInput]] = []
    for index, item in enumerate(inputs, start=1):
        try:
            sequence = normalize_sequence(item.sequence, item.seq_type)
            count = validate_count(item.count, settings.max_copies_per_sequence)
        except SequenceError as exc:
            errors.append(f"Sequence {index}: {exc}")
            continue
        pairs.append((SequenceEntry(item.seq_type, sequence, count), item))

    if not pairs:
        if not errors:
            errors.append("At least one sequence is required.")
        return []

    # Merge identical chains, carrying forward the first MSA source given for
    # each distinct sequence so duplicates share one alignment directory.
    merged: dict[tuple[SeqType, str], tuple[SequenceEntry, SequenceInput]] = {}
    order: list[tuple[SeqType, str]] = []
    for entry, raw in pairs:
        key = (entry.seq_type, entry.sequence)
        if key in merged:
            existing_entry, existing_raw = merged[key]
            existing_entry.count += entry.count
            if not existing_raw.has_msa_source() and raw.has_msa_source():
                merged[key] = (existing_entry, raw)
        else:
            merged[key] = (
                SequenceEntry(entry.seq_type, entry.sequence, entry.count),
                raw,
            )
            order.append(key)
    return [merged[k] for k in order]


def validate_submission(
    request: SubmissionRequest, settings: Settings
) -> tuple[list[tuple[SequenceEntry, SequenceInput]], float, float, list[str]]:
    """Validate everything that does not require touching the filesystem."""
    errors: list[str] = []

    try:
        classify_suffix(request.map_filename)
    except MapError as exc:
        errors.append(str(exc))

    try:
        resolution = validate_resolution(
            request.resolution,
            minimum=settings.min_resolution,
            maximum=settings.max_resolution,
        )
    except SpecError as exc:
        errors.append(str(exc))
        resolution = 0.0

    try:
        contour = validate_contour_level(request.contour_level)
    except SpecError as exc:
        errors.append(str(exc))
        contour = 0.0

    pairs = _normalise_sequences(request.sequences, settings, errors)

    if pairs:
        entries = [e for e, _ in pairs]
        total = total_sequence_length(entries)
        if total > settings.max_total_sequence_length:
            errors.append(
                f"Total sequence length {total} exceeds the configured maximum "
                f"of {settings.max_total_sequence_length}."
            )

        # Fail loudly on missing MSAs rather than submitting a doomed job.
        needs_pairing = not is_homomer_or_monomer(entries)
        for position, (entry, raw) in enumerate(pairs, start=1):
            if entry.seq_type is SeqType.DNA:
                continue
            if not raw.has_msa_source():
                needed = ", ".join(required_filenames(entry.seq_type, needs_pairing))
                errors.append(
                    f"Sequence {position} ({entry.seq_type.label}) has no MSA. "
                    f"CryoZeta requires a precomputed MSA directory containing: {needed}."
                )

    return pairs, resolution, contour, errors


def prepare_job(
    request: SubmissionRequest,
    settings: Settings,
    store: JobStore,
) -> PreparedJob:
    """Validate, stage inputs, generate the JSON and record the job."""
    pairs, resolution, contour, errors = validate_submission(request, settings)
    if errors:
        raise ValidationError(errors)

    entries = [e for e, _ in pairs]
    total_length = total_sequence_length(entries)
    needs_pairing = not is_homomer_or_monomer(entries)
    warnings: list[str] = []

    job_id = str(uuid.uuid4())
    paths = job_paths(settings.jobs_dir, job_id)
    paths.ensure()

    try:
        # -- map ---------------------------------------------------------
        safe_map_name = Path(request.map_filename).name.replace("\\", "_")
        staged_map = paths.input_dir / safe_map_name
        shutil.copy2(request.map_source, staged_map)

        if safe_map_name.lower().endswith(".gz"):
            # Validate the decompressed content, but keep the original: the
            # detection step accepts .map.gz directly.
            probe = paths.input_dir / (safe_map_name[:-3] + ".probe")
            try:
                decompressed = decompress_if_needed(
                    staged_map,
                    probe,
                    max_decompressed_bytes=settings.max_decompressed_bytes,
                )
                check_readable(decompressed)
            finally:
                probe.unlink(missing_ok=True)
        else:
            check_readable(staged_map)

        # -- MSAs --------------------------------------------------------
        chains: list[ChainSpec] = []
        for position, (entry, raw) in enumerate(pairs, start=1):
            if entry.seq_type is SeqType.DNA:
                chains.append(ChainSpec(entry=entry, msa_dir=None))
                continue

            provider = _provider_for(raw, settings)
            destination = paths.msa_dir_for(entry.hash)
            try:
                resolved = provider.provide(
                    sequence=entry.sequence,
                    seq_type=entry.seq_type,
                    needs_pairing=needs_pairing,
                    destination=destination,
                )
            except MSAError as exc:
                raise ValidationError([f"Sequence {position}: {exc}"]) from exc
            chains.append(ChainSpec(entry=entry, msa_dir=resolved))

        # -- spec --------------------------------------------------------
        title = sanitize_display_name(request.title)
        note = sanitize_display_name(request.note, max_length=2000)
        entry_name = safe_entry_name(title, fallback=f"job_{job_id[:8]}")

        spec = JobSpec(
            name=entry_name,
            map_path=staged_map,
            resolution=resolution,
            contour_level=contour,
            chains=chains,
            model_seeds=list(request.model_seeds),
        )
        write_input_json(spec, paths.input_json)

        if recommend_large_mode(total_length, settings.large_complex_threshold) and (
            request.run_mode is RunMode.STANDARD
        ):
            warnings.append(
                f"This complex is {total_length} residues/nucleotides, above the "
                f"{settings.large_complex_threshold} threshold at which CryoZeta "
                "recommends large/cycle mode. Standard mode may run out of memory."
            )

        sequences_meta = [
            {
                "type": e.seq_type.value,
                "label": e.seq_type.label,
                "length": len(e.sequence),
                "count": e.count,
                "hash": e.hash,
            }
            for e in entries
        ]

        paths.meta_file.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "title": title,
                    "note": note,
                    "entry_name": entry_name,
                    "resolution": resolution,
                    "contour_level": contour,
                    "run_mode": request.run_mode.value,
                    "inference_mode": request.inference_mode.value,
                    "total_sequence_length": total_length,
                    "needs_pairing_msa": needs_pairing,
                    "sequences": sequences_meta,
                    "map_filename": safe_map_name,
                    "submitted_by": request.submitted_by,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        job = store.create(
            entry_name=entry_name,
            title=title,
            note=note,
            run_mode=request.run_mode,
            inference_mode=request.inference_mode,
            gpu_index=request.gpu_index,
            resolution=resolution,
            contour_level=contour,
            total_seq_len=total_length,
            map_filename=safe_map_name,
            overwrite=request.overwrite,
            sequences=sequences_meta,
            submitted_by=request.submitted_by,
            job_id=job_id,
        )
    except Exception:
        # A failed submission must not leave a half-built job directory.
        shutil.rmtree(paths.root, ignore_errors=True)
        raise

    return PreparedJob(job=job, paths=paths, spec=spec, warnings=warnings)


def _provider_for(raw: SequenceInput, settings: Settings) -> MSAProvider:
    if raw.msa_archive is not None:
        return UploadedArchiveProvider(
            raw.msa_archive,
            max_total_bytes=settings.max_decompressed_bytes,
            max_members=settings.max_archive_members,
            max_ratio=settings.max_compression_ratio,
        )
    if raw.msa_directory:
        return LocalDirectoryProvider(Path(raw.msa_directory.strip()))
    raise MSAError("no MSA source was provided")
