"""Generation of the CryoZeta inference JSON.

The structure produced here is derived from the current repository, not from
prose:

* top level is a **list** of entries (``inference_detection.py`` iterates it,
  and ``large_inference_demo.sh`` indexes into it)
* ``name`` becomes a directory under the dump dir, so it is sanitised
* each element of ``sequences`` is a single-key wrapper -- ``proteinChain``,
  ``dnaSequence`` or ``rnaSequence`` -- around ``{sequence, count, msa}``
  (``json_parser.py``, ``json_to_feature.py``)
* ``msa`` is omitted entirely for DNA, which CryoZeta never reads
* paths are written absolute; CryoZeta resolves relative paths against the
  JSON file's own directory (``infer_data_pipeline._resolve_json_path``)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .msa import DEFAULT_PAIRING_DB
from .sequences import SequenceEntry, SeqType, is_homomer_or_monomer


class SpecError(ValueError):
    """Raised when the requested job cannot be expressed as valid input."""


@dataclass
class ChainSpec:
    """One chain plus the MSA directory resolved for it (None for DNA)."""

    entry: SequenceEntry
    msa_dir: Path | None = None

    @property
    def seq_type(self) -> SeqType:
        return self.entry.seq_type


@dataclass
class JobSpec:
    name: str
    map_path: Path
    resolution: float
    contour_level: float
    chains: list[ChainSpec]
    model_seeds: list[int] = field(default_factory=list)

    @property
    def entries(self) -> list[SequenceEntry]:
        return [c.entry for c in self.chains]

    @property
    def needs_pairing_msa(self) -> bool:
        """True when CryoZeta will demand ``uniref100_hits.a3m``."""
        return not is_homomer_or_monomer(self.entries)


def validate_resolution(value: Any, *, minimum: float, maximum: float) -> float:
    try:
        resolution = float(value)
    except (TypeError, ValueError) as exc:
        raise SpecError(f"resolution must be a number, got {value!r}") from exc
    if not (minimum <= resolution <= maximum):
        raise SpecError(
            f"resolution must be between {minimum} and {maximum} A, got {resolution}"
        )
    return resolution


def validate_contour_level(value: Any) -> float:
    """Contour level must be present, finite and non-zero.

    ``inference_detection.py`` raises if ``contour_level`` is None, and
    ``strucblur.py`` thresholds the grid with it, so zero silently produces a
    degenerate map.
    """
    try:
        contour = float(value)
    except (TypeError, ValueError) as exc:
        raise SpecError(f"contour level must be a number, got {value!r}") from exc
    if contour != contour or contour in (float("inf"), float("-inf")):
        raise SpecError("contour level must be a finite number")
    if contour == 0.0:
        raise SpecError("contour level must be non-zero")
    return contour


def build_sequence_block(chain: ChainSpec) -> dict[str, Any]:
    """Build one ``sequences`` element."""
    body: dict[str, Any] = {
        "sequence": chain.entry.sequence,
        "count": chain.entry.count,
    }

    if chain.seq_type is SeqType.DNA:
        # DNA never carries an msa block: CryoZeta's featuriser only looks up
        # ["msa"] for protein and RNA entities.
        if chain.msa_dir is not None:
            raise SpecError("DNA chains must not be given an MSA directory")
        return {chain.seq_type.value: body}

    if chain.msa_dir is None:
        raise SpecError(
            f"{chain.seq_type.label} chains require a precomputed MSA directory"
        )

    body["msa"] = {
        "precomputed_msa_dir": str(Path(chain.msa_dir).resolve()),
        "pairing_db": DEFAULT_PAIRING_DB,
    }
    return {chain.seq_type.value: body}


def build_entry(spec: JobSpec) -> dict[str, Any]:
    if not spec.chains:
        raise SpecError("at least one sequence is required")
    if not spec.name:
        raise SpecError("entry name is required")

    return {
        "name": spec.name,
        "modelSeeds": list(spec.model_seeds),
        "map_path": str(Path(spec.map_path).resolve()),
        "resolution": spec.resolution,
        "contour_level": spec.contour_level,
        "sequences": [build_sequence_block(c) for c in spec.chains],
    }


def build_input_json(spec: JobSpec) -> list[dict[str, Any]]:
    """CryoZeta always consumes a list of entries; we emit exactly one.

    A single-entry list is also what ``large_inference_demo.sh`` expects: its
    ``--example`` selector defaults to index ``0``, so no extra flag is needed.
    """
    return [build_entry(spec)]


def write_input_json(spec: JobSpec, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_input_json(spec), indent=2), encoding="utf-8")
    return path


def recommend_large_mode(total_length: int, threshold: int) -> bool:
    """Whether to recommend large/cycle mode.

    The CryoZeta README recommends large-complex mode for complexes of more
    than 2800 residues or nucleotides; the threshold is configurable rather
    than hard-coded so it can track upstream.
    """
    return total_length > threshold
