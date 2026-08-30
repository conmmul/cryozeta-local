"""Sequence normalisation, validation and de-duplication.

The one-letter alphabets accepted here mirror what CryoZeta's data pipeline can
tokenise. Anything outside them is rejected at submission time rather than
producing an opaque crash 20 minutes into a GPU run.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum

PROTEIN_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYX")
DNA_ALPHABET = set("ACGTN")
RNA_ALPHABET = set("ACGUN")


class SeqType(str, Enum):
    PROTEIN = "proteinChain"
    DNA = "dnaSequence"
    RNA = "rnaSequence"

    @property
    def label(self) -> str:
        return {"proteinChain": "Protein", "dnaSequence": "DNA", "rnaSequence": "RNA"}[
            self.value
        ]

    @property
    def requires_msa(self) -> bool:
        """DNA needs no MSA; protein and RNA do."""
        return self is not SeqType.DNA

    @property
    def alphabet(self) -> set[str]:
        return {
            SeqType.PROTEIN: PROTEIN_ALPHABET,
            SeqType.DNA: DNA_ALPHABET,
            SeqType.RNA: RNA_ALPHABET,
        }[self]


class SequenceError(ValueError):
    """Raised for malformed or out-of-range sequence input."""


# FASTA headers, whitespace, digits (from numbered alignments) and '*' stops.
_FASTA_HEADER = re.compile(r"^\s*>.*$", re.MULTILINE)
_STRIP = re.compile(r"[\s\d*\-.]+")


def normalize_sequence(raw: str, seq_type: SeqType) -> str:
    """Clean up pasted sequence text and validate it against the alphabet.

    Tolerates FASTA headers, embedded whitespace/newlines, residue numbering,
    gap characters and a trailing ``*``, because those are what people actually
    paste out of other tools.
    """
    if raw is None:
        raise SequenceError("sequence is empty")

    text = _FASTA_HEADER.sub("", str(raw))
    text = _STRIP.sub("", text).upper()

    if not text:
        raise SequenceError("sequence is empty after cleaning")

    # A common paste error: DNA/RNA mix-up.
    if seq_type is SeqType.RNA and "T" in text and "U" not in text:
        raise SequenceError(
            "RNA sequence contains T but no U -- did you mean to select DNA?"
        )
    if seq_type is SeqType.DNA and "U" in text:
        raise SequenceError(
            "DNA sequence contains U -- did you mean to select RNA?"
        )

    invalid = sorted(set(text) - seq_type.alphabet)
    if invalid:
        raise SequenceError(
            f"{seq_type.label} sequence contains invalid characters: "
            f"{', '.join(invalid)}"
        )
    return text


def sequence_hash(sequence: str, seq_type: SeqType) -> str:
    """Stable identifier used to share one MSA directory between duplicates."""
    digest = hashlib.sha256(f"{seq_type.value}:{sequence}".encode()).hexdigest()
    return digest[:16]


@dataclass
class SequenceEntry:
    """One chain specification from the new-job form."""

    seq_type: SeqType
    sequence: str
    count: int

    @property
    def hash(self) -> str:
        return sequence_hash(self.sequence, self.seq_type)

    @property
    def total_residues(self) -> int:
        return len(self.sequence) * self.count


def validate_count(raw: int | str, max_copies: int) -> int:
    try:
        count = int(raw)
    except (TypeError, ValueError) as exc:
        raise SequenceError(f"copy count must be an integer, got {raw!r}") from exc
    if count < 1:
        raise SequenceError("copy count must be at least 1")
    if count > max_copies:
        raise SequenceError(f"copy count must not exceed {max_copies}")
    return count


def total_sequence_length(entries: list[SequenceEntry]) -> int:
    """Total residues/nucleotides across all copies of all chains.

    This is the figure compared against the large-complex threshold.
    """
    return sum(entry.total_residues for entry in entries)


def distinct_protein_sequences(entries: list[SequenceEntry]) -> set[str]:
    return {e.sequence for e in entries if e.seq_type is SeqType.PROTEIN}


def is_homomer_or_monomer(entries: list[SequenceEntry]) -> bool:
    """Replicates CryoZeta's own pairing decision for protein MSAs.

    See ``src/cryozeta/data/msa_featurizer.py``: the flag is computed as
    ``len(set(entity_id_to_sequence.values())) == 1`` over *protein* entities,
    i.e. copy counts are irrelevant -- only the number of distinct protein
    sequences matters. When this is False, CryoZeta asserts the presence of
    ``uniref100_hits.a3m`` for every protein chain.
    """
    return len(distinct_protein_sequences(entries)) <= 1


def dedupe_entries(entries: list[SequenceEntry]) -> list[SequenceEntry]:
    """Merge byte-identical chains of the same type by summing their counts.

    Two form rows holding the same protein sequence describe the same entity,
    so they collapse into a single entry with a combined copy count and share
    one MSA directory.
    """
    merged: dict[tuple[SeqType, str], SequenceEntry] = {}
    order: list[tuple[SeqType, str]] = []
    for entry in entries:
        key = (entry.seq_type, entry.sequence)
        if key in merged:
            merged[key].count += entry.count
        else:
            merged[key] = SequenceEntry(entry.seq_type, entry.sequence, entry.count)
            order.append(key)
    return [merged[k] for k in order]
