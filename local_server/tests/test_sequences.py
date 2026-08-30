"""Sequence normalisation, validation and de-duplication."""

from __future__ import annotations

import pytest

from app.sequences import (
    SequenceEntry,
    SequenceError,
    SeqType,
    dedupe_entries,
    is_homomer_or_monomer,
    normalize_sequence,
    sequence_hash,
    total_sequence_length,
    validate_count,
)


class TestNormalize:
    def test_strips_whitespace_and_uppercases(self):
        assert normalize_sequence("  mkt ay\nia  ", SeqType.PROTEIN) == "MKTAYIA"

    def test_strips_fasta_header(self):
        raw = ">sp|P12345|Test protein\nMKTAYIAK\nQRQISFVK\n"
        assert normalize_sequence(raw, SeqType.PROTEIN) == "MKTAYIAKQRQISFVK"

    def test_strips_numbering_gaps_and_stop(self):
        assert normalize_sequence("1 MKTA-YIA.K*", SeqType.PROTEIN) == "MKTAYIAK"

    def test_rejects_invalid_protein_letters(self):
        with pytest.raises(SequenceError, match="invalid characters"):
            normalize_sequence("MKTAZZJO", SeqType.PROTEIN)

    def test_accepts_x_in_protein(self):
        assert normalize_sequence("MKTXAY", SeqType.PROTEIN) == "MKTXAY"

    def test_empty_rejected(self):
        with pytest.raises(SequenceError):
            normalize_sequence("   \n  ", SeqType.PROTEIN)

    @pytest.mark.parametrize(
        "raw,seq_type,expected",
        [
            ("acgt", SeqType.DNA, "ACGT"),
            ("acgu", SeqType.RNA, "ACGU"),
            ("ACGTN", SeqType.DNA, "ACGTN"),
        ],
    )
    def test_nucleic_alphabets(self, raw, seq_type, expected):
        assert normalize_sequence(raw, seq_type) == expected

    def test_rna_with_t_and_no_u_is_rejected(self):
        # A very common paste error worth catching before a 30-minute GPU run.
        with pytest.raises(SequenceError, match="did you mean to select DNA"):
            normalize_sequence("ACGTACGT", SeqType.RNA)

    def test_dna_with_u_is_rejected(self):
        with pytest.raises(SequenceError, match="did you mean to select RNA"):
            normalize_sequence("ACGUACGU", SeqType.DNA)

    def test_rna_with_both_t_and_u_reports_invalid_char(self):
        with pytest.raises(SequenceError, match="invalid characters"):
            normalize_sequence("ACGUT", SeqType.RNA)


class TestCounts:
    def test_valid(self):
        assert validate_count("3", 128) == 3

    @pytest.mark.parametrize("bad", ["0", "-1", 0])
    def test_below_one_rejected(self, bad):
        with pytest.raises(SequenceError):
            validate_count(bad, 128)

    def test_above_max_rejected(self):
        with pytest.raises(SequenceError, match="must not exceed"):
            validate_count(200, 128)

    def test_non_numeric_rejected(self):
        with pytest.raises(SequenceError):
            validate_count("two", 128)


class TestDedupe:
    def test_identical_sequences_merge_counts(self):
        entries = [
            SequenceEntry(SeqType.PROTEIN, "MKTA", 1),
            SequenceEntry(SeqType.PROTEIN, "MKTA", 2),
        ]
        merged = dedupe_entries(entries)
        assert len(merged) == 1
        assert merged[0].count == 3

    def test_same_sequence_different_type_not_merged(self):
        entries = [
            SequenceEntry(SeqType.DNA, "ACGT", 1),
            SequenceEntry(SeqType.RNA, "ACGU", 1),
        ]
        assert len(dedupe_entries(entries)) == 2

    def test_order_preserved(self):
        entries = [
            SequenceEntry(SeqType.PROTEIN, "AAA", 1),
            SequenceEntry(SeqType.PROTEIN, "CCC", 1),
            SequenceEntry(SeqType.PROTEIN, "AAA", 1),
        ]
        merged = dedupe_entries(entries)
        assert [e.sequence for e in merged] == ["AAA", "CCC"]

    def test_identical_sequences_share_one_msa_hash(self):
        a = SequenceEntry(SeqType.PROTEIN, "MKTA", 1)
        b = SequenceEntry(SeqType.PROTEIN, "MKTA", 5)
        assert a.hash == b.hash

    def test_hash_is_type_scoped(self):
        assert sequence_hash("ACGT", SeqType.DNA) != sequence_hash("ACGT", SeqType.PROTEIN)


class TestTotals:
    def test_counts_copies(self):
        entries = [
            SequenceEntry(SeqType.PROTEIN, "A" * 100, 2),
            SequenceEntry(SeqType.DNA, "ACGT", 3),
        ]
        assert total_sequence_length(entries) == 212


class TestHomomerDetection:
    """Mirrors CryoZeta: only the count of *distinct protein* sequences matters."""

    def test_single_protein_is_monomer(self):
        assert is_homomer_or_monomer([SequenceEntry(SeqType.PROTEIN, "MKTA", 1)])

    def test_many_copies_of_one_sequence_is_homomer(self):
        assert is_homomer_or_monomer([SequenceEntry(SeqType.PROTEIN, "MKTA", 8)])

    def test_two_distinct_proteins_is_not(self):
        entries = [
            SequenceEntry(SeqType.PROTEIN, "MKTA", 1),
            SequenceEntry(SeqType.PROTEIN, "WWWW", 1),
        ]
        assert not is_homomer_or_monomer(entries)

    def test_protein_plus_dna_still_monomer(self):
        # Nucleic chains do not create a protein pairing requirement.
        entries = [
            SequenceEntry(SeqType.PROTEIN, "MKTA", 1),
            SequenceEntry(SeqType.DNA, "ACGT", 2),
        ]
        assert is_homomer_or_monomer(entries)

    def test_no_protein_at_all(self):
        assert is_homomer_or_monomer([SequenceEntry(SeqType.RNA, "ACGU", 1)])
