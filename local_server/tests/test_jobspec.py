"""Generation of the CryoZeta input JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.jobspec import (
    ChainSpec,
    JobSpec,
    SpecError,
    build_input_json,
    build_sequence_block,
    recommend_large_mode,
    validate_contour_level,
    validate_resolution,
    write_input_json,
)
from app.sequences import SequenceEntry, SeqType


def make_spec(chains, name="test_entry", tmp_path: Path | None = None) -> JobSpec:
    return JobSpec(
        name=name,
        map_path=(tmp_path or Path("/tmp")) / "emd.map",
        resolution=2.99,
        contour_level=0.3,
        chains=chains,
    )


class TestNumericValidation:
    def test_resolution_in_range(self):
        assert validate_resolution("2.99", minimum=0.5, maximum=30.0) == 2.99

    @pytest.mark.parametrize("bad", ["0.1", "100", "-3"])
    def test_resolution_out_of_range(self, bad):
        with pytest.raises(SpecError):
            validate_resolution(bad, minimum=0.5, maximum=30.0)

    def test_resolution_non_numeric(self):
        with pytest.raises(SpecError):
            validate_resolution("high", minimum=0.5, maximum=30.0)

    def test_contour_nonzero_required(self):
        with pytest.raises(SpecError, match="non-zero"):
            validate_contour_level("0")

    def test_negative_contour_allowed(self):
        # Legitimate for some maps; only zero is meaningless.
        assert validate_contour_level("-0.5") == -0.5

    def test_contour_rejects_nan_and_inf(self):
        for bad in ("nan", "inf", "-inf"):
            with pytest.raises(SpecError):
                validate_contour_level(bad)


class TestSequenceBlocks:
    def test_protein_block_shape(self, tmp_path: Path):
        chain = ChainSpec(
            entry=SequenceEntry(SeqType.PROTEIN, "MKTA", 2), msa_dir=tmp_path
        )
        block = build_sequence_block(chain)
        assert list(block) == ["proteinChain"]
        body = block["proteinChain"]
        assert body["sequence"] == "MKTA"
        assert body["count"] == 2
        assert body["msa"]["precomputed_msa_dir"] == str(tmp_path.resolve())
        assert body["msa"]["pairing_db"] == "uniref100"

    def test_rna_block_shape(self, tmp_path: Path):
        chain = ChainSpec(entry=SequenceEntry(SeqType.RNA, "ACGU", 1), msa_dir=tmp_path)
        block = build_sequence_block(chain)
        assert list(block) == ["rnaSequence"]
        assert "msa" in block["rnaSequence"]

    def test_dna_block_has_no_msa(self):
        chain = ChainSpec(entry=SequenceEntry(SeqType.DNA, "ACGT", 1), msa_dir=None)
        block = build_sequence_block(chain)
        assert list(block) == ["dnaSequence"]
        assert "msa" not in block["dnaSequence"]

    def test_dna_with_msa_rejected(self, tmp_path: Path):
        chain = ChainSpec(entry=SequenceEntry(SeqType.DNA, "ACGT", 1), msa_dir=tmp_path)
        with pytest.raises(SpecError, match="must not be given an MSA"):
            build_sequence_block(chain)

    def test_protein_without_msa_rejected(self):
        chain = ChainSpec(entry=SequenceEntry(SeqType.PROTEIN, "MKTA", 1), msa_dir=None)
        with pytest.raises(SpecError, match="require a precomputed MSA"):
            build_sequence_block(chain)


class TestInputJson:
    def test_is_single_entry_list(self, tmp_path: Path):
        spec = make_spec(
            [ChainSpec(SequenceEntry(SeqType.PROTEIN, "MKTA", 1), tmp_path)],
            tmp_path=tmp_path,
        )
        payload = build_input_json(spec)
        assert isinstance(payload, list)
        assert len(payload) == 1

    def test_entry_has_all_required_keys(self, tmp_path: Path):
        spec = make_spec(
            [ChainSpec(SequenceEntry(SeqType.PROTEIN, "MKTA", 1), tmp_path)],
            tmp_path=tmp_path,
        )
        entry = build_input_json(spec)[0]
        for key in ("name", "modelSeeds", "map_path", "resolution",
                    "contour_level", "sequences"):
            assert key in entry
        assert entry["modelSeeds"] == []

    def test_paths_are_absolute(self, tmp_path: Path):
        spec = make_spec(
            [ChainSpec(SequenceEntry(SeqType.PROTEIN, "MKTA", 1), tmp_path)],
            tmp_path=tmp_path,
        )
        entry = build_input_json(spec)[0]
        assert Path(entry["map_path"]).is_absolute()
        assert Path(
            entry["sequences"][0]["proteinChain"]["msa"]["precomputed_msa_dir"]
        ).is_absolute()

    def test_mixed_complex_ordering_preserved(self, tmp_path: Path):
        chains = [
            ChainSpec(SequenceEntry(SeqType.PROTEIN, "MKTA", 2), tmp_path),
            ChainSpec(SequenceEntry(SeqType.DNA, "ACGT", 1), None),
            ChainSpec(SequenceEntry(SeqType.RNA, "ACGU", 1), tmp_path),
        ]
        entry = build_input_json(make_spec(chains, tmp_path=tmp_path))[0]
        assert [list(b)[0] for b in entry["sequences"]] == [
            "proteinChain",
            "dnaSequence",
            "rnaSequence",
        ]

    def test_no_chains_rejected(self, tmp_path: Path):
        with pytest.raises(SpecError, match="at least one sequence"):
            build_input_json(make_spec([], tmp_path=tmp_path))

    def test_written_file_is_valid_json(self, tmp_path: Path):
        spec = make_spec(
            [ChainSpec(SequenceEntry(SeqType.PROTEIN, "MKTA", 1), tmp_path)],
            tmp_path=tmp_path,
        )
        path = write_input_json(spec, tmp_path / "spec" / "input.json")
        assert json.loads(path.read_text())[0]["name"] == "test_entry"

    def test_needs_pairing_flag(self, tmp_path: Path):
        single = make_spec(
            [ChainSpec(SequenceEntry(SeqType.PROTEIN, "MKTA", 4), tmp_path)],
            tmp_path=tmp_path,
        )
        assert not single.needs_pairing_msa

        double = make_spec(
            [
                ChainSpec(SequenceEntry(SeqType.PROTEIN, "MKTA", 1), tmp_path),
                ChainSpec(SequenceEntry(SeqType.PROTEIN, "WWWW", 1), tmp_path),
            ],
            tmp_path=tmp_path,
        )
        assert double.needs_pairing_msa


class TestLargeModeRecommendation:
    def test_at_threshold_is_standard(self):
        # README says ">2800", so exactly 2800 stays standard.
        assert not recommend_large_mode(2800, 2800)

    def test_above_threshold_recommends_large(self):
        assert recommend_large_mode(2801, 2800)
