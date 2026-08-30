"""End-to-end submission validation and job staging (no subprocess)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from app.mapfile import MapError, check_readable, classify_suffix, decompress_if_needed
from app.msa import PROTEIN_NON_PAIRING, PROTEIN_PAIRING
from app.sequences import SeqType
from app.states import InferenceMode, RunMode
from app.submission import (
    SequenceInput,
    SubmissionRequest,
    ValidationError,
    prepare_job,
    validate_submission,
)


def make_request(sample_map: Path, sequences, **overrides) -> SubmissionRequest:
    defaults = dict(
        map_source=sample_map,
        map_filename=sample_map.name,
        resolution="2.99",
        contour_level="0.3",
        sequences=sequences,
        title="Test complex",
        run_mode=RunMode.STANDARD,
        inference_mode=InferenceMode.COMBINED,
    )
    defaults.update(overrides)
    return SubmissionRequest(**defaults)


class TestMapValidation:
    @pytest.mark.parametrize(
        "name", ["a.mrc", "a.map", "a.mrc.gz", "a.map.gz", "A.MAP.GZ"]
    )
    def test_accepted_extensions(self, name):
        assert classify_suffix(name)

    @pytest.mark.parametrize("name", ["a.pdb", "a.cif", "a.txt", "a.zip", "a.map.zip"])
    def test_rejected_extensions(self, name):
        with pytest.raises(MapError):
            classify_suffix(name)

    def test_real_map_is_readable(self, sample_map: Path):
        info = check_readable(sample_map)
        assert info.shape == (8, 8, 8)

    def test_non_map_content_rejected(self, tmp_path: Path):
        fake = tmp_path / "fake.map"
        fake.write_bytes(b"not a density map at all" * 100)
        with pytest.raises(MapError):
            check_readable(fake)

    def test_gzip_roundtrip(self, sample_map: Path, tmp_path: Path):
        gz = tmp_path / "sample.map.gz"
        gz.write_bytes(gzip.compress(sample_map.read_bytes()))
        out = decompress_if_needed(
            gz, tmp_path / "out.map", max_decompressed_bytes=10_000_000
        )
        assert check_readable(out).shape == (8, 8, 8)

    def test_decompression_bomb_rejected(self, tmp_path: Path):
        gz = tmp_path / "bomb.map.gz"
        gz.write_bytes(gzip.compress(b"\0" * 5_000_000))
        with pytest.raises(MapError, match="expands beyond"):
            decompress_if_needed(gz, tmp_path / "out.map", max_decompressed_bytes=1000)


class TestValidation:
    def test_missing_msa_is_rejected(self, settings, sample_map):
        request = make_request(
            sample_map, [SequenceInput(SeqType.PROTEIN, "MKTAYIAK", 1)]
        )
        _, _, _, errors = validate_submission(request, settings)
        assert any("has no MSA" in e for e in errors)
        assert any(PROTEIN_NON_PAIRING in e for e in errors)

    def test_dna_needs_no_msa(self, settings, sample_map):
        request = make_request(sample_map, [SequenceInput(SeqType.DNA, "ACGTACGT", 1)])
        _, _, _, errors = validate_submission(request, settings)
        assert errors == []

    def test_multimer_error_mentions_pairing_file(
        self, settings, sample_map, protein_msa_dir
    ):
        directory = str(protein_msa_dir())
        request = make_request(
            sample_map,
            [
                SequenceInput(SeqType.PROTEIN, "MKTAYIAK", 1, msa_directory=directory),
                SequenceInput(SeqType.PROTEIN, "WWWWCCCC", 1),
            ],
        )
        _, _, _, errors = validate_submission(request, settings)
        assert any(PROTEIN_PAIRING in e for e in errors)

    def test_bad_contour_rejected(self, settings, sample_map):
        request = make_request(
            sample_map,
            [SequenceInput(SeqType.DNA, "ACGT", 1)],
            contour_level="0",
        )
        _, _, _, errors = validate_submission(request, settings)
        assert any("non-zero" in e for e in errors)

    def test_no_sequences_rejected(self, settings, sample_map):
        request = make_request(sample_map, [])
        _, _, _, errors = validate_submission(request, settings)
        assert any("At least one sequence" in e for e in errors)

    def test_total_length_cap(self, settings, sample_map):
        settings.max_total_sequence_length = 10
        request = make_request(
            sample_map, [SequenceInput(SeqType.DNA, "ACGT" * 10, 1)]
        )
        _, _, _, errors = validate_submission(request, settings)
        assert any("exceeds the configured maximum" in e for e in errors)

    def test_all_errors_reported_together(self, settings, sample_map):
        request = make_request(
            sample_map,
            [SequenceInput(SeqType.PROTEIN, "MKTA", 1)],
            resolution="999",
            contour_level="0",
        )
        _, _, _, errors = validate_submission(request, settings)
        assert len(errors) >= 3


class TestPrepareJob:
    def test_creates_job_tree_and_json(
        self, settings, store, sample_map, protein_msa_dir
    ):
        request = make_request(
            sample_map,
            [
                SequenceInput(
                    SeqType.PROTEIN, "MKTAYIAK", 2,
                    msa_directory=str(protein_msa_dir()),
                )
            ],
        )
        prepared = prepare_job(request, settings, store)

        assert prepared.paths.input_json.is_file()
        assert prepared.paths.output_dir.is_dir()
        assert prepared.paths.meta_file.is_file()

        payload = json.loads(prepared.paths.input_json.read_text())
        assert len(payload) == 1
        entry = payload[0]
        assert entry["resolution"] == 2.99
        assert entry["contour_level"] == 0.3
        assert entry["sequences"][0]["proteinChain"]["count"] == 2

    def test_map_is_copied_into_the_job(
        self, settings, store, sample_map, protein_msa_dir
    ):
        request = make_request(
            sample_map,
            [SequenceInput(SeqType.PROTEIN, "MKTA", 1,
                           msa_directory=str(protein_msa_dir()))],
        )
        prepared = prepare_job(request, settings, store)
        staged = prepared.paths.input_dir / sample_map.name
        assert staged.is_file()
        # The generated JSON must point at the job's own copy.
        entry = json.loads(prepared.paths.input_json.read_text())[0]
        assert entry["map_path"] == str(staged.resolve())

    def test_duplicate_sequences_share_one_msa_directory(
        self, settings, store, sample_map, protein_msa_dir
    ):
        directory = str(protein_msa_dir())
        request = make_request(
            sample_map,
            [
                SequenceInput(SeqType.PROTEIN, "MKTAYIAK", 1, msa_directory=directory),
                SequenceInput(SeqType.PROTEIN, "MKTAYIAK", 2, msa_directory=directory),
            ],
        )
        prepared = prepare_job(request, settings, store)

        entry = json.loads(prepared.paths.input_json.read_text())[0]
        # Merged into one chain with the counts summed...
        assert len(entry["sequences"]) == 1
        assert entry["sequences"][0]["proteinChain"]["count"] == 3
        # ...and exactly one MSA directory on disk.
        assert len(list(prepared.paths.msa_dir.iterdir())) == 1

    def test_second_row_supplies_msa_for_duplicate(
        self, settings, store, sample_map, protein_msa_dir
    ):
        # First row has no MSA, the duplicate does: the job should still build.
        request = make_request(
            sample_map,
            [
                SequenceInput(SeqType.PROTEIN, "MKTAYIAK", 1),
                SequenceInput(
                    SeqType.PROTEIN, "MKTAYIAK", 1,
                    msa_directory=str(protein_msa_dir()),
                ),
            ],
        )
        prepared = prepare_job(request, settings, store)
        assert prepared.job.total_seq_len == 16

    def test_title_never_becomes_the_directory(
        self, settings, store, sample_map, protein_msa_dir
    ):
        request = make_request(
            sample_map,
            [SequenceInput(SeqType.PROTEIN, "MKTA", 1,
                           msa_directory=str(protein_msa_dir()))],
            title="../../escape attempt",
        )
        prepared = prepare_job(request, settings, store)
        assert ".." not in prepared.job.entry_name
        # The job directory is the UUID, not the title.
        assert prepared.paths.root.name == prepared.job.id

    def test_large_mode_warning_emitted(
        self, settings, store, sample_map, protein_msa_dir
    ):
        settings.large_complex_threshold = 10
        request = make_request(
            sample_map,
            [SequenceInput(SeqType.PROTEIN, "M" * 50, 1,
                           msa_directory=str(protein_msa_dir()))],
        )
        prepared = prepare_job(request, settings, store)
        assert any("large/cycle mode" in w for w in prepared.warnings)

    def test_failed_submission_leaves_no_directory(
        self, settings, store, sample_map
    ):
        request = make_request(
            sample_map,
            [SequenceInput(SeqType.PROTEIN, "MKTA", 1,
                           msa_directory="/nonexistent/path")],
        )
        before = set(settings.jobs_dir.iterdir())
        with pytest.raises(ValidationError):
            prepare_job(request, settings, store)
        assert set(settings.jobs_dir.iterdir()) == before
