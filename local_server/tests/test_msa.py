"""MSA requirement rules, validation and providers."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.msa import (
    PROTEIN_NON_PAIRING,
    PROTEIN_PAIRING,
    RNA_NON_PAIRING,
    LocalDirectoryProvider,
    MSAError,
    RemoteColabFoldProvider,
    UploadedArchiveProvider,
    find_msa_root,
    required_filenames,
    validate_msa_dir,
)
from app.sequences import SeqType

from .conftest import write_a3m

LIMITS = {"max_total_bytes": 10_000_000, "max_members": 100, "max_ratio": 200}


class TestRequiredFilenames:
    def test_dna_needs_nothing(self):
        assert required_filenames(SeqType.DNA, needs_pairing=True) == []

    def test_rna_needs_only_rnacentral(self):
        assert required_filenames(SeqType.RNA, needs_pairing=True) == [RNA_NON_PAIRING]

    def test_protein_monomer(self):
        assert required_filenames(SeqType.PROTEIN, needs_pairing=False) == [
            PROTEIN_NON_PAIRING
        ]

    def test_protein_multimer_adds_pairing(self):
        assert required_filenames(SeqType.PROTEIN, needs_pairing=True) == [
            PROTEIN_NON_PAIRING,
            PROTEIN_PAIRING,
        ]


class TestValidation:
    def test_valid_protein_dir(self, protein_msa_dir):
        result = validate_msa_dir(protein_msa_dir(), SeqType.PROTEIN, True)
        assert result.ok

    def test_missing_pairing_file_detected(self, protein_msa_dir):
        directory = protein_msa_dir(pairing=False)
        result = validate_msa_dir(directory, SeqType.PROTEIN, needs_pairing=True)
        assert not result.ok
        assert PROTEIN_PAIRING in result.missing
        assert PROTEIN_PAIRING in result.message()

    def test_pairing_file_not_needed_for_monomer(self, protein_msa_dir):
        directory = protein_msa_dir(pairing=False)
        assert validate_msa_dir(directory, SeqType.PROTEIN, needs_pairing=False).ok

    def test_missing_directory(self, tmp_path: Path):
        result = validate_msa_dir(tmp_path / "nope", SeqType.PROTEIN, False)
        assert not result.ok

    def test_empty_file_is_malformed(self, tmp_path: Path):
        directory = tmp_path / "msa"
        directory.mkdir()
        (directory / PROTEIN_NON_PAIRING).write_text("")
        result = validate_msa_dir(directory, SeqType.PROTEIN, False)
        assert PROTEIN_NON_PAIRING in result.malformed

    def test_non_a3m_content_is_malformed(self, tmp_path: Path):
        directory = tmp_path / "msa"
        directory.mkdir()
        (directory / PROTEIN_NON_PAIRING).write_text("this is not an alignment")
        result = validate_msa_dir(directory, SeqType.PROTEIN, False)
        assert PROTEIN_NON_PAIRING in result.malformed

    def test_rna_dir(self, rna_msa_dir):
        assert validate_msa_dir(rna_msa_dir(), SeqType.RNA, needs_pairing=False).ok


class TestFindMsaRoot:
    def test_finds_nested_directory(self, tmp_path: Path):
        nested = tmp_path / "archive" / "chainA"
        write_a3m(nested / PROTEIN_NON_PAIRING)
        assert find_msa_root(tmp_path, SeqType.PROTEIN, False) == nested

    def test_ignores_macosx_resource_fork(self, tmp_path: Path):
        write_a3m(tmp_path / "__MACOSX" / PROTEIN_NON_PAIRING)
        real = tmp_path / "real"
        write_a3m(real / PROTEIN_NON_PAIRING)
        assert find_msa_root(tmp_path, SeqType.PROTEIN, False) == real

    def test_returns_input_when_files_at_top(self, protein_msa_dir):
        directory = protein_msa_dir()
        assert find_msa_root(directory, SeqType.PROTEIN, False) == directory


class TestUploadedArchiveProvider:
    def test_extracts_and_validates(self, make_zip, tmp_path: Path):
        archive = make_zip(
            "msa.zip",
            {
                PROTEIN_NON_PAIRING: ">q\nMKTA\n",
                PROTEIN_PAIRING: ">q\nMKTA\n",
            },
        )
        provider = UploadedArchiveProvider(archive, **LIMITS)
        result = provider.provide(
            sequence="MKTA",
            seq_type=SeqType.PROTEIN,
            needs_pairing=True,
            destination=tmp_path / "out",
        )
        assert (result / PROTEIN_NON_PAIRING).is_file()

    def test_nested_archive_layout_handled(self, make_zip, tmp_path: Path):
        archive = make_zip(
            "nested.zip", {f"msas/chainA/{PROTEIN_NON_PAIRING}": ">q\nMKTA\n"}
        )
        provider = UploadedArchiveProvider(archive, **LIMITS)
        result = provider.provide(
            sequence="MKTA",
            seq_type=SeqType.PROTEIN,
            needs_pairing=False,
            destination=tmp_path / "out",
        )
        assert (result / PROTEIN_NON_PAIRING).is_file()

    def test_missing_required_file_raises(self, make_zip, tmp_path: Path):
        archive = make_zip("bad.zip", {"readme.txt": "nothing useful"})
        provider = UploadedArchiveProvider(archive, **LIMITS)
        with pytest.raises(MSAError, match="unusable"):
            provider.provide(
                sequence="MKTA",
                seq_type=SeqType.PROTEIN,
                needs_pairing=False,
                destination=tmp_path / "out",
            )

    def test_corrupt_archive_raises(self, tmp_path: Path):
        archive = tmp_path / "corrupt.zip"
        archive.write_bytes(b"definitely not a zip file")
        provider = UploadedArchiveProvider(archive, **LIMITS)
        with pytest.raises(MSAError, match="could not read"):
            provider.provide(
                sequence="MKTA",
                seq_type=SeqType.PROTEIN,
                needs_pairing=False,
                destination=tmp_path / "out",
            )


class TestLocalDirectoryProvider:
    def test_copies_required_files(self, protein_msa_dir, tmp_path: Path):
        source = protein_msa_dir()
        provider = LocalDirectoryProvider(source)
        destination = tmp_path / "job_msa"
        result = provider.provide(
            sequence="MKTA",
            seq_type=SeqType.PROTEIN,
            needs_pairing=True,
            destination=destination,
        )
        assert result == destination
        assert (destination / PROTEIN_PAIRING).is_file()
        # The original must be left untouched.
        assert (source / PROTEIN_PAIRING).is_file()

    def test_missing_directory_raises(self, tmp_path: Path):
        provider = LocalDirectoryProvider(tmp_path / "absent")
        with pytest.raises(MSAError, match="does not exist"):
            provider.provide(
                sequence="MKTA",
                seq_type=SeqType.PROTEIN,
                needs_pairing=False,
                destination=tmp_path / "out",
            )

    def test_incomplete_directory_raises_before_copying(
        self, protein_msa_dir, tmp_path: Path
    ):
        provider = LocalDirectoryProvider(protein_msa_dir(pairing=False))
        destination = tmp_path / "job_msa"
        with pytest.raises(MSAError):
            provider.provide(
                sequence="MKTA",
                seq_type=SeqType.PROTEIN,
                needs_pairing=True,
                destination=destination,
            )
        assert not destination.exists()


class TestRemoteProvider:
    def test_is_not_silently_available(self, tmp_path: Path):
        with pytest.raises(MSAError, match="does not contact external services"):
            RemoteColabFoldProvider().provide(
                sequence="MKTA",
                seq_type=SeqType.PROTEIN,
                needs_pairing=False,
                destination=tmp_path,
            )
