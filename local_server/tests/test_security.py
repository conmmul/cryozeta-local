"""Path-traversal, archive and display-name safety."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from app.security import (
    SecurityError,
    is_within,
    resolve_within,
    safe_entry_name,
    safe_extract_zip,
    sanitize_display_name,
)

EXTRACT_LIMITS = {"max_total_bytes": 10_000_000, "max_members": 100, "max_ratio": 200}


class TestPathContainment:
    def test_child_is_within(self, tmp_path: Path):
        assert is_within(tmp_path, tmp_path / "a" / "b")

    def test_sibling_is_not(self, tmp_path: Path):
        (tmp_path / "base").mkdir()
        (tmp_path / "other").mkdir()
        assert not is_within(tmp_path / "base", tmp_path / "other")

    def test_symlink_escaping_base_is_rejected(self, tmp_path: Path):
        base = tmp_path / "base"
        base.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("classified")
        link = base / "link"
        link.symlink_to(secret)
        assert not is_within(base, link)

    @pytest.mark.parametrize(
        "bad",
        [
            "../etc/passwd",
            "a/../../b",
            "/etc/passwd",
            "",
            "../../../../../../etc/shadow",
        ],
    )
    def test_resolve_within_rejects_traversal(self, tmp_path: Path, bad):
        with pytest.raises(SecurityError):
            resolve_within(tmp_path, bad)

    def test_resolve_within_allows_normal_relative(self, tmp_path: Path):
        target = resolve_within(tmp_path, "output/model.cif")
        assert target == tmp_path / "output" / "model.cif"


class TestDisplayNames:
    def test_strips_control_characters(self):
        assert sanitize_display_name("ribo\x00some\x1b[31m") == "ribosome[31m"

    def test_collapses_whitespace(self):
        assert sanitize_display_name("  50S   subunit \n") == "50S subunit"

    def test_truncates(self):
        assert len(sanitize_display_name("x" * 500)) == 200

    def test_handles_none(self):
        assert sanitize_display_name(None) == ""


class TestEntryName:
    """The entry name becomes a directory, so it gets the strict treatment."""

    def test_traversal_neutralised(self):
        assert ".." not in safe_entry_name("../../etc/passwd", "fallback")

    def test_spaces_and_symbols_replaced(self):
        assert safe_entry_name("50S subunit (v2)", "fb") == "50S_subunit__v2"

    def test_empty_falls_back(self):
        assert safe_entry_name("", "job_1234") == "job_1234"

    def test_dots_only_falls_back(self):
        assert safe_entry_name("...", "job_1234") == "job_1234"

    def test_length_capped(self):
        assert len(safe_entry_name("a" * 300, "fb")) == 64

    def test_shell_metacharacters_removed(self):
        result = safe_entry_name("job; rm -rf /", "fb")
        assert ";" not in result and " " not in result and "/" not in result


class TestZipExtraction:
    def test_extracts_normal_archive(self, tmp_path: Path):
        archive = tmp_path / "ok.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("uniref100_hits.a3m", ">q\nMKTA\n")
            zf.writestr("nested/other.a3m", ">q\nMKTA\n")

        dest = tmp_path / "out"
        extracted = safe_extract_zip(archive, dest, **EXTRACT_LIMITS)
        assert len(extracted) == 2
        assert (dest / "uniref100_hits.a3m").is_file()

    @pytest.mark.parametrize(
        "member", ["../escape.a3m", "a/../../escape.a3m", "/abs/escape.a3m"]
    )
    def test_rejects_traversal_members(self, tmp_path: Path, member):
        archive = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr(member, "payload")
        with pytest.raises(SecurityError):
            safe_extract_zip(archive, tmp_path / "out", **EXTRACT_LIMITS)

    def test_rejects_symlink_member(self, tmp_path: Path):
        archive = tmp_path / "link.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            info = zipfile.ZipInfo("link")
            # 0xA1FF = symlink mode in the high bits of external_attr.
            info.external_attr = (0o120777 << 16) | 0o777
            zf.writestr(info, "/etc/passwd")
        with pytest.raises(SecurityError, match="not a regular file"):
            safe_extract_zip(archive, tmp_path / "out", **EXTRACT_LIMITS)

    def test_rejects_too_many_members(self, tmp_path: Path):
        archive = tmp_path / "many.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            for i in range(20):
                zf.writestr(f"f{i}.txt", "x")
        with pytest.raises(SecurityError, match="limit is"):
            safe_extract_zip(
                archive,
                tmp_path / "out",
                max_total_bytes=10_000_000,
                max_members=10,
                max_ratio=200,
            )

    def test_rejects_oversized_expansion(self, tmp_path: Path):
        archive = tmp_path / "big.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("big.bin", "A" * 50_000)
        with pytest.raises(SecurityError, match="expands to more than"):
            safe_extract_zip(
                archive,
                tmp_path / "out",
                max_total_bytes=1000,
                max_members=100,
                max_ratio=100_000,
            )

    def test_rejects_decompression_bomb_ratio(self, tmp_path: Path):
        archive = tmp_path / "bomb.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            # Highly compressible: expands far beyond its stored size.
            zf.writestr("bomb.bin", "\0" * 5_000_000)
        with pytest.raises(SecurityError, match="compression ratio"):
            safe_extract_zip(
                archive,
                tmp_path / "out",
                max_total_bytes=100_000_000,
                max_members=100,
                max_ratio=50,
            )
