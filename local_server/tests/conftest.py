"""Shared fixtures.

Every test gets an isolated data root, so nothing touches a real installation.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

# Make the `app` package importable when pytest runs from local_server/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.db import JobStore  # noqa: E402
from app.msa import (  # noqa: E402
    PROTEIN_NON_PAIRING,
    PROTEIN_PAIRING,
    RNA_NON_PAIRING,
)

FAKE_SCRIPT = Path(__file__).parent / "fake" / "fake_inference_demo.sh"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    config = Settings()
    config.data_root = tmp_path / "data"
    config.host = "127.0.0.1"
    config.cancel_grace_seconds = 2
    config.ensure_dirs()
    return config


@pytest.fixture
def store(settings: Settings) -> JobStore:
    instance = JobStore(settings.db_path)
    yield instance
    instance.close()


def write_a3m(path: Path, sequence: str = "MKTAYIAKQRQ") -> Path:
    """Write a minimal but structurally valid A3M alignment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f">query\n{sequence}\n>hit_1\n{sequence}\n", encoding="utf-8"
    )
    return path


@pytest.fixture
def protein_msa_dir(tmp_path: Path):
    def _make(name: str = "msa_protein", *, pairing: bool = True) -> Path:
        directory = tmp_path / name
        write_a3m(directory / PROTEIN_NON_PAIRING)
        if pairing:
            write_a3m(directory / PROTEIN_PAIRING)
        return directory

    return _make


@pytest.fixture
def rna_msa_dir(tmp_path: Path):
    def _make(name: str = "msa_rna") -> Path:
        directory = tmp_path / name
        write_a3m(directory / RNA_NON_PAIRING, sequence="ACGUACGUA")
        return directory

    return _make


@pytest.fixture
def make_zip(tmp_path: Path):
    def _make(name: str, members: dict[str, str]) -> Path:
        archive = tmp_path / name
        with zipfile.ZipFile(archive, "w") as zf:
            for member, content in members.items():
                zf.writestr(member, content)
        return archive

    return _make


@pytest.fixture
def sample_map(tmp_path: Path) -> Path:
    """A tiny but genuinely valid MRC file."""
    import mrcfile
    import numpy as np

    path = tmp_path / "sample.map"
    data = np.zeros((8, 8, 8), dtype=np.float32)
    data[3:5, 3:5, 3:5] = 1.0
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = 1.0
    return path
