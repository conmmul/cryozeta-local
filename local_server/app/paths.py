"""Per-job directory layout.

Every job gets a UUID directory. Inputs, generated spec, logs and outputs are
kept in separate subtrees so that a results download or a cleanup never has to
distinguish them by filename.

    jobs/<uuid>/
        input/      uploaded map (original name preserved, sanitised)
        msa/<hash>/ one directory per distinct protein/RNA sequence
        spec/       input.json handed to CryoZeta
        output/     CryoZeta dump_dir
        logs/       job.log (combined stdout+stderr)
        meta.json   human-readable snapshot of the submission
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JobPaths:
    root: Path

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def msa_dir(self) -> Path:
        return self.root / "msa"

    @property
    def spec_dir(self) -> Path:
        return self.root / "spec"

    @property
    def input_json(self) -> Path:
        return self.spec_dir / "input.json"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def log_file(self) -> Path:
        return self.logs_dir / "job.log"

    @property
    def meta_file(self) -> Path:
        return self.root / "meta.json"

    def msa_dir_for(self, sequence_hash: str) -> Path:
        return self.msa_dir / sequence_hash

    def ensure(self) -> None:
        for path in (
            self.root,
            self.input_dir,
            self.msa_dir,
            self.spec_dir,
            self.output_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def entry_output_dir(self, entry_name: str) -> Path:
        """CryoZeta writes results to ``<dump_dir>/<name>/``."""
        return self.output_dir / entry_name


def job_paths(jobs_root: Path, job_id: str) -> JobPaths:
    return JobPaths(root=Path(jobs_root) / job_id)
