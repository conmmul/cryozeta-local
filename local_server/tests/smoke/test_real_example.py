"""Real smoke tests against a genuine CryoZeta installation.

These are skipped automatically unless the host actually has CryoZeta set up
(repo + weights + TEASER++ + pixi + GPU), so the default test run stays fast
and GPU-free. Run them explicitly with:

    ./run_tests.sh --smoke

Two levels:

``test_generated_json_matches_bundled_example`` is cheap. It needs only the
downloaded assets, and checks our generated JSON against the *real*
``assets/examples/example.json`` field by field. This is the check that
"CryoZeta accepts our JSON" without burning half an hour of GPU time.

``test_bundled_example_runs_end_to_end`` is the full thing: it submits the
bundled example through the web application and waits for real inference.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.db import JobStore
from app.discovery import find_cryozeta_repo, inspect_install
from app.jobspec import ChainSpec, JobSpec, build_input_json
from app.preflight import run_preflight
from app.sequences import SequenceEntry, SeqType
from app.states import JobStatus

pytestmark = pytest.mark.smoke


def _assets():
    repo = find_cryozeta_repo()
    if repo is None:
        pytest.skip("no CryoZeta repository on this machine")
    install = inspect_install(repo)
    if not install.has_examples:
        pytest.skip("assets/examples/example.json not downloaded")
    return repo, install


def _bundled_example() -> list[dict]:
    repo, install = _assets()
    path = install.assets_dir / "examples" / "example.json"
    return json.loads(path.read_text())


class TestJsonConformance:
    """Compare our generated JSON against the bundled example's real shape."""

    def test_bundled_example_is_a_list_of_entries(self):
        data = _bundled_example()
        assert isinstance(data, list) and data

    def test_generated_entry_has_the_same_top_level_keys(self, tmp_path: Path):
        reference = _bundled_example()[0]

        spec = JobSpec(
            name="smoke_entry",
            map_path=tmp_path / "emd.map",
            resolution=2.99,
            contour_level=0.3,
            chains=[ChainSpec(SequenceEntry(SeqType.PROTEIN, "MKTA", 1), tmp_path)],
        )
        generated = build_input_json(spec)[0]

        # Every key CryoZeta's own example carries must be present in ours.
        # Extra keys in the reference that are optional are reported, not failed.
        required = {"name", "map_path", "resolution", "contour_level", "sequences"}
        missing = required - set(generated)
        assert not missing, f"generated JSON is missing {missing}"

        unexpected = set(reference) - set(generated) - {"em_file"}
        if unexpected:
            pytest.fail(
                "the bundled example carries keys we do not generate: "
                f"{sorted(unexpected)} -- the JSON contract may have changed"
            )

    def test_sequence_blocks_use_the_same_wrapper_keys(self, tmp_path: Path):
        reference = _bundled_example()[0]
        reference_keys = {k for block in reference["sequences"] for k in block}
        assert reference_keys <= {"proteinChain", "dnaSequence", "rnaSequence",
                                  "ligand", "ion"}

        spec = JobSpec(
            name="smoke_entry",
            map_path=tmp_path / "emd.map",
            resolution=2.99,
            contour_level=0.3,
            chains=[
                ChainSpec(SequenceEntry(SeqType.PROTEIN, "MKTA", 1), tmp_path),
                ChainSpec(SequenceEntry(SeqType.DNA, "ACGT", 1), None),
            ],
        )
        generated = build_input_json(spec)[0]
        for block in generated["sequences"]:
            assert len(block) == 1
            (chain_type, body), = block.items()
            assert chain_type in reference_keys or chain_type in {
                "proteinChain", "dnaSequence", "rnaSequence"
            }
            assert "sequence" in body and "count" in body

    def test_protein_msa_block_matches_reference_shape(self, tmp_path: Path):
        reference = _bundled_example()[0]
        reference_msa = None
        for block in reference["sequences"]:
            if "proteinChain" in block and "msa" in block["proteinChain"]:
                reference_msa = block["proteinChain"]["msa"]
                break
        if reference_msa is None:
            pytest.skip("bundled example has no protein msa block to compare")

        spec = JobSpec(
            name="smoke_entry",
            map_path=tmp_path / "emd.map",
            resolution=2.99,
            contour_level=0.3,
            chains=[ChainSpec(SequenceEntry(SeqType.PROTEIN, "MKTA", 1), tmp_path)],
        )
        ours = build_input_json(spec)[0]["sequences"][0]["proteinChain"]["msa"]
        missing = set(reference_msa) - set(ours)
        assert not missing, f"our msa block is missing {missing}"


class TestRealRun:
    """Full inference against the bundled example. Slow and GPU-bound."""

    def test_bundled_example_runs_end_to_end(self, settings, tmp_path: Path):
        report = run_preflight(settings)
        if not report.ok:
            failed = [c.name for c in report.checks if c.status == "fail"]
            pytest.skip(f"host is not CryoZeta-ready: {failed}")

        repo, install = _assets()
        example_path = install.assets_dir / "examples" / "example.json"
        entry = json.loads(example_path.read_text())[0]

        # Run the bundled example unmodified, through our own job machinery:
        # same directory layout, same subprocess handling, real CryoZeta.
        from app.paths import job_paths
        from app.runner import build_command
        from app.states import InferenceMode, RunMode
        from app.worker import JobManager

        store = JobStore(settings.db_path)
        try:
            job = store.create(
                entry_name=entry["name"],
                title="smoke: bundled example",
                note="",
                run_mode=RunMode.STANDARD,
                inference_mode=InferenceMode.COMBINED,
                gpu_index=report.gpu_indices[0] if report.gpu_indices else 0,
                resolution=float(entry["resolution"]),
                contour_level=float(entry["contour_level"]),
                total_seq_len=0,
                map_filename=Path(entry["map_path"]).name,
                overwrite=True,
                sequences=[],
            )

            paths = job_paths(settings.jobs_dir, job.id)
            paths.ensure()
            # Copy the bundled entry verbatim: this is CryoZeta's own input,
            # so a failure here is an installation problem, not a JSON problem.
            paths.input_json.write_text(json.dumps([entry], indent=2))

            manager = JobManager(
                settings,
                store,
                repo,
                report.gpu_indices or [0],
                pixi_env=report.recommended_env,
                command_builder=build_command,
            )
            manager.start()

            deadline = time.time() + 3 * 3600
            while time.time() < deadline:
                current = store.get(job.id)
                if current.status.is_terminal:
                    break
                time.sleep(15)
            else:
                pytest.fail("bundled example did not finish within 3 hours")

            manager.shutdown()
            final = store.get(job.id)
            assert final.status is JobStatus.COMPLETED, final.error_summary

            from app.results import collect_results

            results = collect_results(paths.output_dir)
            assert results.final_models, "no CryoZeta-Final mmCIF files were produced"
            assert results.scores, "no scores.csv was produced"
        finally:
            store.close()
