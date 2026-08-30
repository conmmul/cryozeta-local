"""Integration tests driving the real web app against a fake inference binary.

The fake script accepts the same flags as ``inference_demo.sh``, asserts the
shape of the generated JSON, and writes an output tree with the same layout --
so the whole submit -> schedule -> run -> results path is exercised without
consuming GPU time.
"""

from __future__ import annotations

import io
import json
import os
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db import JobStore
from app.main import LanExposureError, create_app
from app.msa import PROTEIN_NON_PAIRING, PROTEIN_PAIRING
from app.states import JobStatus

from .conftest import FAKE_SCRIPT


@pytest.fixture(autouse=True)
def _make_fake_executable():
    FAKE_SCRIPT.chmod(0o755)


def fake_command_builder(**kwargs):
    """Swap the CryoZeta script for the fake one, keeping every real flag."""

    paths = kwargs["paths"]
    run_mode = kwargs["run_mode"]
    cmd = [
        "bash",
        str(FAKE_SCRIPT),
        "-i", str(paths.input_json),
        "-o", str(paths.output_dir),
        "-g", str(int(kwargs["gpu_index"])),
    ]
    if kwargs.get("pixi_env"):
        cmd += ["-e", kwargs["pixi_env"]]
    if run_mode.value == "large":
        cmd += ["-x", "0", "-r", "auto"]
    else:
        cmd += ["-m", kwargs["inference_mode"].value]
        if kwargs.get("overwrite"):
            cmd.append("--overwrite")
    return cmd


@pytest.fixture
def client(settings):
    app = create_app(settings, command_builder=fake_command_builder)
    with TestClient(app) as test_client:
        yield test_client


def submit(client, sample_map: Path, msa_zip: Path | None = None, **fields):
    files = [("map_file", (sample_map.name, sample_map.read_bytes(), "application/octet-stream"))]
    data = {
        "resolution": "2.99",
        "contour_level": "0.3",
        "title": "Integration test",
        "note": "",
        "gpu_index": "",
        "run_mode": "standard",
        "inference_mode": "combined",
        "seq_type": ["proteinChain"],
        "seq_value": ["MKTAYIAKQRQISFVKSHFSRQ"],
        "seq_count": ["1"],
        "msa_directory": [""],
    }
    data.update(fields)
    if msa_zip is not None:
        files.append(("msa_archive", (msa_zip.name, msa_zip.read_bytes(), "application/zip")))
    return client.post("/jobs", data=data, files=files, follow_redirects=False)


def wait_for_terminal(store: JobStore, job_id: str, timeout: float = 30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = store.get(job_id)
        if job and job.status.is_terminal:
            return job
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


@pytest.fixture
def msa_zip(make_zip):
    return make_zip(
        "msa.zip",
        {PROTEIN_NON_PAIRING: ">q\nMKTAYIAKQRQISFVKSHFSRQ\n",
         PROTEIN_PAIRING: ">q\nMKTAYIAKQRQISFVKSHFSRQ\n"},
    )


class TestPages:
    def test_new_job_page_renders(self, client):
        response = client.get("/new")
        assert response.status_code == 200
        assert "Cryo-EM map" in response.text

    def test_msa_requirements_are_explained(self, client):
        # The UI must say plainly that CryoZeta ships no MSA generation.
        response = client.get("/new")
        assert "does not generate MSAs" in response.text
        assert PROTEIN_NON_PAIRING in response.text

    def test_jobs_page_renders_when_empty(self, client):
        assert client.get("/jobs").status_code == 200

    def test_preflight_page_renders(self, client):
        assert client.get("/preflight").status_code == 200

    def test_healthz(self, client):
        assert client.get("/healthz").json()["ok"] is True


class TestSubmissionRejection:
    def test_missing_msa_rejected(self, client, sample_map):
        response = submit(client, sample_map)
        assert response.status_code == 400
        assert "has no MSA" in response.text

    def test_bad_map_extension_rejected(self, client, tmp_path, msa_zip):
        bad = tmp_path / "structure.pdb"
        bad.write_bytes(b"ATOM      1  N   MET A   1")
        response = submit(client, bad, msa_zip)
        assert response.status_code == 400
        assert "unsupported map file type" in response.text

    def test_invalid_sequence_rejected(self, client, sample_map, msa_zip):
        response = submit(client, sample_map, msa_zip, seq_value=["MKTAZZZJJJ"])
        assert response.status_code == 400
        assert "invalid characters" in response.text

    def test_zero_contour_rejected(self, client, sample_map, msa_zip):
        response = submit(client, sample_map, msa_zip, contour_level="0")
        assert response.status_code == 400
        assert "non-zero" in response.text


class TestFullWorkflow:
    def test_job_runs_to_completion(self, client, settings, sample_map, msa_zip):
        response = submit(client, sample_map, msa_zip)
        assert response.status_code == 303
        job_id = response.headers["location"].rsplit("/", 1)[-1]

        store = JobStore(settings.db_path)
        try:
            job = wait_for_terminal(store, job_id)
            assert job.status is JobStatus.COMPLETED, job.error_summary
            assert job.exit_code == 0
            assert job.gpu_index is not None
        finally:
            store.close()

    def test_generated_json_passes_the_contract_check(
        self, client, settings, sample_map, msa_zip
    ):
        # The fake script exits 66 if the JSON does not match CryoZeta's shape,
        # so a completed job *is* the assertion that our JSON is well-formed.
        response = submit(client, sample_map, msa_zip)
        job_id = response.headers["location"].rsplit("/", 1)[-1]
        store = JobStore(settings.db_path)
        try:
            job = wait_for_terminal(store, job_id)
            assert job.status is JobStatus.COMPLETED
        finally:
            store.close()

    def test_results_page_lists_outputs(self, client, settings, sample_map, msa_zip):
        response = submit(client, sample_map, msa_zip)
        job_id = response.headers["location"].rsplit("/", 1)[-1]
        store = JobStore(settings.db_path)
        try:
            wait_for_terminal(store, job_id)
        finally:
            store.close()

        page = client.get(f"/jobs/{job_id}/results")
        assert page.status_code == 200
        assert "CryoZeta-Final" in page.text
        assert "scores.csv" in page.text
        assert "summary_confidence" in page.text

    def test_individual_file_download(self, client, settings, sample_map, msa_zip):
        response = submit(client, sample_map, msa_zip)
        job_id = response.headers["location"].rsplit("/", 1)[-1]
        store = JobStore(settings.db_path)
        try:
            job = wait_for_terminal(store, job_id)
        finally:
            store.close()

        path = f"output/{job.entry_name}/CryoZeta-Final/{job.entry_name}_sample_0.cif"
        download = client.get(f"/jobs/{job_id}/files/{path}")
        assert download.status_code == 200
        assert "data_" in download.text

    def test_zip_download_contains_everything(
        self, client, settings, sample_map, msa_zip
    ):
        response = submit(client, sample_map, msa_zip)
        job_id = response.headers["location"].rsplit("/", 1)[-1]
        store = JobStore(settings.db_path)
        try:
            wait_for_terminal(store, job_id)
        finally:
            store.close()

        archive = client.get(f"/jobs/{job_id}/download")
        assert archive.status_code == 200
        with zipfile.ZipFile(io.BytesIO(archive.content)) as zf:
            names = zf.namelist()
        assert any("spec/input.json" in n for n in names)
        assert any("logs/job.log" in n for n in names)
        assert any("CryoZeta-Final" in n for n in names)

    def test_large_mode_produces_combined_cif(
        self, client, settings, sample_map, msa_zip
    ):
        response = submit(client, sample_map, msa_zip, run_mode="large")
        job_id = response.headers["location"].rsplit("/", 1)[-1]
        store = JobStore(settings.db_path)
        try:
            job = wait_for_terminal(store, job_id)
            assert job.status is JobStatus.COMPLETED
        finally:
            store.close()

        page = client.get(f"/jobs/{job_id}/results")
        assert "combined.cif" in page.text


class TestFailureHandling:
    def test_nonzero_exit_marks_failed_with_summary(
        self, client, settings, sample_map, msa_zip, monkeypatch
    ):
        monkeypatch.setenv("FAKE_EXIT_CODE", "1")
        monkeypatch.setenv("FAKE_STDERR", "RuntimeError: CUDA out of memory")

        response = submit(client, sample_map, msa_zip)
        job_id = response.headers["location"].rsplit("/", 1)[-1]
        store = JobStore(settings.db_path)
        try:
            job = wait_for_terminal(store, job_id)
            assert job.status is JobStatus.FAILED
            assert job.exit_code == 1
            assert "out of memory" in job.error_summary
        finally:
            store.close()


class TestCancellation:
    def test_running_job_can_be_cancelled(
        self, client, settings, sample_map, msa_zip, monkeypatch
    ):
        monkeypatch.setenv("FAKE_SLEEP", "30")
        response = submit(client, sample_map, msa_zip)
        job_id = response.headers["location"].rsplit("/", 1)[-1]

        store = JobStore(settings.db_path)
        try:
            deadline = time.time() + 15
            while time.time() < deadline:
                job = store.get(job_id)
                if job and job.status is JobStatus.RUNNING and job.pid:
                    break
                time.sleep(0.2)
            else:
                pytest.fail("job never started running")

            pid = store.get(job_id).pid
            client.post(f"/jobs/{job_id}/cancel", follow_redirects=False)
            job = wait_for_terminal(store, job_id, timeout=30)
            assert job.status is JobStatus.CANCELLED

            # The whole process group must be gone, not just the bash wrapper.
            time.sleep(0.5)
            with pytest.raises(OSError):
                os.killpg(os.getpgid(pid), 0)
        finally:
            store.close()


class TestRerun:
    def test_rerun_creates_an_independent_job(
        self, client, settings, sample_map, msa_zip
    ):
        response = submit(client, sample_map, msa_zip)
        original_id = response.headers["location"].rsplit("/", 1)[-1]
        store = JobStore(settings.db_path)
        try:
            wait_for_terminal(store, original_id)

            rerun = client.post(f"/jobs/{original_id}/rerun", follow_redirects=False)
            new_id = rerun.headers["location"].rsplit("/", 1)[-1]
            assert new_id != original_id

            new_job = wait_for_terminal(store, new_id)
            assert new_job.status is JobStatus.COMPLETED
        finally:
            store.close()

        # The rerun's JSON must point into its own directory, not the original's.
        spec = settings.jobs_dir / new_id / "spec" / "input.json"
        entry = json.loads(spec.read_text())[0]
        assert new_id in entry["map_path"]
        assert original_id not in entry["map_path"]


class TestSecurityEndpoints:
    def test_path_traversal_on_file_download_blocked(
        self, client, settings, sample_map, msa_zip
    ):
        response = submit(client, sample_map, msa_zip)
        job_id = response.headers["location"].rsplit("/", 1)[-1]
        blocked = client.get(f"/jobs/{job_id}/files/../../../../etc/passwd")
        assert blocked.status_code in (400, 404)

    def test_unknown_job_is_404(self, client):
        assert client.get("/jobs/does-not-exist").status_code == 404


class TestNetworkBinding:
    def test_refuses_non_loopback_without_optin(self, settings):
        settings.host = "0.0.0.0"
        settings.allow_lan = False
        with pytest.raises(LanExposureError, match="Refusing to bind"):
            create_app(settings, start_workers=False)

    def test_allows_non_loopback_with_explicit_optin(self, settings):
        settings.host = "0.0.0.0"
        settings.allow_lan = True
        assert create_app(settings, start_workers=False) is not None


class TestRestartRecovery:
    def test_orphaned_running_job_becomes_interrupted(self, settings):
        store = JobStore(settings.db_path)
        try:
            job = store.create(
                entry_name="orphan", title="Orphan", note="",
                run_mode=__import__("app.states", fromlist=["RunMode"]).RunMode.STANDARD,
                inference_mode=__import__(
                    "app.states", fromlist=["InferenceMode"]
                ).InferenceMode.COMBINED,
                gpu_index=0, resolution=3.0, contour_level=0.3,
                total_seq_len=10, map_filename="a.map", overwrite=False, sequences=[],
            )
            store.set_status(job.id, JobStatus.RUNNING)
        finally:
            store.close()

        # Starting the app performs crash recovery.
        create_app(settings, start_workers=False)
        with TestClient(create_app(settings, start_workers=False)):
            pass

        store = JobStore(settings.db_path)
        try:
            assert store.get(job.id).status is JobStatus.INTERRUPTED
        finally:
            store.close()
