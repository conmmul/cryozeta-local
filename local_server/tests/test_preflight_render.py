"""Rendering of the System page, including the GPU-present branch.

Development machines usually have no NVIDIA GPU, so the ``{% if
report.nvidia.gpus %}`` branch of preflight.html would otherwise never be
exercised by any test. These fake a GPU inventory so both branches render.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.discovery import GpuInfo, NvidiaInfo, query_nvidia_cached, select_pixi_env
from app.main import create_app

TWO_GPUS = NvidiaInfo(
    available=True,
    driver_version="550.54.14",
    driver_cuda_version="12.4",
    gpus=[
        GpuInfo(0, "NVIDIA A100-SXM4-40GB", 40960, "8.0"),
        GpuInfo(1, "NVIDIA RTX A6000", 49140, "8.6"),
    ],
)

SMALL_GPU = NvidiaInfo(
    available=True,
    driver_version="470.82.01",
    driver_cuda_version="11.4",
    gpus=[GpuInfo(0, "NVIDIA GeForce RTX 2080 Ti", 11264, "7.5")],
)

NO_GPU = NvidiaInfo(available=False, error="nvidia-smi not found on PATH")


def client_with(settings, info):
    with patch("app.main.query_nvidia_cached", return_value=info), patch(
        "app.preflight.query_nvidia_cached", return_value=info
    ):
        app = create_app(settings, start_workers=False)
        with TestClient(app) as c:
            yield c


@pytest.fixture
def gpu_client(settings):
    yield from client_with(settings, TWO_GPUS)


class TestSystemPageWithGpus:
    def test_renders(self, gpu_client):
        assert gpu_client.get("/preflight").status_code == 200

    def test_lists_every_gpu(self, gpu_client):
        text = gpu_client.get("/preflight").text
        assert "A100-SXM4-40GB" in text
        assert "RTX A6000" in text

    def test_shows_vram_and_compute_capability(self, gpu_client):
        text = gpu_client.get("/preflight").text
        assert "40.0 GiB" in text
        assert "8.6" in text

    def test_shows_driver_and_cuda_ceiling(self, gpu_client):
        text = gpu_client.get("/preflight").text
        assert "550.54.14" in text
        assert "12.4" in text

    def test_new_job_page_offers_a_gpu_selector(self, gpu_client):
        text = gpu_client.get("/new").text
        assert "A100-SXM4-40GB" in text
        assert "RTX A6000" in text

    def test_nav_reports_gpu_count(self, gpu_client):
        assert "2 GPUs" in gpu_client.get("/jobs").text


class TestSystemPageWithoutGpus:
    def test_renders_and_explains(self, settings):
        for c in client_with(settings, NO_GPU):
            response = c.get("/preflight")
            assert response.status_code == 200
            assert "No NVIDIA GPU detected" in response.text
            assert "nvidia-smi not found" in response.text


class TestUnderpoweredGpu:
    def test_below_32gb_is_warned_not_failed(self, settings):
        for c in client_with(settings, SMALL_GPU):
            text = c.get("/preflight").text
            assert "RTX 2080 Ti" in text
            # 11 GiB is under CryoZeta's documented 32 GB minimum.
            assert "32 GB minimum" in text


class TestEnvironmentSelection:
    @pytest.mark.parametrize(
        "info,expected",
        [
            (TWO_GPUS, "default"),
            (SMALL_GPU, "cu11"),
            (NO_GPU, "default"),
            (
                NvidiaInfo(
                    available=True, driver_version="580", driver_cuda_version="13.0",
                    gpus=[GpuInfo(0, "NVIDIA B200", 183000, "10.0")],
                ),
                "cu13",
            ),
        ],
    )
    def test_matches_cryozeta_detection_logic(self, info, expected):
        assert select_pixi_env(info) == expected


class TestGpuProbeCaching:
    def test_a_failed_probe_is_not_cached_for_the_process_lifetime(self):
        """A driver that appears after startup must be picked up.

        Regression: GPU info used to be read once in create_app, so a server
        started before nvidia-smi was ready reported "no GPU" forever.
        """
        import app.discovery as discovery

        discovery._NVIDIA_CACHE = None
        calls = []

        def flaky():
            calls.append(1)
            return NO_GPU if len(calls) == 1 else TWO_GPUS

        with patch.object(discovery, "query_nvidia", side_effect=flaky):
            first = query_nvidia_cached(ttl=0)
            assert not first.available
            # Expire the short negative-result window and probe again.
            discovery._NVIDIA_CACHE = None
            second = query_nvidia_cached(ttl=0)
            assert second.available
            assert len(second.gpus) == 2

        discovery._NVIDIA_CACHE = None
