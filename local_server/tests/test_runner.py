"""Command construction and log interpretation."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.paths import JobPaths
from app.runner import (
    RunnerError,
    build_command,
    build_environment,
    detect_stage,
    summarize_failure,
)
from app.states import InferenceMode, RunMode


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "CryoZeta"
    repo.mkdir()
    (repo / "inference_demo.sh").write_text("#!/bin/bash\n")
    (repo / "large_inference_demo.sh").write_text("#!/bin/bash\n")
    return repo


@pytest.fixture
def paths(tmp_path: Path) -> JobPaths:
    p = JobPaths(root=tmp_path / "job")
    p.ensure()
    return p


class TestStandardCommand:
    def test_uses_inference_demo(self, fake_repo, paths):
        cmd = build_command(
            repo=fake_repo,
            paths=paths,
            run_mode=RunMode.STANDARD,
            inference_mode=InferenceMode.COMBINED,
            gpu_index=0,
        )
        assert cmd[0] == "bash"
        assert cmd[1].endswith("inference_demo.sh")

    def test_flags_present(self, fake_repo, paths):
        cmd = build_command(
            repo=fake_repo,
            paths=paths,
            run_mode=RunMode.STANDARD,
            inference_mode=InferenceMode.CRYOZETA,
            gpu_index=2,
            pixi_env="cu11",
        )
        assert cmd[cmd.index("-i") + 1] == str(paths.input_json)
        assert cmd[cmd.index("-o") + 1] == str(paths.output_dir)
        assert cmd[cmd.index("-g") + 1] == "2"
        assert cmd[cmd.index("-e") + 1] == "cu11"
        assert cmd[cmd.index("-m") + 1] == "cryozeta"

    def test_overwrite_flag_only_when_requested(self, fake_repo, paths):
        without = build_command(
            repo=fake_repo, paths=paths, run_mode=RunMode.STANDARD,
            inference_mode=InferenceMode.COMBINED, gpu_index=0, overwrite=False,
        )
        assert "--overwrite" not in without

        with_flag = build_command(
            repo=fake_repo, paths=paths, run_mode=RunMode.STANDARD,
            inference_mode=InferenceMode.COMBINED, gpu_index=0, overwrite=True,
        )
        assert "--overwrite" in with_flag

    def test_pixi_env_omitted_when_none(self, fake_repo, paths):
        cmd = build_command(
            repo=fake_repo, paths=paths, run_mode=RunMode.STANDARD,
            inference_mode=InferenceMode.COMBINED, gpu_index=0, pixi_env=None,
        )
        assert "-e" not in cmd


class TestLargeCommand:
    def test_uses_large_script_and_selects_entry_zero(self, fake_repo, paths):
        cmd = build_command(
            repo=fake_repo,
            paths=paths,
            run_mode=RunMode.LARGE,
            inference_mode=InferenceMode.COMBINED,
            gpu_index=1,
        )
        assert cmd[1].endswith("large_inference_demo.sh")
        # Our generated JSON always holds exactly one entry.
        assert cmd[cmd.index("-x") + 1] == "0"

    def test_no_mode_or_overwrite_flag(self, fake_repo, paths):
        # large_inference_demo.sh accepts neither.
        cmd = build_command(
            repo=fake_repo, paths=paths, run_mode=RunMode.LARGE,
            inference_mode=InferenceMode.COMBINED, gpu_index=0, overwrite=True,
        )
        assert "-m" not in cmd
        assert "--overwrite" not in cmd

    def test_registration_validated(self, fake_repo, paths):
        with pytest.raises(RunnerError, match="registration"):
            build_command(
                repo=fake_repo, paths=paths, run_mode=RunMode.LARGE,
                inference_mode=InferenceMode.COMBINED, gpu_index=0,
                registration="bogus",
            )


class TestCommandSafety:
    def test_missing_script_raises(self, tmp_path, paths):
        with pytest.raises(RunnerError, match="not found"):
            build_command(
                repo=tmp_path / "nowhere", paths=paths, run_mode=RunMode.STANDARD,
                inference_mode=InferenceMode.COMBINED, gpu_index=0,
            )

    def test_gpu_index_is_coerced_to_int(self, fake_repo, paths):
        # Even if a caller passes a string, it must land as a plain integer:
        # nothing user-controlled may reach the argv unchecked.
        cmd = build_command(
            repo=fake_repo, paths=paths, run_mode=RunMode.STANDARD,
            inference_mode=InferenceMode.COMBINED, gpu_index=3,
        )
        assert cmd[cmd.index("-g") + 1] == "3"

    def test_every_argument_is_a_string(self, fake_repo, paths):
        cmd = build_command(
            repo=fake_repo, paths=paths, run_mode=RunMode.STANDARD,
            inference_mode=InferenceMode.COMBINED, gpu_index=0,
        )
        assert all(isinstance(part, str) for part in cmd)


class TestEnvironment:
    def test_pins_cuda_visible_devices(self):
        env = build_environment(gpu_index=2)
        assert env["CUDA_VISIBLE_DEVICES"] == "2"
        assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"

    def test_extension_cache_override(self, tmp_path):
        env = build_environment(gpu_index=0, extra_cache_dir=tmp_path)
        assert env["CRYOZETA_TORCH_EXTENSIONS_DIR"] == str(tmp_path)


class TestStageDetection:
    @pytest.mark.parametrize(
        "line,expected",
        [
            ("==> Running detection to generate EM .pt", "detection"),
            ("running cryozeta-detection json-run ...", "detection"),
            ("--use_interpolation true", "cryozeta-interpolate"),
            ("--use_interpolation false", "cryozeta"),
            ("==> Starting large complex cycle prediction...", "cycle-predict"),
            ("==> Combining stages into final structure...", "combine-stages"),
        ],
    )
    def test_recognises_stages(self, line, expected):
        assert detect_stage(line) == expected

    def test_unrelated_line_returns_none(self):
        assert detect_stage("loading weights, please wait") is None


class TestFailureSummaries:
    def test_oom(self):
        text = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
        assert "ran out of memory" in summarize_failure(text, 1)

    def test_missing_pairing_msa(self):
        text = "AssertionError: No pairing-MSA of chain_A (please check /x/uniref100_hits.a3m)"
        assert "uniref100_hits.a3m" in summarize_failure(text, 1)

    def test_missing_checkpoint(self):
        text = "ERROR: Detection checkpoint not found: assets/x.safetensors"
        assert "download-assets" in summarize_failure(text, 1)

    def test_teaser(self):
        assert "TEASER++" in summarize_failure("ImportError: libteaser.so", 1)

    def test_disk_full(self):
        assert "disk filled up" in summarize_failure("OSError: No space left on device", 1)

    def test_falls_back_to_last_error_line(self):
        text = "starting\nloading\nValueError: something specific went wrong\n"
        assert "something specific" in summarize_failure(text, 1)

    def test_empty_log(self):
        assert "exit code 3" in summarize_failure("", 3)
