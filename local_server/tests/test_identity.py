"""Submitter identity, and the conditions under which it may be trusted.

The security-relevant property: an identity header is only meaningful when it
was injected by a local `tailscale serve` proxy. Trusting it unconditionally
would let anyone who can reach the port claim to be a colleague.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.db import JobStore
from app.identity import (
    LOGIN_HEADER,
    describe_source,
    is_loopback,
    is_tailscale_address,
    resolve_submitter,
    sanitize_identity,
    whois,
)
from app.main import create_app
from app.states import InferenceMode, RunMode

SPOOFED = {LOGIN_HEADER: "director@lab.edu"}


class TestAddressClassification:
    @pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
    def test_loopback(self, host):
        assert is_loopback(host)

    @pytest.mark.parametrize("host", ["10.0.0.5", "192.168.1.7", "8.8.8.8", None, ""])
    def test_not_loopback(self, host):
        assert not is_loopback(host)

    @pytest.mark.parametrize("host", ["100.64.0.1", "100.101.102.103", "100.127.255.254"])
    def test_tailscale_range(self, host):
        assert is_tailscale_address(host)

    @pytest.mark.parametrize(
        "host", ["100.128.0.1", "99.64.0.1", "192.168.1.1", "10.0.0.1", "not-an-ip"]
    )
    def test_outside_tailscale_range(self, host):
        # 100.128.0.1 is just past the end of 100.64.0.0/10.
        assert not is_tailscale_address(host)

    def test_tailscale_ipv6_ula(self):
        assert is_tailscale_address("fd7a:115c:a1e0::1")


class TestHeaderTrust:
    def test_header_ignored_when_mode_disabled(self):
        # The single most important case: identity headers must do nothing
        # unless the operator has turned tailnet mode on.
        assert (
            resolve_submitter(
                client_host="127.0.0.1", headers=SPOOFED, trust_proxy_headers=False
            )
            == ""
        )

    def test_header_ignored_from_non_loopback_peer(self):
        # Enabled, but the request did not come from the local proxy: a remote
        # client setting the header itself must not be believed.
        assert (
            resolve_submitter(
                client_host="192.168.1.50", headers=SPOOFED, trust_proxy_headers=True
            )
            == ""
        )

    def test_header_trusted_from_local_proxy(self):
        assert (
            resolve_submitter(
                client_host="127.0.0.1", headers=SPOOFED, trust_proxy_headers=True
            )
            == "director@lab.edu"
        )

    def test_tailnet_peer_header_is_still_not_trusted(self):
        # A tailnet peer is authenticated, but its *header* is still user
        # input; identity for direct connections comes from whois instead.
        with patch("app.identity.whois", return_value="real@lab.edu") as mock:
            result = resolve_submitter(
                client_host="100.64.0.9", headers=SPOOFED, trust_proxy_headers=False
            )
        assert result == "real@lab.edu"
        mock.assert_called_once_with("100.64.0.9")

    def test_missing_header_is_anonymous_not_an_error(self):
        assert (
            resolve_submitter(
                client_host="127.0.0.1", headers={}, trust_proxy_headers=True
            )
            == ""
        )


class TestSanitisation:
    def test_strips_control_characters(self):
        assert sanitize_identity("alice\x00\x1b[31m@lab.edu") == "alice[31m@lab.edu"

    def test_strips_html_metacharacters(self):
        assert "<" not in sanitize_identity("<script>alert(1)</script>")

    def test_collapses_whitespace_and_truncates(self):
        assert sanitize_identity("  a   b  ") == "a b"
        assert len(sanitize_identity("x" * 500)) == 128

    def test_none_is_anonymous(self):
        assert sanitize_identity(None) == ""


class TestWhois:
    def test_absent_tailscale_binary_is_anonymous(self):
        with patch("app.identity.shutil.which", return_value=None):
            assert whois("100.64.0.1") == ""

    def test_parses_login_name(self):
        payload = '{"UserProfile": {"LoginName": "bob@lab.edu", "DisplayName": "Bob"}}'
        with patch("app.identity.shutil.which", return_value="/usr/bin/tailscale"), patch(
            "app.identity.subprocess.run"
        ) as run:
            run.return_value.returncode = 0
            run.return_value.stdout = payload
            assert whois("100.64.0.77") == "bob@lab.edu"

    def test_malformed_output_is_anonymous(self):
        with patch("app.identity.shutil.which", return_value="/usr/bin/tailscale"), patch(
            "app.identity.subprocess.run"
        ) as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "not json"
            assert whois("100.64.0.78") == ""

    def test_subprocess_failure_is_anonymous(self):
        with patch("app.identity.shutil.which", return_value="/usr/bin/tailscale"), patch(
            "app.identity.subprocess.run", side_effect=OSError("boom")
        ):
            assert whois("100.64.0.79") == ""

    def test_invocation_uses_argv_array_without_a_shell(self):
        with patch("app.identity.shutil.which", return_value="/usr/bin/tailscale"), patch(
            "app.identity.subprocess.run"
        ) as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "{}"
            whois("100.64.0.80")
            args, kwargs = run.call_args
        assert isinstance(args[0], list)
        assert kwargs["shell"] is False


class TestSourceDescription:
    def test_describes_each_mode(self):
        assert describe_source("127.0.0.1", True) == "tailscale serve"
        assert describe_source("100.64.0.1", False) == "tailnet peer"
        assert describe_source("127.0.0.1", False) == "local"
        assert describe_source("192.168.1.2", False) == "unidentified"


class TestAttributionEndToEnd:
    def test_job_records_its_submitter(self, settings, sample_map, make_zip):
        from app.msa import PROTEIN_NON_PAIRING

        settings.trust_tailscale_headers = True
        archive = make_zip("msa.zip", {PROTEIN_NON_PAIRING: ">q\nMKTA\n"})

        from .test_integration_fake import fake_command_builder

        app = create_app(settings, command_builder=fake_command_builder)
        # TestClient reports its peer as the literal string "testclient",
        # which is correctly rejected as non-loopback; supply a real address
        # so this exercises the trusted-proxy path.
        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            response = client.post(
                "/jobs",
                data={
                    "resolution": "2.99",
                    "contour_level": "0.3",
                    "title": "Attributed job",
                    "note": "",
                    "gpu_index": "",
                    "run_mode": "standard",
                    "inference_mode": "combined",
                    "seq_type": ["proteinChain"],
                    "seq_value": ["MKTAYIAK"],
                    "seq_count": ["1"],
                    "msa_directory": [""],
                },
                files=[
                    ("map_file", (sample_map.name, sample_map.read_bytes(), "application/octet-stream")),
                    ("msa_archive", ("msa.zip", archive.read_bytes(), "application/zip")),
                ],
                headers={LOGIN_HEADER: "carol@lab.edu"},
                follow_redirects=False,
            )
            assert response.status_code == 303
            job_id = response.headers["location"].rsplit("/", 1)[-1]

        store = JobStore(settings.db_path)
        try:
            assert store.get(job_id).submitted_by == "carol@lab.edu"
        finally:
            store.close()

    def test_submitter_is_blank_for_a_purely_local_install(
        self, settings, store, sample_map
    ):
        # Default configuration: no identity, no behaviour change.
        assert settings.trust_tailscale_headers is False
        job = store.create(
            entry_name="e", title="t", note="", run_mode=RunMode.STANDARD,
            inference_mode=InferenceMode.COMBINED, gpu_index=None, resolution=3.0,
            contour_level=0.3, total_seq_len=1, map_filename="a.map",
            overwrite=False, sequences=[],
        )
        assert job.submitted_by == ""
