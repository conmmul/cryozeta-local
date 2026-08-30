"""Passphrase authentication, sessions and the network-bind policy."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.auth import (
    SESSION_COOKIE,
    AuthError,
    RateLimiter,
    hash_passphrase,
    is_public_path,
    issue_session,
    load_or_create_secret,
    read_session,
    verify_passphrase,
)
from app.main import LanExposureError, create_app

PASSPHRASE = "correct-horse-battery"


@pytest.fixture
def auth_settings(settings):
    settings.passphrase_file = settings.data_root / "passphrase.hash"
    settings.passphrase_file.write_text(hash_passphrase(PASSPHRASE))
    return settings


@pytest.fixture
def client(auth_settings):
    app = create_app(auth_settings, start_workers=False)
    with TestClient(app, client=("127.0.0.1", 40000)) as c:
        yield c


class TestPassphraseHashing:
    def test_round_trip(self):
        stored = hash_passphrase(PASSPHRASE)
        assert verify_passphrase(PASSPHRASE, stored)

    def test_wrong_passphrase_rejected(self):
        assert not verify_passphrase("nope", hash_passphrase(PASSPHRASE))

    def test_plaintext_is_never_stored(self):
        assert PASSPHRASE not in hash_passphrase(PASSPHRASE)

    def test_salt_makes_hashes_unique(self):
        assert hash_passphrase(PASSPHRASE) != hash_passphrase(PASSPHRASE)

    def test_short_passphrase_refused(self):
        with pytest.raises(AuthError, match="at least 8"):
            hash_passphrase("short")

    @pytest.mark.parametrize("bad", ["", "garbage", "pbkdf2$notanint$a$b", "a$b$c$d"])
    def test_malformed_stored_hash_is_rejected(self, bad):
        assert not verify_passphrase(PASSPHRASE, bad)

    def test_empty_passphrase_never_verifies(self):
        assert not verify_passphrase("", hash_passphrase(PASSPHRASE))


class TestSessions:
    def test_round_trip(self):
        secret = b"x" * 48
        assert read_session(issue_session(secret, subject="a@b"), secret) == "a@b"

    def test_tampered_payload_rejected(self):
        secret = b"x" * 48
        token = issue_session(secret)
        body, _, sig = token.partition(".")
        assert read_session(f"{body}x.{sig}", secret) is None

    def test_forged_signature_rejected(self):
        secret = b"x" * 48
        body = issue_session(secret).partition(".")[0]
        assert read_session(f"{body}.deadbeef", secret) is None

    def test_token_from_a_different_secret_rejected(self):
        token = issue_session(b"a" * 48)
        assert read_session(token, b"b" * 48) is None

    def test_expired_session_rejected(self):
        secret = b"x" * 48
        old = issue_session(secret, now=time.time() - 10_000)
        assert read_session(old, secret, max_age=100) is None

    def test_future_dated_session_rejected(self):
        secret = b"x" * 48
        future = issue_session(secret, now=time.time() + 100_000)
        assert read_session(future, secret) is None

    @pytest.mark.parametrize("bad", [None, "", "nodot", "..", "a.b.c"])
    def test_malformed_tokens_rejected(self, bad):
        assert read_session(bad, b"x" * 48) is None


class TestSecretFile:
    def test_created_with_owner_only_permissions(self, tmp_path):
        path = tmp_path / "run" / "session.key"
        load_or_create_secret(path)
        assert oct(path.stat().st_mode & 0o777) == "0o600"

    def test_stable_across_calls(self, tmp_path):
        path = tmp_path / "session.key"
        assert load_or_create_secret(path) == load_or_create_secret(path)


class TestRateLimiter:
    def test_locks_out_after_repeated_failures(self):
        limiter = RateLimiter(max_failures=3, lockout=60)
        for _ in range(3):
            assert limiter.retry_after("1.2.3.4") == 0
            limiter.record_failure("1.2.3.4")
        assert limiter.retry_after("1.2.3.4") > 0

    def test_lockout_is_per_client(self):
        limiter = RateLimiter(max_failures=2, lockout=60)
        limiter.record_failure("1.1.1.1")
        limiter.record_failure("1.1.1.1")
        assert limiter.retry_after("1.1.1.1") > 0
        assert limiter.retry_after("2.2.2.2") == 0

    def test_success_clears_failures(self):
        limiter = RateLimiter(max_failures=3, lockout=60)
        limiter.record_failure("1.2.3.4")
        limiter.record_success("1.2.3.4")
        assert limiter.retry_after("1.2.3.4") == 0


class TestPublicPaths:
    @pytest.mark.parametrize("path", ["/login", "/logout", "/healthz", "/static/app.css"])
    def test_public(self, path):
        assert is_public_path(path)

    @pytest.mark.parametrize("path", ["/", "/new", "/jobs", "/jobs/abc/files/x", "/preflight"])
    def test_protected(self, path):
        assert not is_public_path(path)


class TestLoginFlow:
    def test_protected_page_redirects_to_login(self, client):
        response = client.get("/jobs", follow_redirects=False)
        assert response.status_code == 303
        assert "/login" in response.headers["location"]

    def test_login_page_is_reachable(self, client):
        assert client.get("/login").status_code == 200

    def test_correct_passphrase_grants_access(self, client):
        response = client.post(
            "/login", data={"passphrase": PASSPHRASE, "next": "/jobs"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert SESSION_COOKIE in response.cookies
        assert client.get("/jobs").status_code == 200

    def test_wrong_passphrase_denied(self, client):
        response = client.post(
            "/login", data={"passphrase": "wrong"}, follow_redirects=False
        )
        assert response.status_code == 401
        assert SESSION_COOKIE not in response.cookies

    def test_session_cookie_is_httponly(self, client):
        response = client.post(
            "/login", data={"passphrase": PASSPHRASE}, follow_redirects=False
        )
        assert "httponly" in response.headers["set-cookie"].lower()

    def test_logout_revokes_access(self, client):
        client.post("/login", data={"passphrase": PASSPHRASE}, follow_redirects=False)
        assert client.get("/jobs").status_code == 200
        client.post("/logout", follow_redirects=False)
        assert client.get("/jobs", follow_redirects=False).status_code == 303

    def test_forged_cookie_rejected(self, client):
        client.cookies.set(SESSION_COOKIE, "eyJzdWIiOiIifQ.deadbeef")
        assert client.get("/jobs", follow_redirects=False).status_code == 303

    def test_healthz_stays_public(self, client):
        assert client.get("/healthz").status_code == 200

    def test_job_file_download_is_protected(self, client):
        # The download endpoint serves job data, so it must not be public.
        response = client.get("/jobs/any-id/files/spec/input.json", follow_redirects=False)
        assert response.status_code == 303


class TestOpenRedirect:
    @pytest.mark.parametrize(
        "target", ["https://evil.example", "//evil.example", "http://evil.example/x"]
    )
    def test_absolute_targets_are_dropped(self, client, target):
        response = client.post(
            "/login", data={"passphrase": PASSPHRASE, "next": target},
            follow_redirects=False,
        )
        assert response.headers["location"] == "/"

    def test_relative_target_is_honoured(self, client):
        response = client.post(
            "/login", data={"passphrase": PASSPHRASE, "next": "/preflight"},
            follow_redirects=False,
        )
        assert response.headers["location"] == "/preflight"


class TestBindPolicy:
    def test_non_loopback_without_passphrase_is_refused(self, settings):
        settings.host = "0.0.0.0"
        settings.allow_lan = False
        with pytest.raises(LanExposureError, match="set-password"):
            create_app(settings, start_workers=False)

    def test_non_loopback_with_passphrase_is_allowed(self, auth_settings):
        auth_settings.host = "0.0.0.0"
        auth_settings.allow_lan = False
        assert create_app(auth_settings, start_workers=False) is not None

    def test_explicit_override_still_works(self, settings):
        settings.host = "0.0.0.0"
        settings.allow_lan = True
        assert create_app(settings, start_workers=False) is not None

    def test_loopback_needs_no_passphrase(self, settings):
        assert not settings.auth_required()
        assert create_app(settings, start_workers=False) is not None


class TestTailscaleBypass:
    def test_tailnet_user_is_not_asked_to_log_in_again(self, auth_settings):
        # Tailscale already authenticated them; a second passphrase prompt
        # would be pure friction.
        from app.identity import LOGIN_HEADER

        auth_settings.trust_tailscale_headers = True
        app = create_app(auth_settings, start_workers=False)
        with TestClient(app, client=("127.0.0.1", 40000)) as c:
            response = c.get("/jobs", headers={LOGIN_HEADER: "dave@lab.edu"})
        assert response.status_code == 200

    def test_without_the_header_a_login_is_still_required(self, auth_settings):
        auth_settings.trust_tailscale_headers = True
        app = create_app(auth_settings, start_workers=False)
        with TestClient(app, client=("127.0.0.1", 40000)) as c:
            assert c.get("/jobs", follow_redirects=False).status_code == 303
