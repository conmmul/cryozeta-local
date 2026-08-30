"""Passphrase authentication for network-reachable deployments.

Used when the server is bound somewhere other than loopback -- typically an
institutional-VPN interface, where the VPN is the outer perimeter but is far
too broad to be the only control (a whole university can usually reach it).

Deliberately stdlib-only: no new dependencies, nothing to keep patched.

Design notes:

* the passphrase is stored only as a PBKDF2-HMAC-SHA256 hash with a random
  per-install salt, never in plaintext
* comparisons are constant-time
* sessions are stateless signed cookies (HMAC-SHA256 over the payload) keyed by
  a secret generated on first use and stored 0600, so restarting the server
  does not log everyone out but a stolen cookie cannot be forged
* failed logins are rate-limited per client address to make guessing a weak
  passphrase impractical
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# PBKDF2 rounds. High enough to make offline guessing painful, low enough that
# a login stays instant.
PBKDF2_ROUNDS = 480_000
SALT_BYTES = 16
SESSION_COOKIE = "cryozeta_session"
DEFAULT_SESSION_MAX_AGE = 14 * 24 * 3600  # 14 days

# Rate limiting.
MAX_FAILURES = 8
LOCKOUT_SECONDS = 300


class AuthError(Exception):
    pass


# --------------------------------------------------------------------------
# Passphrase hashing
# --------------------------------------------------------------------------
def hash_passphrase(passphrase: str, *, salt: bytes | None = None) -> str:
    """Return a ``pbkdf2$rounds$salt$digest`` string safe to store on disk."""
    if not passphrase or len(passphrase) < 8:
        raise AuthError("passphrase must be at least 8 characters")
    salt = salt or secrets.token_bytes(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256", passphrase.encode("utf-8"), salt, PBKDF2_ROUNDS
    )
    return "pbkdf2${}${}${}".format(
        PBKDF2_ROUNDS,
        base64.b64encode(salt).decode(),
        base64.b64encode(digest).decode(),
    )


def verify_passphrase(passphrase: str, stored: str) -> bool:
    """Constant-time check of ``passphrase`` against a stored hash."""
    if not passphrase or not stored:
        return False
    try:
        scheme, rounds_s, salt_b64, digest_b64 = stored.split("$")
        if scheme != "pbkdf2":
            return False
        rounds = int(rounds_s)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
    except (ValueError, TypeError):
        return False

    candidate = hashlib.pbkdf2_hmac(
        "sha256", passphrase.encode("utf-8"), salt, rounds
    )
    return hmac.compare_digest(candidate, expected)


# --------------------------------------------------------------------------
# Server secret
# --------------------------------------------------------------------------
def load_or_create_secret(path: Path) -> bytes:
    """Read the cookie-signing secret, creating it on first use.

    Written 0600 so other users on a shared workstation cannot forge sessions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        data = path.read_bytes().strip()
        if len(data) >= 32:
            return data

    secret = base64.b64encode(secrets.token_bytes(48))
    # Create with restrictive permissions from the outset, not afterwards.
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(secret)
    return secret


# --------------------------------------------------------------------------
# Session tokens
# --------------------------------------------------------------------------
def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64url(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def issue_session(secret: bytes, *, subject: str = "", now: float | None = None) -> str:
    payload = json.dumps(
        {"sub": subject, "iat": int(now if now is not None else time.time())},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    body = _b64url(payload)
    signature = hmac.new(secret, body.encode(), hashlib.sha256).digest()
    return f"{body}.{_b64url(signature)}"


def read_session(
    token: str | None,
    secret: bytes,
    *,
    max_age: int = DEFAULT_SESSION_MAX_AGE,
    now: float | None = None,
) -> str | None:
    """Return the session subject, or None if absent, forged or expired."""
    if not token or "." not in token:
        return None
    body, _, signature = token.partition(".")

    expected = hmac.new(secret, body.encode(), hashlib.sha256).digest()
    try:
        provided = _unb64url(signature)
    except (ValueError, TypeError):
        return None
    if not hmac.compare_digest(expected, provided):
        return None

    try:
        payload = json.loads(_unb64url(body))
        issued = int(payload["iat"])
    except (ValueError, TypeError, KeyError):
        return None

    current = now if now is not None else time.time()
    if current - issued > max_age:
        return None
    if issued - current > 300:  # clock skew tolerance; reject future-dated
        return None
    return str(payload.get("sub", ""))


# --------------------------------------------------------------------------
# Login rate limiting
# --------------------------------------------------------------------------
@dataclass
class _Attempts:
    failures: int = 0
    locked_until: float = 0.0


class RateLimiter:
    """Per-client lockout after repeated failures."""

    def __init__(
        self, max_failures: int = MAX_FAILURES, lockout: int = LOCKOUT_SECONDS
    ) -> None:
        self.max_failures = max_failures
        self.lockout = lockout
        self._state: dict[str, _Attempts] = {}
        self._lock = threading.Lock()

    def retry_after(self, client: str, *, now: float | None = None) -> int:
        """Seconds remaining in a lockout, or 0 if the client may try."""
        current = now if now is not None else time.monotonic()
        with self._lock:
            entry = self._state.get(client)
            if entry is None or entry.locked_until <= current:
                return 0
            return int(entry.locked_until - current) + 1

    def record_failure(self, client: str, *, now: float | None = None) -> None:
        current = now if now is not None else time.monotonic()
        with self._lock:
            entry = self._state.setdefault(client, _Attempts())
            entry.failures += 1
            if entry.failures >= self.max_failures:
                entry.locked_until = current + self.lockout
                entry.failures = 0

    def record_success(self, client: str) -> None:
        with self._lock:
            self._state.pop(client, None)


# --------------------------------------------------------------------------
# Paths that never require a session
# --------------------------------------------------------------------------
PUBLIC_PATHS = frozenset({"/login", "/logout", "/healthz"})
PUBLIC_PREFIXES = ("/static/",)


def is_public_path(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)
