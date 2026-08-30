"""Who submitted a job.

The server still has no accounts and no passwords. When it is published to a
tailnet, identity comes from Tailscale, which has already authenticated the
user before the request reaches us. We only *record* that identity so jobs are
attributable and a shared GPU queue is legible.

Two ways in, in order of preference:

1. ``tailscale serve`` terminates TLS locally and injects identity headers.
   Those headers are only trusted when the operator has explicitly enabled
   Tailscale mode **and** the request arrived from loopback -- i.e. from the
   local proxy. Without both conditions a header is just user input and is
   ignored, otherwise anyone able to reach the port could impersonate a
   colleague by setting a header themselves.

2. The app is bound directly to the tailnet address. Then the peer IP is real
   and ``tailscale whois`` resolves it.

This is deliberately *attribution*, not authorisation. Everyone on the tailnet
can see and cancel everything; the tailnet boundary is the security control.
"""

from __future__ import annotations

import ipaddress
import json
import shutil
import subprocess
import time

# Injected by `tailscale serve` for tailnet (non-Funnel) requests.
LOGIN_HEADER = "tailscale-user-login"
NAME_HEADER = "tailscale-user-name"

ANONYMOUS = ""

_WHOIS_CACHE: dict[str, tuple[float, str]] = {}
_WHOIS_TTL = 300.0


def is_loopback(host: str | None) -> bool:
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def is_tailscale_address(host: str | None) -> bool:
    """True for Tailscale's CGNAT range (100.64.0.0/10) or its IPv6 ULA."""
    if not host:
        return False
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    if addr.version == 4:
        return addr in ipaddress.ip_network("100.64.0.0/10")
    return addr in ipaddress.ip_network("fd7a:115c:a1e0::/48")


def sanitize_identity(raw: str | None, max_length: int = 128) -> str:
    """Identities are displayed, so strip anything that is not printable."""
    if not raw:
        return ANONYMOUS
    cleaned = "".join(ch for ch in str(raw) if ch.isprintable() and ch not in "<>\"'")
    return " ".join(cleaned.split())[:max_length]


def whois(peer_ip: str, *, timeout: float = 3.0) -> str:
    """Ask the local tailscaled who owns ``peer_ip``.

    Never raises; an unknown peer resolves to the empty string.
    """
    if not peer_ip:
        return ANONYMOUS

    now = time.monotonic()
    if cached := _WHOIS_CACHE.get(peer_ip):
        cached_at, value = cached
        if now - cached_at < _WHOIS_TTL:
            return value

    binary = shutil.which("tailscale")
    if binary is None:
        return ANONYMOUS

    try:
        proc = subprocess.run(  # noqa: S603 - argv array, shell=False
            [binary, "whois", "--json", peer_ip],
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        if proc.returncode != 0:
            return ANONYMOUS
        payload = json.loads(proc.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return ANONYMOUS

    profile = payload.get("UserProfile") or {}
    identity = sanitize_identity(
        profile.get("LoginName") or profile.get("DisplayName")
    )
    _WHOIS_CACHE[peer_ip] = (now, identity)
    return identity


def resolve_submitter(
    *,
    client_host: str | None,
    headers,
    trust_proxy_headers: bool,
) -> str:
    """Best-effort identity for the person making this request.

    Returns "" when identity cannot be established, which is the normal case
    for a purely local install.
    """
    # Case 1: behind `tailscale serve`, which connects from loopback.
    if trust_proxy_headers and is_loopback(client_host):
        login = headers.get(LOGIN_HEADER) or headers.get(NAME_HEADER)
        if login:
            return sanitize_identity(login)
        return ANONYMOUS

    # Case 2: bound directly to the tailnet; the peer address is trustworthy.
    if is_tailscale_address(client_host):
        return whois(client_host)

    return ANONYMOUS


def describe_source(client_host: str | None, trust_proxy_headers: bool) -> str:
    """Short explanation of how identity was (or was not) determined."""
    if trust_proxy_headers and is_loopback(client_host):
        return "tailscale serve"
    if is_tailscale_address(client_host):
        return "tailnet peer"
    if is_loopback(client_host):
        return "local"
    return "unidentified"
