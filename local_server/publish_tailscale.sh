#!/usr/bin/env bash
#
# Publish the local server to a Tailscale tailnet WITHOUT touching this
# machine's networking.
#
#   ./publish_tailscale.sh start     # connect and publish
#   ./publish_tailscale.sh status
#   ./publish_tailscale.sh stop      # withdraw and disconnect
#
# Why this exists
# ---------------
# A normal `tailscale up` creates a TUN device, edits the kernel routing table
# and rewrites /etc/resolv.conf for MagicDNS. On a server you reach through an
# institutional VPN, that can break DNS for internal hostnames or capture
# routes to campus subnets -- i.e. it can cut your SSH access.
#
# This script instead runs tailscaled in **userspace networking mode**:
#
#   * no TUN device is created
#   * the kernel routing table is never modified
#   * /etc/resolv.conf is never touched
#   * it does not need root
#
# Tailscale can then only do one thing: accept inbound connections for the
# tailnet and proxy them to 127.0.0.1. It has no ability to affect your
# existing SSH session, your VPN, or anything else on the host.
#
# The trade-off: this machine cannot *reach out* to other tailnet nodes by
# their tailnet address. That is irrelevant here -- we only need inbound.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${CRYOZETA_WEB_PORT:-8000}"
DATA_ROOT="${CRYOZETA_WEB_DATA_ROOT:-$HOME/cryozeta-web-data}"
STATE_DIR="$DATA_ROOT/tailscale"
SOCKET="$STATE_DIR/tailscaled.sock"
PID_FILE="$STATE_DIR/tailscaled.pid"
LOG_FILE="$STATE_DIR/tailscaled.log"
HOSTNAME_TAG="${CRYOZETA_TS_HOSTNAME:-cryozeta-$(hostname -s 2>/dev/null || echo server)}"

ACTION="${1:-start}"

ts() { tailscale --socket="$SOCKET" "$@"; }

need_binaries() {
    for binary in tailscale tailscaled; do
        command -v "$binary" >/dev/null 2>&1 || {
            echo "error: '$binary' not found." >&2
            echo "       Install Tailscale: https://tailscale.com/download/linux" >&2
            echo "       You do NOT need to run 'tailscale up' or enable the system" >&2
            echo "       service -- this script runs its own userspace daemon." >&2
            exit 69
        }
    done
}

daemon_running() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

case "$ACTION" in
start)
    need_binaries
    mkdir -p "$STATE_DIR"
    chmod 700 "$STATE_DIR"

    if daemon_running; then
        echo "==> userspace tailscaled already running (pid $(cat "$PID_FILE"))"
    else
        echo "==> Starting tailscaled in userspace mode (no TUN, no route changes)"
        nohup tailscaled \
            --tun=userspace-networking \
            --socket="$SOCKET" \
            --statedir="$STATE_DIR" \
            >>"$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        sleep 2
        daemon_running || {
            echo "error: tailscaled failed to start. Last lines of $LOG_FILE:" >&2
            tail -20 "$LOG_FILE" >&2
            rm -f "$PID_FILE"
            exit 1
        }
    fi

    if ! ts status >/dev/null 2>&1; then
        echo "==> Authenticating this machine to your tailnet"
        echo "    A login URL will be printed below -- open it in any browser."
        echo
        # --accept-dns=false is belt-and-braces: userspace mode does not touch
        # resolv.conf anyway, but this makes the intent explicit and survives
        # anyone later switching this to kernel mode.
        ts up \
            --hostname="$HOSTNAME_TAG" \
            --accept-dns=false \
            --accept-routes=false \
            --ssh=false || {
                echo "error: 'tailscale up' failed; see above." >&2
                exit 1
            }
    fi

    echo "==> Publishing http://127.0.0.1:${PORT} to the tailnet over HTTPS"
    if ! ts serve --bg --https=443 "http://127.0.0.1:${PORT}" >/dev/null 2>&1; then
        echo "error: 'tailscale serve' failed." >&2
        echo "       Enable HTTPS certificates for your tailnet:" >&2
        echo "       https://login.tailscale.com/admin/dns -> HTTPS Certificates" >&2
        exit 1
    fi

    TS_HOST="$(ts status --json 2>/dev/null \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' \
        2>/dev/null || true)"

    echo
    echo "=============================================================="
    if [ -n "$TS_HOST" ]; then
        echo "  Lab members on your tailnet can now open:"
        echo "      https://${TS_HOST}"
    else
        echo "  Published. Run './publish_tailscale.sh status' for the URL."
    fi
    echo
    echo "  Your SSH access is unaffected: no TUN device was created, the"
    echo "  routing table was not modified, and /etc/resolv.conf was not"
    echo "  touched. Verify with:  ip route  and  cat /etc/resolv.conf"
    echo "=============================================================="
    ;;

status)
    need_binaries
    if ! daemon_running; then
        echo "userspace tailscaled is not running"
        exit 1
    fi
    echo "==> tailscaled pid $(cat "$PID_FILE")"
    ts status || true
    echo
    ts serve status || true
    ;;

stop)
    need_binaries
    if daemon_running; then
        echo "==> Withdrawing the published service"
        ts serve --https=443 off >/dev/null 2>&1 || true
        echo "==> Disconnecting from the tailnet"
        ts down >/dev/null 2>&1 || true
        echo "==> Stopping userspace tailscaled"
        kill -TERM "$(cat "$PID_FILE")" 2>/dev/null || true
        sleep 1
        kill -KILL "$(cat "$PID_FILE")" 2>/dev/null || true
        rm -f "$PID_FILE"
    fi
    echo "==> Stopped. Nothing on this host was modified."
    ;;

*)
    echo "usage: $0 {start|status|stop}" >&2
    exit 64
    ;;
esac
