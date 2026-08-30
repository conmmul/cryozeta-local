#!/usr/bin/env bash
#
# Stop the CryoZeta local web server.
#
# Running CryoZeta jobs are NOT killed by default: they are long and expensive,
# and the server marks them as "interrupted" on restart rather than pretending
# they completed. Pass --kill-jobs to terminate them too.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${CRYOZETA_WEB_DATA_ROOT:-$HOME/cryozeta-web-data}"
KILL_JOBS=0
GRACE=15

while [ $# -gt 0 ]; do
    case "$1" in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --kill-jobs) KILL_JOBS=1; shift ;;
        -h|--help)
            sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "error: unknown option '$1'" >&2; exit 64 ;;
    esac
done

PID_FILE="$DATA_ROOT/run/server.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file at $PID_FILE -- the server does not appear to be running."
    exit 0
fi

PID="$(cat "$PID_FILE")"
if ! kill -0 "$PID" 2>/dev/null; then
    echo "Stale PID file (process $PID is gone). Removing it."
    rm -f "$PID_FILE"
    exit 0
fi

if [ "$KILL_JOBS" -eq 1 ]; then
    echo "==> Terminating running CryoZeta jobs"
    # Each job runs in its own session, so signal the children's groups too.
    for child in $(pgrep -P "$PID" 2>/dev/null || true); do
        pgid="$(ps -o pgid= -p "$child" 2>/dev/null | tr -d ' ' || true)"
        [ -n "$pgid" ] && kill -TERM "-$pgid" 2>/dev/null || true
    done
fi

# Stop publishing to the tailnet first, so nobody hits a dead proxy.
if command -v tailscale >/dev/null 2>&1; then
    if tailscale serve status 2>/dev/null | grep -q "127.0.0.1:"; then
        echo "==> Withdrawing tailscale serve"
        tailscale serve --https=443 off >/dev/null 2>&1 || true
    fi
fi

echo "==> Stopping server (pid $PID)"
kill -TERM "$PID" 2>/dev/null || true

for _ in $(seq 1 "$GRACE"); do
    if ! kill -0 "$PID" 2>/dev/null; then
        rm -f "$PID_FILE"
        echo "==> Stopped."
        exit 0
    fi
    sleep 1
done

echo "==> Did not exit within ${GRACE}s; sending SIGKILL"
kill -KILL "$PID" 2>/dev/null || true
rm -f "$PID_FILE"
echo "==> Stopped."
