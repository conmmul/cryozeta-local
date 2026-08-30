#!/usr/bin/env bash
#
# Start the CryoZeta local web server.
#
# Portable by design: the CryoZeta checkout, the pixi binary and the GPU
# inventory are all discovered at run time, so this script works unchanged on
# any workstation.
#
# Usage:
#   ./start_local_server.sh                 # 127.0.0.1:8000
#   ./start_local_server.sh --port 8080
#   ./start_local_server.sh --foreground    # do not daemonise
#   ./start_local_server.sh --skip-install  # do not touch the virtualenv
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST="${CRYOZETA_WEB_HOST:-127.0.0.1}"
PORT="${CRYOZETA_WEB_PORT:-8000}"
DATA_ROOT="${CRYOZETA_WEB_DATA_ROOT:-$HOME/cryozeta-web-data}"
VENV_DIR="${CRYOZETA_WEB_VENV:-$SCRIPT_DIR/.venv}"
FOREGROUND=0
SKIP_INSTALL=0

while [ $# -gt 0 ]; do
    case "$1" in
        --host)          HOST="$2"; shift 2 ;;
        --port)          PORT="$2"; shift 2 ;;
        --data-root)     DATA_ROOT="$2"; shift 2 ;;
        --foreground|-f) FOREGROUND=1; shift ;;
        --skip-install)  SKIP_INSTALL=1; shift ;;
        -h|--help)
            sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "error: unknown option '$1'" >&2; exit 64 ;;
    esac
done

# ── Refuse to expose the app on the network by accident ──────────────────────
# There is no authentication of any kind. Binding anything other than loopback
# requires a deliberate, separate opt-in.
case "$HOST" in
    127.0.0.1|localhost|::1) ;;
    *)
        if [ "${CRYOZETA_WEB_ALLOW_LAN:-0}" != "1" ]; then
            cat >&2 <<EOF
error: refusing to bind ${HOST}.

  This server has no login, no accounts and no authorisation checks. Anyone who
  can reach this address can upload maps, read every result, and run code on
  this machine's GPUs.

  To expose it anyway (only on a trusted, firewalled network):
      export CRYOZETA_WEB_ALLOW_LAN=1

  The recommended alternative is an SSH tunnel from your laptop:
      ssh -N -L ${PORT}:127.0.0.1:${PORT} $(whoami)@$(hostname)
  then open http://127.0.0.1:${PORT} locally.
EOF
            exit 78
        fi
        echo "WARNING: binding ${HOST} -- this server has no authentication." >&2
        ;;
esac

# ── Locate the CryoZeta repository (informational; the app re-checks) ────────
find_cryozeta() {
    local candidates=(
        "${CRYOZETA_WEB_REPO:-}"
        "${CRYOZETA_REPO:-}"
        "$SCRIPT_DIR/../external/CryoZeta"
        "$SCRIPT_DIR/../CryoZeta"
        "$HOME/CryoZeta"
    )
    for dir in "${candidates[@]}"; do
        [ -n "$dir" ] || continue
        if [ -f "$dir/inference_demo.sh" ] && [ -f "$dir/pyproject.toml" ]; then
            (CDPATH= cd -- "$dir" && pwd)
            return 0
        fi
    done
    return 1
}

if REPO="$(find_cryozeta)"; then
    echo "==> CryoZeta repository: $REPO"
    export CRYOZETA_WEB_REPO="$REPO"
else
    echo "WARNING: no CryoZeta checkout found." >&2
    echo "         Set CRYOZETA_WEB_REPO=/path/to/CryoZeta before submitting jobs." >&2
fi

if command -v pixi >/dev/null 2>&1; then
    echo "==> pixi: $(command -v pixi)"
elif [ -x "$HOME/.pixi/bin/pixi" ]; then
    echo "==> pixi: $HOME/.pixi/bin/pixi"
    export PATH="$HOME/.pixi/bin:$PATH"
else
    echo "WARNING: pixi not found. Install it with:" >&2
    echo "         curl -fsSL https://pixi.sh/install.sh | bash" >&2
fi

# ── Python virtualenv for the web app (separate from CryoZeta's pixi env) ────
if [ "$SKIP_INSTALL" -eq 0 ]; then
    if [ ! -d "$VENV_DIR" ]; then
        echo "==> Creating virtualenv: $VENV_DIR"
        python3 -m venv "$VENV_DIR"
    fi
    echo "==> Installing web dependencies"
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"
fi

PYTHON="$VENV_DIR/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"

# ── Data directories and database ────────────────────────────────────────────
export CRYOZETA_WEB_HOST="$HOST"
export CRYOZETA_WEB_PORT="$PORT"
export CRYOZETA_WEB_DATA_ROOT="$DATA_ROOT"

mkdir -p "$DATA_ROOT"/{jobs,msa_library,run}
echo "==> Data root: $DATA_ROOT"

# Initialise / migrate the SQLite schema before serving, and recover any jobs
# that were left RUNNING by a previous process.
# cwd is $SCRIPT_DIR, so the `app` package imports directly.
"$PYTHON" - <<'PY'
import sys

sys.path.insert(0, ".")

from app.config import get_settings
from app.db import JobStore

settings = get_settings()
settings.ensure_dirs()
store = JobStore(settings.db_path)
orphans = store.mark_orphans_interrupted()
if orphans:
    print(f"==> Marked {len(orphans)} abandoned job(s) as interrupted")
print(f"==> Database ready: {settings.db_path}")
store.close()
PY

PID_FILE="$DATA_ROOT/run/server.pid"
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "error: a server is already running (pid $(cat "$PID_FILE"))." >&2
    echo "       Stop it first: ./stop_local_server.sh" >&2
    exit 1
fi

URL="http://${HOST}:${PORT}"
echo
echo "  ┌────────────────────────────────────────────────┐"
printf "  │  CryoZeta local server                         │\n"
printf "  │  %-44s  │\n" "$URL"
echo "  └────────────────────────────────────────────────┘"
echo

if [ "$FOREGROUND" -eq 1 ]; then
    exec "$PYTHON" -m uvicorn app.main:create_app --factory \
        --host "$HOST" --port "$PORT"
else
    LOG_FILE="$DATA_ROOT/run/server.log"
    nohup "$PYTHON" -m uvicorn app.main:create_app --factory \
        --host "$HOST" --port "$PORT" >>"$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 2
    if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "==> Running in the background (pid $(cat "$PID_FILE"))"
        echo "==> Server log: $LOG_FILE"
        echo "==> Stop with: ./stop_local_server.sh"
    else
        echo "error: the server exited immediately. Last lines of $LOG_FILE:" >&2
        tail -20 "$LOG_FILE" >&2
        rm -f "$PID_FILE"
        exit 1
    fi
fi
