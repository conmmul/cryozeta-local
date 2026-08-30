#!/usr/bin/env bash
#
# Read-only preflight: is this machine able to run CryoZeta?
#
#   ./preflight.sh           human-readable report
#   ./preflight.sh --json    machine-readable
#
# Changes nothing. Exit status 0 = ready, 1 = not ready.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${CRYOZETA_WEB_VENV:-$SCRIPT_DIR/.venv}"
if [ -x "$VENV_DIR/bin/python" ]; then
    PYTHON="$VENV_DIR/bin/python"
else
    # Preflight deliberately has no third-party dependencies, so the system
    # interpreter is enough when the virtualenv does not exist yet.
    PYTHON="$(command -v python3)"
fi

exec "$PYTHON" -m app.cli preflight "$@"
