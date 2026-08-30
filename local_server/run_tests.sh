#!/usr/bin/env bash
#
# Run the CryoZeta local-server test suite.
#
#   ./run_tests.sh              # unit + fake-runner integration tests (no GPU)
#   ./run_tests.sh --smoke      # additionally run the real bundled example
#   ./run_tests.sh -k pattern   # forward arguments to pytest
#
# The default run needs no GPU, no CryoZeta assets and no pixi: the inference
# binary is replaced by tests/fake/fake_inference_demo.sh, which validates the
# generated JSON and writes an output tree of the same shape.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${CRYOZETA_WEB_VENV:-$SCRIPT_DIR/.venv}"
RUN_SMOKE=0
PYTEST_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --smoke) RUN_SMOKE=1; shift ;;
        -h|--help)
            sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) PYTEST_ARGS+=("$1"); shift ;;
    esac
done

if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating test virtualenv: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet -r requirements.txt
fi

PYTHON="$VENV_DIR/bin/python"
chmod +x tests/fake/fake_inference_demo.sh

echo "==> Unit and integration tests (fake runner, no GPU)"
"$PYTHON" -m pytest tests/ -q "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}"

if [ "$RUN_SMOKE" -eq 1 ]; then
    echo
    echo "==> Real smoke test against the bundled CryoZeta example"
    echo "    (needs a GPU, pixi, and downloaded assets -- takes ~30 minutes)"
    "$PYTHON" -m pytest tests/smoke -q -m smoke -s \
        "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}"
fi
