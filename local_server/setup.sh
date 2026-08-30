#!/usr/bin/env bash
#
# One-shot setup: takes a fresh clone to a passing preflight.
#
#   ./setup.sh              # do everything, prompting before long downloads
#   ./setup.sh --yes        # never prompt (for unattended installs)
#   ./setup.sh --env cu11   # force a Pixi environment instead of auto-detecting
#
# Every step is idempotent and skips work that is already done, so it is safe
# to re-run after a failure or an interruption. Expect 15-40 minutes and
# several GB on a first run, almost all of it downloading model weights.
#
# What it does:
#   1. populates the CryoZeta submodule if it is missing
#   2. installs pixi if it is missing
#   3. installs the matching CUDA environment
#   4. downloads model weights and the bundled example
#   5. builds TEASER++
#   6. creates the web server's virtualenv
#   7. re-runs the preflight and reports
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

ASSUME_YES=0
FORCED_ENV=""
VENV_DIR="${CRYOZETA_WEB_VENV:-$SCRIPT_DIR/.venv}"

while [ $# -gt 0 ]; do
    case "$1" in
        --yes|-y)  ASSUME_YES=1; shift ;;
        --env)     FORCED_ENV="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown option '$1'" >&2; exit 64 ;;
    esac
done

STEP=0
step() { STEP=$((STEP + 1)); printf '\n\033[1m[%d/7] %s\033[0m\n' "$STEP" "$1"; }
ok()   { printf '      \033[32mok\033[0m %s\n' "$1"; }
warn() { printf '      \033[33m!\033[0m  %s\n' "$1"; }
die()  { printf '\n\033[31merror:\033[0m %s\n' "$1" >&2; exit 1; }

# Prompts default to NO when there is no terminal. These gates guard actions
# that download and execute remote code, so a non-interactive run must never
# silently consent -- use --yes to opt in explicitly.
confirm() {
    [ "$ASSUME_YES" -eq 1 ] && return 0
    if [ ! -r /dev/tty ]; then
        printf '      %s\n' "not running interactively; re-run with --yes to accept"
        return 1
    fi
    printf '      %s [Y/n] ' "$1"
    if ! { read -r reply </dev/tty; } 2>/dev/null; then
        printf '\n      %s\n' "could not read a reply; assuming no"
        return 1
    fi
    case "$reply" in [nN]*) return 1 ;; *) return 0 ;; esac
}

echo "CryoZeta local server -- setup"
echo "=============================================================="
echo "Repository: $REPO_ROOT"

# ── 1. CryoZeta submodule ────────────────────────────────────────────────────
step "CryoZeta source"
CRYOZETA_DIR="${CRYOZETA_WEB_REPO:-$REPO_ROOT/external/CryoZeta}"

if [ ! -f "$CRYOZETA_DIR/inference_demo.sh" ]; then
    if [ -d "$REPO_ROOT/.git" ] && [ -f "$REPO_ROOT/.gitmodules" ]; then
        warn "submodule is empty; fetching it"
        git -C "$REPO_ROOT" submodule update --init --recursive \
            || die "could not fetch the CryoZeta submodule. Check network access to github.com."
    else
        # Not a git checkout (e.g. downloaded as a zip): clone upstream directly.
        warn "no submodule metadata; cloning CryoZeta from GitHub"
        mkdir -p "$(dirname "$CRYOZETA_DIR")"
        git clone https://github.com/kiharalab/CryoZeta.git "$CRYOZETA_DIR" \
            || die "could not clone CryoZeta into $CRYOZETA_DIR"
    fi
fi

[ -f "$CRYOZETA_DIR/inference_demo.sh" ] || die "CryoZeta is still missing at $CRYOZETA_DIR"
ok "$CRYOZETA_DIR"
export CRYOZETA_WEB_REPO="$CRYOZETA_DIR"

# ── 2. pixi ──────────────────────────────────────────────────────────────────
step "pixi"
if command -v pixi >/dev/null 2>&1; then
    ok "$(command -v pixi)"
elif [ -x "$HOME/.pixi/bin/pixi" ]; then
    export PATH="$HOME/.pixi/bin:$PATH"
    ok "$HOME/.pixi/bin/pixi"
else
    warn "pixi is not installed"
    echo "      This runs the official installer from https://pixi.sh:"
    echo "          curl -fsSL https://pixi.sh/install.sh | bash"
    if confirm "Install pixi now?"; then
        curl -fsSL https://pixi.sh/install.sh | bash || die "pixi installation failed"
        export PATH="$HOME/.pixi/bin:$PATH"
        command -v pixi >/dev/null 2>&1 || die "pixi installed but not on PATH; open a new shell and re-run"
        ok "installed"
    else
        die "pixi is required. Install it, then re-run this script."
    fi
fi

# ── 3. CUDA environment ──────────────────────────────────────────────────────
step "CUDA environment"
if [ -n "$FORCED_ENV" ]; then
    PIXI_ENVIRONMENT="$FORCED_ENV"
    ok "forced: $PIXI_ENVIRONMENT"
else
    # Reuse the app's own detection so this matches what the server will run.
    PIXI_ENVIRONMENT="$(python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from app.discovery import query_nvidia, select_pixi_env
print(select_pixi_env(query_nvidia()))
" 2>/dev/null || echo default)"
    ok "auto-detected: $PIXI_ENVIRONMENT"
fi

if [ -d "$CRYOZETA_DIR/.pixi/envs/$PIXI_ENVIRONMENT" ]; then
    ok "already installed"
else
    echo "      Installing dependencies (CUDA toolkit, PyTorch, ~10 GB). This is slow."
    pixi install --manifest-path "$CRYOZETA_DIR" -e "$PIXI_ENVIRONMENT" \
        || die "pixi install failed for environment '$PIXI_ENVIRONMENT'.
       If your driver is older than the toolkit, try: ./setup.sh --env cu11"
    ok "installed"
fi

# ── 4. Model weights ─────────────────────────────────────────────────────────
step "Model weights and bundled example"
ASSETS_DIR="$CRYOZETA_DIR/assets"
NEEDED=(cryozeta-detection-v0.0.1.safetensors cryozeta-v0.0.1.safetensors
        cryozeta-interpolate-v0.0.1.safetensors)
MISSING=0
for f in "${NEEDED[@]}"; do
    [ -f "$ASSETS_DIR/$f" ] || MISSING=1
done

if [ "$MISSING" -eq 0 ]; then
    ok "already downloaded"
else
    echo "      Downloading from Hugging Face (several GB)."
    echo "      The weights are for academic / non-commercial research use only."
    if confirm "Download now?"; then
        pixi run --manifest-path "$CRYOZETA_DIR" -e "$PIXI_ENVIRONMENT" download-assets \
            || die "asset download failed. Check network access to huggingface.co, then re-run."
        ok "downloaded"
    else
        warn "skipped -- jobs cannot run without weights"
    fi
fi

# ── 5. TEASER++ ──────────────────────────────────────────────────────────────
step "TEASER++"
if [ -f "$CRYOZETA_DIR/externals/TEASER-plusplus/build/libteaser.so" ]; then
    ok "already built"
else
    # The build shells out to cmake and a C++ compiler; neither is provided by
    # the pixi environment, so check before spending time on a doomed build.
    for tool in cmake git; do
        command -v "$tool" >/dev/null 2>&1 \
            || die "'$tool' is required to build TEASER++ but was not found.
       On Debian/Ubuntu: sudo apt install cmake build-essential git"
    done
    echo "      Cloning and compiling TEASER++ (a few minutes)."
    pixi run --manifest-path "$CRYOZETA_DIR" -e "$PIXI_ENVIRONMENT" build-teaser \
        || die "TEASER++ build failed. See the output above.
       Most often this is a missing compiler or cmake older than 3.10."
    ok "built"
fi

# ── 6. Web server virtualenv ─────────────────────────────────────────────────
step "Web server dependencies"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR" || die "could not create virtualenv at $VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt" \
    || die "could not install web dependencies"
ok "$VENV_DIR"

# ── 7. Verify ────────────────────────────────────────────────────────────────
step "Verifying"
echo
set +e
"$SCRIPT_DIR/preflight.sh"
PREFLIGHT_STATUS=$?
set -e

echo
if [ "$PREFLIGHT_STATUS" -eq 0 ]; then
    cat <<EOF
==============================================================
Setup complete. Start the server with:

    ./start_local_server.sh

then open http://127.0.0.1:8000

To let lab members reach it over your Tailscale network:

    ./start_local_server.sh --tailscale
==============================================================
EOF
else
    cat <<EOF
==============================================================
Setup ran, but the preflight still reports problems above.
Fix the FAIL items and re-run this script -- completed steps
are skipped, so it will pick up where it left off.
==============================================================
EOF
fi
exit "$PREFLIGHT_STATUS"
