#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUNDLE_DIR="${PULSERVER_TEST_INSTALL_DIR:-$ROOT_DIR/.nim-test-bundle}"
INSTALLER_PATH="${PULSERVER_TEST_INSTALLER:-$ROOT_DIR/.nim-test-bundle-installer.sh}"

NIM_MAJOR="$(nim --version | awk '/Version/{print $4}' | cut -d. -f1)"
if [[ "$NIM_MAJOR" -lt 2 ]]; then
  echo "Nim >= 2.0 is required to run bridge tests (found $(nim --version | awk '/Version/{print $4}'))."
  exit 1
fi

echo "Running bridge Nim tests"

NEED_BUNDLE_BUILD=0
if [[ ! -x "$BUNDLE_DIR/bin/pypulseq_host" || ! -f "$BUNDLE_DIR/python/pyvenv.cfg" ]]; then
  NEED_BUNDLE_BUILD=1
fi

if [[ "$NEED_BUNDLE_BUILD" -eq 1 ]]; then
  echo "Building pulserver bundle for tests"
  rm -rf "$BUNDLE_DIR"
  rm -f "$INSTALLER_PATH"
  # For local/CI validation, prefer installing pulserver from this checkout.
  PULSERVER_PIP_SPEC="${PULSERVER_TEST_PIP_SPEC:-$ROOT_DIR}" \
    bash "$ROOT_DIR/scripts/make_installer.sh" "$INSTALLER_PATH"
  bash "$INSTALLER_PATH" "$BUNDLE_DIR"
fi

PYBIN="$BUNDLE_DIR/python/bin/python3"
if [[ ! -x "$PYBIN" ]]; then
  echo "Error: bundled Python not found at $PYBIN"
  exit 1
fi

echo "Using bundled Python venv: $($PYBIN -c 'import sys; print(sys.executable)')"

# Verify bundled environment content matches production expectations.
"$PYBIN" -c "import numpy, scipy, pypulseq, pulserver"

# Create EnvPulserver as a copy of the bundle venv to test the canonical-path
# code path in resolveBundledPythonHome (candidate 1 / pythonVenvPath define).
ENV_PULSERVER="$BUNDLE_DIR/EnvPulserver"
if [[ ! -d "$ENV_PULSERVER" ]]; then
  echo "Creating EnvPulserver test venv at $ENV_PULSERVER"
  cp -a "$BUNDLE_DIR/python" "$ENV_PULSERVER"
  "$PYBIN" -m venv --upgrade "$ENV_PULSERVER"
fi

# Explicitly strip any nimpyTestLibPython define (including inherited NIMFLAGS),
# because nimpy's test-only libpython hook has shown CI instability/OOM here.
# Keep only the bridge-specific canonical venv define.
NIMFLAGS_CLEAN="$(echo "${NIMFLAGS:-}" | sed -E 's@(^| )-d:nimpyTestLibPython=[^ ]+@@g' | xargs)"
if [[ -n "$NIMFLAGS_CLEAN" ]]; then
  export NIMFLAGS="$NIMFLAGS_CLEAN -d:pythonVenvPath=$ENV_PULSERVER"
else
  export NIMFLAGS="-d:pythonVenvPath=$ENV_PULSERVER"
fi

# Activate the venv for the test sub-shell: update PATH and PYTHONPATH
# to site-packages.  Do NOT set PYTHONHOME (breaks venv activation).
export VIRTUAL_ENV="$BUNDLE_DIR/python"
export PATH="$BUNDLE_DIR/python/bin:$PATH"
export PYTHONEXECUTABLE="$PYBIN"

PY_PATHS="$($PYBIN -c 'import site; print(":".join(site.getsitepackages()))' 2>/dev/null || echo "")"
if [[ -n "$PY_PATHS" ]]; then
  export PYTHONPATH="$PY_PATHS"
fi
echo "Using VIRTUAL_ENV: $VIRTUAL_ENV"

(
  cd "$ROOT_DIR/bridge"
  rm -rf "$HOME/.cache/nim/test_bridge_common_d"
  echo "Installing nimpulseqgui from GitHub"
  nimble install -y https://github.com/nimpulseq/nimpulseqgui
  echo "Running tests with pythonVenvPath=$ENV_PULSERVER"
  nim r $NIMFLAGS "$ROOT_DIR/tests/nim/test_bridge_common.nim"

  # Integration check: compile host with canonical EnvPulserver path and verify
  # it can import and execute a user plugin backed by a PulseqSequence class.
  TEST_HOST_BIN="$BUNDLE_DIR/bin/pypulseq_host_test"
  TEST_PLUGIN="$ROOT_DIR/bridge/tests/test_pulserver_sequence_plugin.py"
  TEST_SEQ_OUT="$BUNDLE_DIR/pulserver_sequence_plugin.seq"

  nim c -d:release -d:pythonVenvPath="$ENV_PULSERVER" -o:"$TEST_HOST_BIN" "$ROOT_DIR/bridge/pypulseq_host.nim"

  VALIDATION_OUT="$("$TEST_HOST_BIN" --script "$TEST_PLUGIN" --validate-only)"
  echo "$VALIDATION_OUT"
  echo "$VALIDATION_OUT" | grep '"valid": true'

  {
    echo "GENERATE $TEST_SEQ_OUT"
    echo "[NimPulseqGUI Protocol]"
    echo "[NimPulseqGUI Protocol End]"
    echo "QUIT"
  } | "$TEST_HOST_BIN" --script "$TEST_PLUGIN" --persistent > "$BUNDLE_DIR/persistent_host_test.out"

  grep "GENERATED $TEST_SEQ_OUT" "$BUNDLE_DIR/persistent_host_test.out"
  test -s "$TEST_SEQ_OUT"
)
