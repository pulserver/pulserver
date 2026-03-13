#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUNDLE_DIR="${PULSERVER_TEST_INSTALL_DIR:-$ROOT_DIR/.nim-test-bundle}"
INSTALLER_PATH="${PULSERVER_TEST_INSTALLER:-$ROOT_DIR/.nim-test-bundle-installer.sh}"
EXPECTED_BUNDLE_PY_MINOR="${PULSERVER_TEST_PY_MINOR:-3.11}"

NIM_MAJOR="$(nim --version | awk '/Version/{print $4}' | cut -d. -f1)"
if [[ "$NIM_MAJOR" -lt 2 ]]; then
  echo "Nim >= 2.0 is required to run bridge tests (found $(nim --version | awk '/Version/{print $4}'))."
  exit 1
fi

echo "Running bridge Nim tests"

NEED_BUNDLE_BUILD=0
if [[ ! -x "$BUNDLE_DIR/bin/pypulseq_host" || ! -x "$BUNDLE_DIR/python/bin/python3" ]]; then
  NEED_BUNDLE_BUILD=1
else
  BUNDLE_PY_MINOR="$($BUNDLE_DIR/python/bin/python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo unknown)"
  if [[ "$BUNDLE_PY_MINOR" != "$EXPECTED_BUNDLE_PY_MINOR" ]]; then
    NEED_BUNDLE_BUILD=1
  fi
fi

if [[ "$NEED_BUNDLE_BUILD" -eq 1 ]]; then
  echo "Building standalone pulserver bundle for tests"
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

echo "Using bundled Python runtime: $($PYBIN -c 'import sys; print(sys.executable)')"

# Verify bundled environment content matches production expectations.
"$PYBIN" -c "import numpy, scipy, pypulseq, pulserver"

# Force nimpy to load the selected interpreter's libpython.
PY_LIB="$($PYBIN - <<'PY'
import glob
import os
import sys
import sysconfig

libdir = sysconfig.get_config_var("LIBDIR") or ""
ldlib = sysconfig.get_config_var("LDLIBRARY") or ""
candidates = []
if libdir and ldlib:
    candidates.append(os.path.join(libdir, ldlib))

maj, min_ = sys.version_info[:2]
patterns = [
    os.path.join(libdir, f"libpython{maj}.{min_}*.so*"),
    os.path.join(sys.base_prefix, "lib", f"libpython{maj}.{min_}*.so*"),
]
for pat in patterns:
    candidates.extend(sorted(glob.glob(pat)))

for c in candidates:
    if c and os.path.exists(c):
        print(c)
        raise SystemExit(0)

raise SystemExit(1)
PY
)"

if [[ -z "$PY_LIB" ]]; then
  echo "Error: could not locate libpython for $PYBIN"
  exit 1
fi

export NIMFLAGS="${NIMFLAGS:-} -d:nimpyTestLibPython=$PY_LIB"
echo "Using libpython: $PY_LIB"

export PATH="$BUNDLE_DIR/python/bin:$PATH"
export PYTHONEXECUTABLE="$PYBIN"

PY_HOME="$BUNDLE_DIR/python"
PY_PATHS="$($PYBIN - <<'PY'
import site
paths = []
try:
  paths.extend(site.getsitepackages())
except Exception:
  pass
print(":".join(paths))
PY
)"
export PYTHONHOME="$PY_HOME"
export PYTHONPLATLIBDIR="lib"
export PYTHONNOUSERSITE="1"
if [[ -n "$PY_PATHS" ]]; then
  export PYTHONPATH="$PY_PATHS"
fi
echo "Using PYTHONHOME: $PYTHONHOME"

(
  cd "$ROOT_DIR/bridge"
  rm -rf "$HOME/.cache/nim/test_bridge_common_d" "$HOME/.cache/nim/test_isPyNone_d"
  echo "Installing nimpulseqgui from GitHub"
  nimble install -y https://github.com/nimpulseq/nimpulseqgui
  nimble test
)
