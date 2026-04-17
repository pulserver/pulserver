#!/usr/bin/env bash
# make_installer.sh — build a self-extracting installer for pypulseq_host.
#
# Packages the compiled bridge binary, a Python venv with all dependencies,
# and example plugins into a single .sh archive that can be deployed to any
# Linux host.  The bundled venv includes:
#   - numpy
#   - scipy
#   - pypulseq
#   - pulserver
#
# The system Python (python3) is used to create the venv, so it must be
# >= 3.14 on the build host.
#
# Usage:
#   ./scripts/make_installer.sh [output.sh]
#
# Environment variables:
#   PULSERVER_PIP_SPEC  — pip spec to install for pulserver (default: "pulserver"
#                         from PyPI; set to a local checkout path for testing)
#   PYTHON              — path to the python3 binary to use (default: python3)
#
# Prerequisites:
#   - nim & nimble on PATH
#   - python3 >= 3.11 on PATH (or PYTHON env var)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BRIDGE_DIR="$ROOT_DIR/bridge"
BUILD_DIR="$(mktemp -d)"
OUTPUT="${1:-$ROOT_DIR/pulserver_bridge_installer.sh}"

PYTHON="${PYTHON:-python3}"
PULSERVER_PIP_SPEC="${PULSERVER_PIP_SPEC:-pulserver}"

cleanup() { rm -rf "$BUILD_DIR"; }
trap cleanup EXIT

echo "=== Building bridge binary (pypulseq_host) ==="
cd "$BRIDGE_DIR"
echo "Installing nimpulseqgui from GitHub"
nimble install -y https://github.com/nimpulseq/nimpulseqgui
nimble c -y -d:release pypulseq_host.nim

echo "=== Creating Python venv ==="
PYTHON_DIR="$BUILD_DIR/python"
"$PYTHON" -m venv "$PYTHON_DIR"

echo "=== Installing Python packages ==="
"$PYTHON_DIR/bin/pip" install --quiet --upgrade pip
"$PYTHON_DIR/bin/pip" install --quiet numpy scipy pypulseq

if ! env -u CC -u CXX "$PYTHON_DIR/bin/pip" install --quiet "$PULSERVER_PIP_SPEC"; then
  echo "Warning: pip install for pulserver failed; falling back to source package copy"
  SITE_PACKAGES="$("$PYTHON_DIR/bin/python3" -c 'import site; print(site.getsitepackages()[0])')"
  if [[ -d "$ROOT_DIR/python/pulserver" ]]; then
    cp -a "$ROOT_DIR/python/pulserver" "$SITE_PACKAGES/"
  else
    echo "Error: fallback source package path not found: $ROOT_DIR/python/pulserver" >&2
    exit 1
  fi
fi

echo "=== Assembling payload ==="
PAYLOAD_DIR="$BUILD_DIR/payload"
mkdir -p "$PAYLOAD_DIR/bin" "$PAYLOAD_DIR/plugins"

# Copy bridge binaries
cp "$BRIDGE_DIR/pypulseq_host" "$PAYLOAD_DIR/bin/" 2>/dev/null || \
  echo "Warning: pypulseq_host binary not found — build first with 'nimble build'"

# Copy bundled Python (relocatable — no hardcoded paths)
cp -a "$PYTHON_DIR" "$PAYLOAD_DIR/python"

# Copy example plugin(s)
if [ -d "$ROOT_DIR/python/pulserver/sequences" ]; then
  cp "$ROOT_DIR/python/pulserver/sequences/"*.py "$PAYLOAD_DIR/plugins/" 2>/dev/null || true
fi

echo "=== Creating self-extracting archive ==="
ARCHIVE="$BUILD_DIR/payload.tar.gz"
tar czf "$ARCHIVE" -C "$BUILD_DIR" payload

cat > "$OUTPUT" << 'INSTALLER_HEADER'
#!/bin/bash
# Self-extracting installer for pulserver bridge
set -euo pipefail

INSTALL_DIR="${1:-$HOME/pulserver}"
echo "Installing pulserver bridge to $INSTALL_DIR ..."

ARCHIVE_LINE=$(awk '/^__ARCHIVE__$/ {print NR + 1; exit}' "$0")
mkdir -p "$INSTALL_DIR"
tail -n +"$ARCHIVE_LINE" "$0" | tar xz -C "$INSTALL_DIR" --strip-components=1

# The venv was built on the build host with absolute paths inside the venv's
# pyvenv.cfg.  Relocate it to the actual install directory by re-running
# venv creation over the extracted tree (--upgrade re-uses the existing
# site-packages, just rewrites the activation scripts and pyvenv.cfg).
PYTHON="$INSTALL_DIR/python"
if [ -d "$PYTHON" ]; then
  echo "Relocating Python venv to $PYTHON ..."
  python3 -m venv --upgrade "$PYTHON"
  echo "Python environment: $PYTHON"
fi

echo ""
echo "Installation complete."
echo ""
echo "Usage:"
echo "  $INSTALL_DIR/bin/pypulseq_host --script <plugin.py> [options]"
echo ""
echo "For persistent mode (GE driver):"
echo "  $INSTALL_DIR/bin/pypulseq_host --script <plugin.py> --persistent"
echo ""
exit 0
__ARCHIVE__
INSTALLER_HEADER

cat "$ARCHIVE" >> "$OUTPUT"
chmod +x "$OUTPUT"

echo ""
echo "=== Done ==="
echo "Installer: $OUTPUT"
echo "Run it with: bash $OUTPUT [install_dir]"
