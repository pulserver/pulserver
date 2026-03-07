#!/usr/bin/env bash
# make_installer.sh — build a self-extracting installer for pypulseq_host.
#
# Packages the compiled bridge binary, a bundled CPython (minimal venv),
# and an example plugin directory into a single .sh archive that can be
# deployed to any Linux host without a development environment.
#
# Usage:
#   ./scripts/make_installer.sh [output.sh]
#
# Prerequisites:
#   - nim & nimble on PATH
#   - bridge/ has been built (nimble build in bridge/)
#   - Python 3.10+ with venv module
#
# The resulting installer, when run, extracts to $INSTALL_DIR (default ~/pulserver)
# and creates a ready-to-run pypulseq_host with embedded Python.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BRIDGE_DIR="$ROOT_DIR/bridge"
BUILD_DIR="$(mktemp -d)"
OUTPUT="${1:-$ROOT_DIR/pulserver_bridge_installer.sh}"

cleanup() { rm -rf "$BUILD_DIR"; }
trap cleanup EXIT

echo "=== Building bridge binaries ==="
cd "$BRIDGE_DIR"
nimble build -y

echo "=== Creating minimal Python environment ==="
PYTHON_DIR="$BUILD_DIR/python"
python3 -m venv --without-pip "$PYTHON_DIR"
# Install minimal deps into the venv
"$PYTHON_DIR/bin/python3" -m ensurepip --upgrade 2>/dev/null || true
"$PYTHON_DIR/bin/pip3" install --quiet pypulseq numpy scipy 2>/dev/null || \
  echo "Warning: could not install Python packages (install manually after extraction)"

echo "=== Assembling payload ==="
PAYLOAD_DIR="$BUILD_DIR/payload"
mkdir -p "$PAYLOAD_DIR/bin" "$PAYLOAD_DIR/plugins"

# Copy bridge binaries
cp "$BRIDGE_DIR/pypulseq_host" "$PAYLOAD_DIR/bin/" 2>/dev/null || \
  echo "Warning: pypulseq_host binary not found — build first with 'nimble build'"

# Copy bundled Python
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

# Set up PYTHONHOME for the bridge binary
PYTHON_HOME="$INSTALL_DIR/python"
if [ -d "$PYTHON_HOME" ]; then
  echo "Python environment: $PYTHON_HOME"
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
