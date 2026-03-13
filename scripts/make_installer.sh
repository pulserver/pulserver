#!/usr/bin/env bash
# make_installer.sh — build a self-extracting installer for pypulseq_host.
#
# Packages the compiled bridge binary, a standalone relocatable CPython
# (from python-build-standalone), and example plugins into a single .sh
# archive that can be deployed to any Linux host without a development
# environment. The bundled Python environment includes:
#   - numpy
#   - scipy
#   - pypulseq
#   - pulserver
#
# Usage:
#   ./scripts/make_installer.sh [output.sh]
#
# Prerequisites:
#   - nim & nimble on PATH
#   - bridge/ has been built (nimble build in bridge/)
#   - curl or wget for downloading python-build-standalone
#
# The resulting installer, when run, extracts to $INSTALL_DIR (default ~/pulserver)
# and creates a ready-to-run pypulseq_host with embedded Python.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BRIDGE_DIR="$ROOT_DIR/bridge"
BUILD_DIR="$(mktemp -d)"
OUTPUT="${1:-$ROOT_DIR/pulserver_bridge_installer.sh}"

# python-build-standalone release configuration
PYTHON_VERSION="3.11"
PBS_TRIPLE="x86_64-unknown-linux-gnu"
PBS_VARIANT="install_only"
PULSERVER_PIP_SPEC="${PULSERVER_PIP_SPEC:-git+ssh://git@github.com/pulserver/pulserver.git@dev}"

echo "=== Resolving standalone Python ${PYTHON_VERSION} asset ==="
PBS_URL="$(
  PBS_PY_MINOR="$PYTHON_VERSION" PBS_TRIPLE="$PBS_TRIPLE" PBS_VARIANT="$PBS_VARIANT" python3 - <<'PY'
import json
import os
import sys
import urllib.request

minor = os.environ["PBS_PY_MINOR"]
triple = os.environ["PBS_TRIPLE"]
variant = os.environ["PBS_VARIANT"]
prefix = f"cpython-{minor}."
suffix = f"-{triple}-{variant}.tar.gz"

api_url = "https://api.github.com/repos/astral-sh/python-build-standalone/releases?per_page=10"
with urllib.request.urlopen(api_url) as resp:
  releases = json.load(resp)

for rel in releases:
  for asset in rel.get("assets", []):
    name = asset.get("name", "")
    if name.startswith(prefix) and name.endswith(suffix):
      print(asset["browser_download_url"])
      raise SystemExit(0)

raise SystemExit("No matching python-build-standalone asset found")
PY
)"
echo "Using standalone asset: $PBS_URL"

cleanup() { rm -rf "$BUILD_DIR"; }
trap cleanup EXIT

echo "=== Building bridge binaries ==="
cd "$BRIDGE_DIR"
nimble build -y

echo "=== Downloading standalone Python ${PYTHON_VERSION} ==="
PBS_ARCHIVE="$BUILD_DIR/cpython.tar.gz"
if command -v curl &>/dev/null; then
  curl -fSL -o "$PBS_ARCHIVE" "$PBS_URL"
elif command -v wget &>/dev/null; then
  wget -q -O "$PBS_ARCHIVE" "$PBS_URL"
else
  echo "Error: need curl or wget to download python-build-standalone" >&2
  exit 1
fi

echo "=== Extracting standalone Python ==="
PYTHON_DIR="$BUILD_DIR/python"
mkdir -p "$PYTHON_DIR"
tar xf "$PBS_ARCHIVE" -C "$PYTHON_DIR" --strip-components=1

echo "=== Installing Python packages ==="
SITE_PACKAGES="$($PYTHON_DIR/bin/python3 - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

"$PYTHON_DIR/bin/python3" -m pip install --quiet --target "$SITE_PACKAGES" numpy scipy pypulseq

if ! env -u CC -u CXX "$PYTHON_DIR/bin/python3" -m pip install --quiet --target "$SITE_PACKAGES" --no-deps "$PULSERVER_PIP_SPEC"; then
  echo "Warning: pip install for pulserver failed; falling back to source package copy"
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
