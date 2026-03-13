#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

NIM_MAJOR="$(nim --version | awk '/Version/{print $4}' | cut -d. -f1)"
if [[ "$NIM_MAJOR" -lt 2 ]]; then
  echo "Nim >= 2.0 is required to run bridge tests (found $(nim --version | awk '/Version/{print $4}'))."
  exit 1
fi

echo "Running bridge Nim tests"
(
  cd "$ROOT_DIR/bridge"
  echo "Installing nimpulseqgui from GitHub"
  nimble install -y https://github.com/nimpulseq/nimpulseqgui
  nimble test
)
