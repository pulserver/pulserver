#!/bin/bash
##
## Build and test the bridge (Nim host executables).
##
## Usage: ./scripts/build_and_test_bridge.sh [--no-test]
##
## Requires: nim, nimble

set -e

BUILD_DIR="bridge"
NO_TEST=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-test)
      NO_TEST=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check that Nim tools are available
if ! command -v nim &> /dev/null; then
  echo "ERROR: nim compiler not found in PATH"
  echo "Please install Nim from https://nim-lang.org/"
  exit 1
fi

if ! command -v nimble &> /dev/null; then
  echo "ERROR: nimble package manager not found in PATH"
  echo "Please install Nim from https://nim-lang.org/"
  exit 1
fi

echo "=== Building bridge ==="
echo "nim version: $(nim --version 2>&1 | head -1)"
echo "nimble version: $(nimble --version 2>&1 | head -1)"
echo ""

cd "$BUILD_DIR"

echo "Running: nimble build"
nimble build

if [ "$NO_TEST" = true ]; then
  echo ""
  echo "=== Skipping tests (--no-test flag) ==="
  exit 0
fi

echo ""
echo "=== Testing bridge ==="
echo "Running: nimble test"
nimble test

echo ""
echo "=== Bridge build and tests completed successfully ==="
