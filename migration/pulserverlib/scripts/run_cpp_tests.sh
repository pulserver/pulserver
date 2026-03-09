#!/bin/bash

set -e

BUILD_DIR="tests/cpptests/build"

echo "Running C++ tests with CTest..."
ctest --test-dir "$BUILD_DIR" --output-on-failure
