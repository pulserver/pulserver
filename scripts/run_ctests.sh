#!/bin/bash

# Exit on errors
set -e

# Define paths
BUILD_DIR="tests/ctests/build"

# Run the compiled tests
echo "Running the tests..."
$BUILD_DIR/bin/run_tests
