#!/bin/bash

# Exit on errors
set -e

# Define paths
SCRIPT_DIR="$(dirname "$0")"

$SCRIPT_DIR/build_ctests.sh
$SCRIPT_DIR/run_ctests.sh