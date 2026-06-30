#!/bin/bash

# Exit on errors
set -e

# Define paths
SCRIPT_DIR="$(dirname "$0")"

bash $SCRIPT_DIR/build_ctests.sh
bash $SCRIPT_DIR/run_ctests.sh