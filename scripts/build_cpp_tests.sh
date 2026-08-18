#!/bin/bash

set -e

BUILD_DIR="tests/cpptests/build"
SOURCE_DIR="tests/cpptests"

if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning the previous C++ test build directory..."
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

echo "Configuring C++ tests with CMake..."
cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug

echo "Building C++ tests..."
cmake --build "$BUILD_DIR"
