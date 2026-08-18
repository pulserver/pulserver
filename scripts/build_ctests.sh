#!/bin/bash

# Exit on errors
set -e

# Define paths
BUILD_DIR="tests/ctests/build"
SOURCE_DIR="tests/ctests"

# Clean the build directory if it exists, then recreate it
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning the previous build directory..."
    rm -rf $BUILD_DIR
fi
mkdir -p $BUILD_DIR

# Run cmake to configure the project, pointing to the source directory
echo "Configuring the project with CMake..."
cmake -S $SOURCE_DIR -B $BUILD_DIR -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER= -DCMAKE_CXX_COMPILER_WORKS=FALSE

# Build the project using cmake
echo "Building the project with CMake..."
cmake --build $BUILD_DIR --target run_tests