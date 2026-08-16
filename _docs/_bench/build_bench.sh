#!/usr/bin/env bash
# build_bench.sh -- build docs/_bench/bench_pipeline against the C library.
#
# Documentation-only tooling; the binary lands in docs/_bench/ and is not
# installed. Reuses the C test suite's CMake build for libpulseg/libpulseq so
# this stays a one-file compile.
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$BENCH_DIR/../.." && pwd)"
# Dedicated build dir. This used to share tests/ctests/build, but
# scripts/build_ctests.sh configures that dir as Debug -- so any bench run
# after a test run silently measured Debug libraries (parse and safety both
# read ~1.5-2x slower, peak RSS ~30% higher) while still reporting itself as
# the timing binary. Owning the dir keeps the build type under our control.
BUILD_DIR="$REPO_ROOT/tests/ctests/build-bench"
LIB_DIR="$BUILD_DIR/csrc_build"

# Configure every time: cheap when the cache is already Release, and it is the
# only thing standing between a stale build type and a wrong benchmark.
cmake -S "$REPO_ROOT/tests/ctests" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BUILD_DIR" --target pulseg pulseq -j"$(nproc)" >/dev/null

# --wrap routes every allocation the static libraries make through
# bench_alloc.c, so the reported heap is what the process really holds rather
# than what one of several allocation paths happened to request.
cc -O2 -std=gnu99 -D_POSIX_C_SOURCE=200112L \
    -I"$REPO_ROOT/csrc/include" \
    -I"$REPO_ROOT/csrc/include/pulseg" \
    -I"$REPO_ROOT/csrc/include/pulseq" \
    -I"$REPO_ROOT/csrc/src" \
    -I"$BENCH_DIR" \
    -o "$BENCH_DIR/bench_pipeline" \
    "$BENCH_DIR/bench_pipeline.c" "$BENCH_DIR/bench_alloc.c" \
    "$LIB_DIR/libpulseg.a" "$LIB_DIR/libpulseq.a" -lm \
    -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc -Wl,--wrap=free

echo "Built $BENCH_DIR/bench_pipeline"

# Runtime-only companion: no malloc interposition, so this is the binary to
# use when deciding whether a C stage needs optimization. Heap fields are zero
# and heap_accounting is false in its JSON output.
cc -O2 -std=gnu99 -D_POSIX_C_SOURCE=200112L -DBENCH_ALLOC_DISABLED \
    -I"$REPO_ROOT/csrc/include" \
    -I"$REPO_ROOT/csrc/include/pulseg" \
    -I"$REPO_ROOT/csrc/include/pulseq" \
    -I"$REPO_ROOT/csrc/src" \
    -I"$BENCH_DIR" \
    -o "$BENCH_DIR/bench_pipeline_timing" \
    "$BENCH_DIR/bench_pipeline.c" "$BENCH_DIR/bench_alloc.c" \
    "$LIB_DIR/libpulseg.a" "$LIB_DIR/libpulseq.a" -lm

echo "Built $BENCH_DIR/bench_pipeline_timing"
