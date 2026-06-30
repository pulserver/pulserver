cmake_minimum_required(VERSION 3.17)
project(mrdserver_cpp_tests C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------
# Locate ISMRMRD (system install at /usr/local).
# ------------------------------------------------------------------
find_package(ISMRMRD REQUIRED)

find_package(Boost REQUIRED)

# ------------------------------------------------------------------
# Fetch GoogleTest.
# ------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ------------------------------------------------------------------
# Build trajectory_cache_reader as an OBJECT library to reuse here
# without depending on the full GadgetronClient build.
# ------------------------------------------------------------------
set(MRDSERVER_CSRC ${CMAKE_CURRENT_SOURCE_DIR}/../../cxxsrc)

add_library(trajectory_cache_reader_obj OBJECT
    ${MRDSERVER_CSRC}/trajectory_cache_reader.cpp)
target_include_directories(trajectory_cache_reader_obj PUBLIC
    ${MRDSERVER_CSRC}
    ${ISMRMRD_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS})

# ------------------------------------------------------------------
# Test executable.
# ------------------------------------------------------------------
add_executable(test_trajectory_cache_loader
    test_trajectory_cache_loader.cpp
    $<TARGET_OBJECTS:trajectory_cache_reader_obj>)

target_include_directories(test_trajectory_cache_loader PRIVATE
    ${MRDSERVER_CSRC}
    ${ISMRMRD_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS})

# Path to the .pge fixtures, baked in at configure time.
target_compile_definitions(test_trajectory_cache_loader PRIVATE
    MRDSERVER_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../pulserverlib-tests/expected")

target_link_libraries(test_trajectory_cache_loader PRIVATE
    GTest::gtest_main
    ISMRMRD::ISMRMRD)

# ------------------------------------------------------------------
# Section 5 (SEQUENCEDESCRIPTION) byte-exact truth test.
# Validates that the C-side cache writer (pulseqlib_cache_seqdesc.c)
# and the MATLAB TruthBuilder.exportSequenceDescription emit the
# identical Section 5 wire format.
# ------------------------------------------------------------------
add_executable(test_seq_desc_truth test_seq_desc_truth.cpp)
target_compile_definitions(test_seq_desc_truth PRIVATE
    MRDSERVER_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../pulserverlib-tests/expected")
target_link_libraries(test_seq_desc_truth PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_trajectory_cache_loader)
gtest_discover_tests(test_seq_desc_truth)
enable_testing()
