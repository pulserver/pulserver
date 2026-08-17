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
# Build the pulseg C library (the scanner-side model several tests
# exercise, and the cache writer the parity staging uses).
# ------------------------------------------------------------------
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../csrc ${CMAKE_CURRENT_BINARY_DIR}/csrc_build)

set(MRDSERVER_CSRC ${CMAKE_CURRENT_SOURCE_DIR}/../../cxx/recon)

# ------------------------------------------------------------------
# pulseq::File / pulseq::FileSet C++ wrapper (cxx/include/pulseq).
# pulseq (csrc) is already built unconditionally above (pulseg depends on it).
# ------------------------------------------------------------------
add_executable(test_pulseq_file test_pulseq_file.cpp)
target_include_directories(test_pulseq_file PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cxx/include/pulseq)
target_compile_definitions(test_pulseq_file PRIVATE
    PULSEQ_FILE_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../python/fixtures")
target_link_libraries(test_pulseq_file PRIVATE pulseq GTest::gtest_main)

# ------------------------------------------------------------------
# pulseqpp -- the standalone C++ Pulseq sequence library (cxx/pulseq).
# ------------------------------------------------------------------
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../cxx/pulseq
                 ${CMAKE_CURRENT_BINARY_DIR}/pulseqpp_build)

# The recon-side reader: SequenceCache from the .seq chain plus its
# format-agnostic consumers. No cache file involved.
add_library(sequence_reader_obj OBJECT
    ${MRDSERVER_CSRC}/sequence_file_reader.cpp
    ${MRDSERVER_CSRC}/sequence_cache.cpp)
target_include_directories(sequence_reader_obj PUBLIC
    ${MRDSERVER_CSRC}
    ${ISMRMRD_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/include/pulseq)
target_link_libraries(sequence_reader_obj PUBLIC
    pulseqpp ISMRMRD::ISMRMRD)

add_executable(test_sequence_reader test_sequence_reader.cpp)
target_compile_definitions(test_sequence_reader PRIVATE
    PULSEQ_CORPUS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../python/fixtures")
target_link_libraries(test_sequence_reader PRIVATE
    sequence_reader_obj
    GTest::gtest_main
    ISMRMRD::ISMRMRD)

add_executable(test_pulseq_sequence test_pulseq_sequence.cpp)
target_link_libraries(test_pulseq_sequence PRIVATE pulseqpp GTest::gtest_main)

add_executable(test_sequence_file test_sequence_file.cpp)
target_compile_definitions(test_sequence_file PRIVATE
    PULSEQ_CORPUS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../python/fixtures")
target_link_libraries(test_sequence_file PRIVATE pulseqpp GTest::gtest_main)

add_executable(test_pulseq_dedup test_pulseq_dedup.cpp)
target_compile_definitions(test_pulseq_dedup PRIVATE
    PULSEQ_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../utils/expected")
target_link_libraries(test_pulseq_dedup PRIVATE pulseqpp GTest::gtest_main)

add_executable(test_pulseq_roundtrip test_pulseq_roundtrip.cpp)
target_compile_definitions(test_pulseq_roundtrip PRIVATE
    PULSEQ_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../utils/expected")
target_link_libraries(test_pulseq_roundtrip PRIVATE pulseqpp GTest::gtest_main)

add_executable(test_pulseq_trajectory test_pulseq_trajectory.cpp)
target_link_libraries(test_pulseq_trajectory PRIVATE pulseqpp GTest::gtest_main)

add_executable(test_pulseq_kspace test_pulseq_kspace.cpp)
target_compile_definitions(test_pulseq_kspace PRIVATE
    PULSEQ_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../utils/expected"
    PULSEQ_CORPUS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../python/fixtures")
target_link_libraries(test_pulseq_kspace PRIVATE pulseqpp GTest::gtest_main)

add_executable(test_pulseq_ktraj test_pulseq_ktraj.cpp)
target_compile_definitions(test_pulseq_ktraj PRIVATE
    PULSEQ_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../utils/expected"
    PULSEQ_CORPUS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../python/fixtures")
target_link_libraries(test_pulseq_ktraj PRIVATE pulseqpp GTest::gtest_main)

add_executable(test_pulseq_moments test_pulseq_moments.cpp)
target_compile_definitions(test_pulseq_moments PRIVATE
    PULSEQ_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../utils/expected"
    PULSEQ_CORPUS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../python/fixtures")
target_link_libraries(test_pulseq_moments PRIVATE pulseqpp GTest::gtest_main)

add_executable(test_pulseq_autolabel test_pulseq_autolabel.cpp)
target_compile_definitions(test_pulseq_autolabel PRIVATE
    PULSEQ_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../utils/expected"
    PULSEQ_CORPUS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../python/fixtures")
target_link_libraries(test_pulseq_autolabel PRIVATE pulseqpp GTest::gtest_main)

add_executable(test_pulseq_expand test_pulseq_expand.cpp)
target_link_libraries(test_pulseq_expand PRIVATE pulseqpp GTest::gtest_main)

add_executable(test_pulseq_shape test_pulseq_shape.cpp)
target_link_libraries(test_pulseq_shape PRIVATE pulseqpp GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_pulseq_file)
gtest_discover_tests(test_pulseq_sequence)
gtest_discover_tests(test_sequence_file)
gtest_discover_tests(test_sequence_reader)
gtest_discover_tests(test_pulseq_dedup)
gtest_discover_tests(test_pulseq_roundtrip)
gtest_discover_tests(test_pulseq_trajectory)
gtest_discover_tests(test_pulseq_kspace)
gtest_discover_tests(test_pulseq_ktraj)
gtest_discover_tests(test_pulseq_moments)
gtest_discover_tests(test_pulseq_autolabel)
gtest_discover_tests(test_pulseq_expand)
gtest_discover_tests(test_pulseq_shape)


enable_testing()
