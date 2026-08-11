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
# Build the pulseg C library once, unconditionally (trajectory_cache_reader
# now calls into it -- Stage 4 recon-cache-to-C port -- so it is no longer
# needed only inside the pulseq-adapter block below).
# ------------------------------------------------------------------
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../csrc ${CMAKE_CURRENT_BINARY_DIR}/csrc_build)

# ------------------------------------------------------------------
# Build trajectory_cache_reader as an OBJECT library to reuse here
# without depending on the full GadgetronClient build.
# ------------------------------------------------------------------
set(MRDSERVER_CSRC ${CMAKE_CURRENT_SOURCE_DIR}/../../cxx/recon)

add_library(trajectory_cache_reader_obj OBJECT
    ${MRDSERVER_CSRC}/trajectory_cache_reader.cpp)
target_include_directories(trajectory_cache_reader_obj PUBLIC
    ${MRDSERVER_CSRC}
    ${ISMRMRD_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS})
target_link_libraries(trajectory_cache_reader_obj PUBLIC pulseg_recon ISMRMRD::ISMRMRD)

# ------------------------------------------------------------------
# Test executable.
# ------------------------------------------------------------------
add_executable(test_trajectory_cache_loader test_trajectory_cache_loader.cpp)

target_include_directories(test_trajectory_cache_loader PRIVATE
    ${MRDSERVER_CSRC}
    ${ISMRMRD_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS})

# Path to the .pge fixtures, baked in at configure time.
target_compile_definitions(test_trajectory_cache_loader PRIVATE
    MRDSERVER_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../utils/expected")

target_link_libraries(test_trajectory_cache_loader PRIVATE
    trajectory_cache_reader_obj
    GTest::gtest_main
    ISMRMRD::ISMRMRD)

# ------------------------------------------------------------------
# Section 5 (SEQUENCEDESCRIPTION) byte-exact truth test.
# Validates that the C-side cache writer (pulseg_cache_seqdesc.c)
# and the MATLAB TruthBuilder.exportSequenceDescription emit the
# identical Section 5 wire format.
# ------------------------------------------------------------------
add_executable(test_seq_desc_truth test_seq_desc_truth.cpp)
target_compile_definitions(test_seq_desc_truth PRIVATE
    MRDSERVER_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../utils/expected")
target_link_libraries(test_seq_desc_truth PRIVATE GTest::gtest_main)

# ------------------------------------------------------------------
# pulseq::File / pulseq::FileSet C++ wrapper (cxx/include/pulseq).
# pulseq (csrc) is already built unconditionally above (pulseg depends on it).
# ------------------------------------------------------------------
add_executable(test_pulseq_file test_pulseq_file.cpp)
target_include_directories(test_pulseq_file PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cxx/include/pulseq)
target_compile_definitions(test_pulseq_file PRIVATE
    PULSEQ_FILE_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../utils/expected")
target_link_libraries(test_pulseq_file PRIVATE pulseq GTest::gtest_main)

# ------------------------------------------------------------------
# pulseqpp -- the standalone C++ Pulseq sequence library (cxx/pulseq).
# ------------------------------------------------------------------
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../cxx/pulseq
                 ${CMAKE_CURRENT_BINARY_DIR}/pulseqpp_build)

add_executable(test_pulseq_sequence test_pulseq_sequence.cpp)
target_link_libraries(test_pulseq_sequence PRIVATE pulseqpp GTest::gtest_main)

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
    PULSEQ_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../utils/expected")
target_link_libraries(test_pulseq_kspace PRIVATE pulseqpp GTest::gtest_main)

add_executable(test_pulseq_autolabel test_pulseq_autolabel.cpp)
target_compile_definitions(test_pulseq_autolabel PRIVATE
    PULSEQ_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../utils/expected")
target_link_libraries(test_pulseq_autolabel PRIVATE pulseqpp GTest::gtest_main)

add_executable(test_pulseq_expand test_pulseq_expand.cpp)
target_link_libraries(test_pulseq_expand PRIVATE pulseqpp GTest::gtest_main)

add_executable(test_pulseq_shape test_pulseq_shape.cpp)
target_link_libraries(test_pulseq_shape PRIVATE pulseqpp GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_trajectory_cache_loader)
gtest_discover_tests(test_seq_desc_truth)
gtest_discover_tests(test_pulseq_file)
gtest_discover_tests(test_pulseq_sequence)
gtest_discover_tests(test_pulseq_dedup)
gtest_discover_tests(test_pulseq_roundtrip)
gtest_discover_tests(test_pulseq_trajectory)
gtest_discover_tests(test_pulseq_kspace)
gtest_discover_tests(test_pulseq_autolabel)
gtest_discover_tests(test_pulseq_expand)
gtest_discover_tests(test_pulseq_shape)

# ------------------------------------------------------------------
# Stage 4 (recon-cache-to-C) Step 0: golden-dump CLI tool. Dumps every
# field trajectory_cache_reader parses from a .pge/.pseg fixture to a
# deterministic text file; used to capture pre-port golden dumps and,
# later, to prove the C-ported reader is byte-for-byte equivalent.
# ------------------------------------------------------------------
add_executable(dump_recon_cache dump_recon_cache.cpp)
target_include_directories(dump_recon_cache PRIVATE
    ${MRDSERVER_CSRC}
    ${ISMRMRD_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS})
target_link_libraries(dump_recon_cache PRIVATE trajectory_cache_reader_obj ISMRMRD::ISMRMRD)

# ------------------------------------------------------------------
# Pulseq adapter equivalence test (Stage 3, D4): loads a fixture .seq
# through the ExternalSequence adapter and through pulseg's native C
# reader, and asserts the resulting collections agree.
# ------------------------------------------------------------------
set(PULSEQ_SUBMODULE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../external/pulseq)
if(EXISTS ${PULSEQ_SUBMODULE_DIR}/src/ExternalSequence.cpp)
    # pulseg (csrc) is already built unconditionally above.
    add_library(pulseq_upstream STATIC ${PULSEQ_SUBMODULE_DIR}/src/ExternalSequence.cpp)
    target_include_directories(pulseq_upstream PUBLIC ${PULSEQ_SUBMODULE_DIR}/src)

    set(PULSEQ_ADAPTER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../cxx/pulseq_adapter)
    add_library(pulseg_pulseq_adapter STATIC
        ${PULSEQ_ADAPTER_DIR}/pulseq_adapter.cpp
        ${PULSEQ_ADAPTER_DIR}/collection_ext_seq.cpp)
    target_include_directories(pulseg_pulseq_adapter PUBLIC
        ${PULSEQ_ADAPTER_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cxx/include/pulseg)
    target_link_libraries(pulseg_pulseq_adapter PUBLIC pulseg pulseq_upstream)

    add_executable(test_pulseq_adapter_equivalence test_pulseq_adapter_equivalence.cpp)
    target_include_directories(test_pulseq_adapter_equivalence PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../cxx/include/pulseg)
    target_compile_definitions(test_pulseq_adapter_equivalence PRIVATE
        PULSEQ_ADAPTER_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../utils/expected")
    target_link_libraries(test_pulseq_adapter_equivalence PRIVATE
        pulseg_pulseq_adapter GTest::gtest_main)

    gtest_discover_tests(test_pulseq_adapter_equivalence)
else()
    message(WARNING "external/pulseq submodule not initialized -- "
                     "skipping test_pulseq_adapter_equivalence "
                     "(run 'git submodule update --init external/pulseq')")
endif()

enable_testing()
