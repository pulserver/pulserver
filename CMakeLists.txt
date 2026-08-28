cmake_minimum_required(VERSION 3.18)
project(pulserver_extensions LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Development.Module REQUIRED)
find_package(Threads REQUIRED)

# The libraries the extension binds, each built once. src/c/CMakeLists.txt
# owns the C targets (pulseq, then pulseg on top of it) and src/cpp/pulseq
# owns pulseqpp, which links the same pulseq rather than compiling a second
# copy of it.
add_subdirectory(${CMAKE_SOURCE_DIR}/src/c ${CMAKE_BINARY_DIR}/c_build)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/cpp/pulseq ${CMAKE_BINARY_DIR}/pulseqpp_build)

# pulserver._ext -- one extension module holding every compiled kernel, as
# five submodules. src/python/bindings/module.cpp creates them; each source
# below fills one in.
#
# Built into the build tree, never beside the sources: the wheel takes the
# module from the install step, so a binary left in the source tree would be
# swept into wheels built for every other interpreter too.
pybind11_add_module(_ext
    ${CMAKE_SOURCE_DIR}/src/python/bindings/module.cpp
    ${CMAKE_SOURCE_DIR}/src/python/bindings/pulseg.cpp
    ${CMAKE_SOURCE_DIR}/src/python/bindings/pulseqpp.cpp
    ${CMAKE_SOURCE_DIR}/src/python/bindings/pulseqpp_eventtypes.cpp
    ${CMAKE_SOURCE_DIR}/src/python/bindings/arbgrad.cpp
    ${CMAKE_SOURCE_DIR}/src/python/bindings/sampling.cpp
    ${CMAKE_SOURCE_DIR}/src/python/bindings/recon_cpu.cpp
    # The vendored MRArbGrad core; see src/cpp/arbgrad/vendor/mrarbgrad/NOTICE.md.
    ${CMAKE_SOURCE_DIR}/src/cpp/arbgrad/vendor/mrarbgrad/mag/Mag.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/arbgrad/vendor/mrarbgrad/utility/global.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/arbgrad/vendor/mrarbgrad/utility/v3.cpp
)

target_include_directories(_ext PRIVATE
    ${CMAKE_SOURCE_DIR}/src/python/bindings
    ${CMAKE_SOURCE_DIR}/src/c
    ${CMAKE_SOURCE_DIR}/src/c/pulseq
    ${CMAKE_SOURCE_DIR}/src/c/vendor
    ${CMAKE_SOURCE_DIR}/src/cpp/include
    ${CMAKE_SOURCE_DIR}/src/cpp/include/pulseg
    ${CMAKE_SOURCE_DIR}/src/cpp/arbgrad/vendor/mrarbgrad
)

# recon_cpu.cpp partitions independent frequency locations across standard
# C++ threads; pulseqpp splits the two scan-length sections the same way.
target_link_libraries(_ext PRIVATE pulseg pulseqpp Threads::Threads)

set_target_properties(_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/src/python/pulserver"
)

install(TARGETS _ext DESTINATION pulserver)

option(PULSERVER_BUILD_CPP_TESTS "Build lightweight C++ unit tests" OFF)
if(PULSERVER_BUILD_CPP_TESTS)
    enable_testing()
    add_subdirectory(${CMAKE_SOURCE_DIR}/tests/cpptests ${CMAKE_BINARY_DIR}/cpptests_build)
endif()

option(PULSERVER_BUILD_EXAMPLES "Build the C examples" OFF)
if(PULSERVER_BUILD_EXAMPLES)
    add_subdirectory(${CMAKE_SOURCE_DIR}/examples/c ${CMAKE_BINARY_DIR}/examples_build)
endif()
