# The C examples, built so a broken one fails the build rather than the
# reader. Each is a whole stage of an interpreter and links nothing but the
# public C library; `vendor.h` stubs the scanner side.
#
# Usable two ways: as a subdirectory of the top-level project, where the
# `pulseg` target already exists, and on its own (scripts/build_examples.sh),
# where it builds the library too.
cmake_minimum_required(VERSION 3.10)

if(NOT TARGET pulseg)
    project(pulserver_c_examples C CXX)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../src/c ${CMAKE_BINARY_DIR}/c_build)
endif()

set(PULSERVER_C_EXAMPLES
    safety_gate
    stage1_setup
    stage2_prescription
    stage3_preparation
    stage4_waveform_generation
    stage5_playout
)

foreach(example ${PULSERVER_C_EXAMPLES})
    add_executable(example_${example} ${CMAKE_CURRENT_SOURCE_DIR}/${example}.c)
    target_link_libraries(example_${example} PRIVATE pulseg pulseg_bridge)
    if(NOT WIN32)
        target_link_libraries(example_${example} PRIVATE m)
    endif()
    target_include_directories(example_${example} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(example_${example} PROPERTIES
        OUTPUT_NAME ${example}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
    )
    # A vendor's toolchain is the point of these, so they are held to the
    # same C89 the library is.
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(example_${example} PRIVATE -std=c89 -pedantic -Wall -Werror)
    endif()
endforeach()

# The C++ counterparts live beside these and share the same library target.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cpp ${CMAKE_BINARY_DIR}/cpp_examples_build)
