# The C++ examples, built so a broken one fails the build rather than the
# reader. Each mirrors a C example; `vendor.hpp` stubs the scanner side.
cmake_minimum_required(VERSION 3.10)

if(NOT TARGET pulseg)
    project(pulserver_cpp_examples C CXX)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../src/c ${CMAKE_BINARY_DIR}/c_build)
endif()

set(PULSERVER_CPP_EXAMPLES
    safety_gate
    stage1_setup
    stage2_prescription
    stage3_preparation
    stage4_waveform_generation
    stage5_playout
)

foreach(example ${PULSERVER_CPP_EXAMPLES})
    add_executable(cpp_example_${example} ${CMAKE_CURRENT_SOURCE_DIR}/${example}.cpp)
    target_link_libraries(cpp_example_${example} PRIVATE pulseg pulseg_bridge)
    if(NOT WIN32)
        target_link_libraries(cpp_example_${example} PRIVATE m)
    endif()
    target_include_directories(cpp_example_${example} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cpp/include
    )
    set_target_properties(cpp_example_${example} PROPERTIES
        OUTPUT_NAME cpp_${example}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(cpp_example_${example} PRIVATE -Wall -Wextra -Werror)
    endif()
endforeach()
