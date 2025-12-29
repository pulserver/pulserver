
# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)
project(pulserverlibTests C)

# Add the csrc library folder as a subdirectory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../csrc ${CMAKE_BINARY_DIR}/csrc_build)

# Collect all test source files
file(GLOB TEST_SOURCES "*.c")

# Create an executable for the tests
add_executable(run_tests ${TEST_SOURCES})

# Define the test root folder at compile time
target_compile_definitions(run_tests PRIVATE
    TEST_ROOT_DIR="${CMAKE_SOURCE_DIR}/.."
)

# Specify the output directory for the compiled test executable
set_target_properties(run_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Link the test executable with the libraries under test
target_link_libraries(run_tests PRIVATE pulseqlib pulserverlib)

# Use the top-level csrc directory for public API headers
target_include_directories(run_tests PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/pulseq)
target_include_directories(run_tests PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/pulserver)

# Enable warnings and enforce ANSI C89 compliance
target_compile_options(run_tests PRIVATE -Wall -Wextra -std=c89 -pedantic -Werror)