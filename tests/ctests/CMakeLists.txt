# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)
project(pulserverlibTests C)

# Add the csrc library folder as a subdirectory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../csrc ${CMAKE_BINARY_DIR}/csrc_build)

# Collect all test source files
file(GLOB TEST_SOURCES "*.c")

# Create an executable for the tests
add_executable(run_tests ${TEST_SOURCES})

# Define the test root folder at compile time (repo root = two levels up from tests/ctests)
target_compile_definitions(run_tests PRIVATE
    TEST_ROOT_DIR="${CMAKE_SOURCE_DIR}/../.."
)

# Specify the output directory for the compiled test executable
set_target_properties(run_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Link the test executable with the libraries under test
target_link_libraries(run_tests PRIVATE pulseqlib m)

# Public API headers live in csrc/
target_include_directories(run_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc)

# Expose POSIX timers for minunit.h (clockid_t, clock_gettime, etc.)
target_compile_definitions(run_tests PRIVATE _POSIX_C_SOURCE=200112L)

# Enable warnings; use C99 for the test harness (minunit needs POSIX timers).
# The library itself is verified as C89-clean by scripts/check_c89_compliance.sh.
target_compile_options(run_tests PRIVATE -Wall -Wextra -std=c99 -pedantic -Werror)