# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

project(pulserverlib C)

# Minimal ANSI C89 Pulseq reader library
file(GLOB PULSEQ "pulseq/*.c" "pulseq/*.h")
# file(GLOB PULSERVER "pulserver/*.c" "pulserver/*.h")

add_library(pulseqlib ${PULSEQ})
target_include_directories(pulseqlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pulseqlib m)

# dd_library(pulserverlib ${PULSERVER})
# target_include_directories(pulserverlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_include_directories(pulserverlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/pulseq)
# target_link_libraries(pulserverlib PUBLIC pulseqlib)