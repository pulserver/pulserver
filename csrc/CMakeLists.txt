# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

project(pulserverlib C)

# External libraries
file(GLOB KISSFFT "../external/kissfft/*.c")

# Minimal ANSI C89 Pulseq reader library
file(GLOB PULSEQ "pulseq/*.c" "pulseq/*.h")

add_library(pulseqlib ${PULSEQ} ${KISSFFT})
target_include_directories(pulseqlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(pulseqlib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../external/kissfft)
target_link_libraries(pulseqlib m)