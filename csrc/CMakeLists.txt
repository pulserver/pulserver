# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

project(pulserverlib C)

# Minimal ANSI C89 Pulseq reader library
file(GLOB PULSEQ "pulseq/*.c" "pulseq/*.h")

add_library(pulseqlib ${PULSEQ})
target_include_directories(pulseqlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pulseqlib m)

# Higher abstraction library (future, placeholder)
# file(GLOB PULSEG_INTERNAL "pulseg/*.c" "pulseg/*.h")
# add_library(pulseg pulseg.c ${PULSEG_INTERNAL})
# target_include_directories(pulseg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})