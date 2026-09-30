cmake_minimum_required(VERSION 3.18)
project(pulserver_extensions LANGUAGES C CXX)

set(CMAKE_C_STANDARD 90)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Development.Module REQUIRED)
find_package(Threads REQUIRED)

# The scanner compiles src/c as ANSI C: a construct C89 rejects fails this
# build too, before it can reach the interpreter.
add_compile_options($<$<COMPILE_LANGUAGE:C>:-pedantic-errors>)

add_subdirectory(${CMAKE_SOURCE_DIR}/src/c ${CMAKE_BINARY_DIR}/c_build)

pybind11_add_module(_ext
    ${CMAKE_SOURCE_DIR}/src/cpp/module.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/playout.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/ir/from_libraries.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/ir/pulseq_file.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/ir/dedup.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/ir/structure.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/ir/collection.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/ir/waves.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/ir/wave_plan.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/ir/repetition.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/bloch/bindings.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/bloch/bloch.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/bloch/events.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/bloch/nufft.cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/bloch/repetitions.cpp
)
# The isochromat engine partitions its work across standard C++ threads.
target_link_libraries(_ext PRIVATE pulseg Threads::Threads)
# The passes are pulseg's own and see its private headers.
target_include_directories(_ext PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_SOURCE_DIR}/src/c
    ${CMAKE_SOURCE_DIR}/src/c/pulseq
    ${CMAKE_SOURCE_DIR}/src/cpp/ir
)

# The extension is loaded by whatever interpreter it lands in front of, and an
# interpreter ships the C++ runtime it was built against -- often older than
# the compiler that builds this. Carrying the runtime inside the extension
# removes that dependency rather than betting on the host's: without it, a
# distribution whose libstdc++ predates the build compiler refuses to load the
# extension at all. Nothing here hands C++ objects to another extension, which
# is what would make one runtime per module unsafe.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_link_options(_ext PRIVATE -static-libstdc++ -static-libgcc)
endif()

# Built into the build tree, never beside the sources: a binary left in the
# source tree would be swept into wheels built for other interpreters.
set_target_properties(_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/src/pulserver"
)

install(TARGETS _ext DESTINATION pulserver)
