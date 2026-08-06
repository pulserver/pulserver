# pulseqpp -- the standalone C++ Pulseq sequence library.
#
# Owns the Sequence model (libraries, block table, deduplication, both
# writers) and the reader.  No Python, no pulseg: the same "reusable on its
# own" contract csrc/include/pulseq/pulseq.h states for the C reader below it.

add_library(pulseqpp STATIC
    sequence.cpp
    dedup.cpp
    shape.cpp
    write_text.cpp
    write_binary.cpp
    read.cpp
    raw64.cpp
    # The vendored MD5, compiled as C for the [SIGNATURE] section. raw64.cpp
    # has its own copy inside its namespace; the two do not meet.
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/src/vendor/external_md5.c
)

target_include_directories(pulseqpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/include/pulseq
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/src/pulseq
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/src/vendor
)

set_target_properties(pulseqpp PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# raw64.cpp compiles the C reader as C++ (see cxx/include/pulseq/raw64.hpp).
# The vendored MD5 it pulls in still says `register`, which C++17 removed;
# that is the vendor's file and not ours to modernize.
if(NOT MSVC)
    set_source_files_properties(raw64.cpp PROPERTIES COMPILE_OPTIONS "-Wno-register")
endif()

# render_rows splits the two scan-length sections across threads.
find_package(Threads REQUIRED)
target_link_libraries(pulseqpp PUBLIC Threads::Threads)
