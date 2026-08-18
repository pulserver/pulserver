# pulseqpp -- the standalone C++ Pulseq sequence library.
#
# Owns the Sequence model (libraries, block table, deduplication, both
# writers) and the reader.  No Python, no pulseg: the same "reusable on its
# own" contract src/c/include/pulseq/pulseq.h states for the C reader below it.

add_library(pulseqpp STATIC
    sequence.cpp
    dedup.cpp
    shape.cpp
    write_text.cpp
    write_binary.cpp
    read.cpp
    sequence_file.cpp
    raw64.cpp
    trajectory.cpp
    kspace.cpp
    ktraj.cpp
    moments.cpp
    autolabel.cpp
    expand.cpp
)

# The C reader, its MD5 for the [SIGNATURE] section and the transform the RF
# spectrum runs on all come from src/c as one library rather than being
# compiled again here.  raw64.cpp does keep its own copy of the reader, inside
# its namespace; that one is compiled as C++ and the two never meet.
if(NOT TARGET pulseq)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../c ${CMAKE_CURRENT_BINARY_DIR}/c_build)
endif()

target_include_directories(pulseqpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../c/pulseq
    ${CMAKE_CURRENT_SOURCE_DIR}/../../c/vendor
)

set_target_properties(pulseqpp PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# raw64.cpp compiles the C reader as C++ (see src/cpp/include/pulseq/raw64.hpp).
# The vendored MD5 it pulls in still says `register`, which C++17 removed;
# that is the vendor's file and not ours to modernize.
if(NOT MSVC)
    set_source_files_properties(raw64.cpp PROPERTIES COMPILE_OPTIONS "-Wno-register")
endif()

# render_rows splits the two scan-length sections across threads.
find_package(Threads REQUIRED)
target_link_libraries(pulseqpp PUBLIC pulseq Threads::Threads)
