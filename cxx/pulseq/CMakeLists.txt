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
    sequence_file.cpp
    raw64.cpp
    trajectory.cpp
    kspace.cpp
    kspace_sequence.cpp
    moments.cpp
    autolabel.cpp
    expand.cpp
    # The vendored MD5, compiled as C for the [SIGNATURE] section. raw64.cpp
    # has its own copy inside its namespace; the two do not meet.
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/src/vendor/external_md5.c
    # The RF spectrum, at the library's ordinary (float) precision, and the
    # transform under it.  Unlike the reader this is NOT compiled a second time
    # inside raw64: a -6 dB width read off a hundred-thousand-point grid has
    # nothing to gain from float64, and kissfft would have to be instantiated
    # twice to give it that.
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/src/pulseq/pulseq_rf.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/src/vendor/external_kiss_fft.c
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
