# csrc/CMakeLists.txt -- pulseqlib library build

# Collect all library sources
set(PULSEQLIB_SOURCES
    pulseqlib_error.c
    pulseqlib_math.c
    pulseqlib_parse.c
    pulseqlib_core.c
    pulseqlib_segment.c
    pulseqlib_safety.c
    pulseqlib_accessor.c
)

# kissfft sources (co-located for self-containment)
set(KISSFFT_SOURCES
    external_kiss_fft.c
    external_kiss_fftr.c
)

# Public headers (for IDE / install)
set(PULSEQLIB_HEADERS
    pulseqlib.h
    pulseqlib_config.h
    pulseqlib_types.h
)

# Build the library
add_library(pulseqlib STATIC
    ${PULSEQLIB_SOURCES}
    ${KISSFFT_SOURCES}
)

# Include the current directory so #include "pulseqlib.h" works
target_include_directories(pulseqlib
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
)