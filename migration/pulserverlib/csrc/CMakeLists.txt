# csrc/CMakeLists.txt -- pulseqlib library build

# Collect all library sources
set(PULSEQLIB_SOURCES
    pulseqlib_error.c
    pulseqlib_math.c
    pulseqlib_parse.c
    pulseqlib_dedup.c
    pulseqlib_structure.c
    pulseqlib_core.c
    pulseqlib_cache.c
    pulseqlib_waveforms.c
    pulseqlib_safety.c
    pulseqlib_getters.c
    pulseqlib_freqmod.c
)

# MD5 source (needed by pulseqlib_parse for signature verification)
set(MD5_SOURCES
    external_md5.c
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
    ${MD5_SOURCES}
)

# Include the current directory so #include "pulseqlib.h" works
target_include_directories(pulseqlib
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
)