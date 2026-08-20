# ---------------------------------------------------------------------------
#  pulseq -- standalone Pulseq .seq reader.
#  Depends on nothing else in this repository; pulseg links it, never the
#  reverse.  Keep this target free of any pulseg include directory so the
#  one-way dependency stays enforced by the build itself.
# ---------------------------------------------------------------------------
set(PULSEQ_SOURCES
    pulseq/pulseq_parse.c
    pulseq/pulseq_binary.c
    pulseq/pulseq_names.c
    pulseq/pulseq_rf.c
)

set(MD5_SOURCES     vendor/external_md5.c)
set(KISSFFT_SOURCES vendor/external_kiss_fft.c vendor/external_kiss_fftr.c
                    vendor/external_kiss_fft_double.c)

# kissfft lives here rather than with pulseg because pulseq_rf.c needs it and
# pulseg links pulseq: compiling it into both would define every kissfft symbol
# twice.  pulseg keeps vendor/ on its include path for the headers.
add_library(pulseq STATIC ${PULSEQ_SOURCES} ${MD5_SOURCES} ${KISSFFT_SOURCES})
target_include_directories(pulseq PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pulseq
)
target_include_directories(pulseq PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/pulseq
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor
)

# ---------------------------------------------------------------------------
#  pulseg -- the vendor-neutral intermediate representation, safety analysis,
#  waveform derivation and binary cache built on top of pulseq.
# ---------------------------------------------------------------------------
set(PULSEG_SOURCES
    core/pulseg_error.c
    core/pulseg_math.c
    core/pulseg_core.c
    io/pulseg_opts.c
    io/pulseg_protocol.c
    structure/pulseg_dedup.c
    structure/pulseg_structure.c
    structure/pulseg_getters.c
    structure/pulseg_seqdesc.c
    waveforms/pulseg_waveforms.c
    waveforms/pulseg_chunk.c
    safety/pulseg_safety.c
    safety/pulseg_pns_memo.c
    safety/pulseg_pns_irnich.c
    safety/pulseg_pns_safe.c
    cache/pulseg_cache.c
)

add_library(pulseg STATIC ${PULSEG_SOURCES})
target_link_libraries(pulseg PUBLIC pulseq)
target_include_directories(pulseg PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pulseg
)
# The tree root and vendor/ hold pulseg_internal.h and the bundled kissfft:
# private, so `include/` is exactly the public API and nothing else.
target_include_directories(pulseg PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor
)

# ---------------------------------------------------------------------------
#  pulseg_bridge -- out-of-process sequence-generation bridge (server infra).
#  It drives a child process over pipes with fork/exec and select, so it is
#  built where those exist: the console hosts it serves are POSIX, and the
#  Python extension links pulseg directly rather than through it.
# ---------------------------------------------------------------------------
if(NOT WIN32)
    add_library(pulseg_bridge STATIC bridge/pulseg_bridge.c)
    target_link_libraries(pulseg_bridge PUBLIC pulseg)
endif()
