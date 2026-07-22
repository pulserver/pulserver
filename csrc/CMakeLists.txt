# ---------------------------------------------------------------------------
#  pulseq -- standalone Pulseq .seq reader.
#  Depends on nothing else in this repository; pulseg links it, never the
#  reverse.  Keep this target free of any pulseg include directory so the
#  one-way dependency stays enforced by the build itself.
# ---------------------------------------------------------------------------
set(PULSEQ_SOURCES
    src/pulseq/pulseq_parse.c
    src/pulseq/pulseq_names.c
)

set(MD5_SOURCES     src/vendor/external_md5.c)
set(KISSFFT_SOURCES src/vendor/external_kiss_fft.c src/vendor/external_kiss_fftr.c)

add_library(pulseq STATIC ${PULSEQ_SOURCES} ${MD5_SOURCES})
target_include_directories(pulseq PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pulseq
)
target_include_directories(pulseq PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pulseq
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vendor
)

# ---------------------------------------------------------------------------
#  pulseg -- the vendor-neutral intermediate representation, safety analysis,
#  waveform/trajectory derivation and binary cache built on top of pulseq.
# ---------------------------------------------------------------------------
set(PULSEG_SOURCES
    src/core/pulseg_error.c
    src/core/pulseg_math.c
    src/core/pulseg_core.c
    src/io/pulseg_opts.c
    src/io/pulseg_protocol.c
    src/structure/pulseg_dedup.c
    src/structure/pulseg_structure.c
    src/structure/pulseg_getters.c
    src/structure/pulseg_seqdesc.c
    src/waveforms/pulseg_waveforms.c
    src/safety/pulseg_safety.c
    src/freqmod/pulseg_freqmod.c
    src/cache/pulseg_cache.c
    src/cache/pulseg_cache_seqdesc.c
    src/cache/pulseg_trajectory.c
)

add_library(pulseg STATIC ${PULSEG_SOURCES} ${KISSFFT_SOURCES})
target_link_libraries(pulseg PUBLIC pulseq)
target_include_directories(pulseg PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pulseg
)
# src/ and src/vendor hold pulseg_internal.h and the bundled kissfft: private,
# so `include/` is exactly the public API and nothing else.
target_include_directories(pulseg PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vendor
)

# ---------------------------------------------------------------------------
#  pulseg_recon -- standalone reader for the recon-relevant cache sections.
#  A leaf consumer: the PSD never compiles it, and nothing in pulseg core
#  calls into it.
# ---------------------------------------------------------------------------
add_library(pulseg_recon STATIC src/recon/pulseg_recon.c)
target_link_libraries(pulseg_recon PUBLIC pulseg)

# ---------------------------------------------------------------------------
#  pulseg_bridge -- out-of-process sequence-generation bridge (server infra).
# ---------------------------------------------------------------------------
add_library(pulseg_bridge STATIC src/bridge/pulseg_bridge.c)
target_link_libraries(pulseg_bridge PUBLIC pulseg)
