# ---------------------------------------------------------------------------
#  pulseq -- the Pulseq file model the IR keeps its waveforms in, and the
#  run-length codec that reads them.
#  Depends on nothing else in this repository; pulseg links it, never the
#  reverse.  Keep this target free of any pulseg include directory so the
#  one-way dependency stays enforced by the build itself.
# ---------------------------------------------------------------------------
set(PULSEQ_SOURCES pulseq/pulseq_shape.c)

add_library(pulseq STATIC ${PULSEQ_SOURCES})
target_include_directories(pulseq PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pulseq
)
target_include_directories(pulseq PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/pulseq)

# ---------------------------------------------------------------------------
#  pulseg -- the vendor-neutral intermediate representation and the binary
#  cache a scanner loads it from.
# ---------------------------------------------------------------------------
set(PULSEG_SOURCES
    core/pulseg_error.c
    core/pulseg_math.c
    core/pulseg_core.c
    io/pulseg_opts.c
    io/pulseg_protocol.c
    structure/pulseg_getters.c
    structure/pulseg_descriptor.c
    structure/pulseg_waves.c
    cache/pulseg_cache.c
)

add_library(pulseg STATIC ${PULSEG_SOURCES})
target_link_libraries(pulseg PUBLIC pulseq)
target_include_directories(pulseg PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pulseg
)
# The tree root holds pulseg_internal.h: private, so `include/` is exactly the
# public API and nothing else.
target_include_directories(pulseg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
