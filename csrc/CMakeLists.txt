option(PULSEQLIB_BUILD_CACHE   "Build the binary cache writer (pulseqlib_cache.c)" ON)
option(PULSEQLIB_BUILD_SEQDESC "Build the sequence-description / Bloch-sim helpers" OFF)

set(PULSEQLIB_SOURCES
    src/core/pulseqlib_error.c
    src/core/pulseqlib_math.c
    src/io/pulseqlib_parse.c
    src/structure/pulseqlib_dedup.c
    src/structure/pulseqlib_structure.c
    src/core/pulseqlib_core.c
    src/waveforms/pulseqlib_waveforms.c
    src/safety/pulseqlib_safety.c
    src/structure/pulseqlib_getters.c
    src/freqmod/pulseqlib_freqmod.c
    src/io/pulseqlib_protocol.c
    src/waveforms/pulseqlib_trajectory.c
)

if(PULSEQLIB_BUILD_CACHE)
    list(APPEND PULSEQLIB_SOURCES src/cache/pulseqlib_cache.c)
endif()

if(PULSEQLIB_BUILD_SEQDESC)
    list(APPEND PULSEQLIB_SOURCES src/structure/pulseqlib_seqdesc.c)
    if(PULSEQLIB_BUILD_CACHE)
        list(APPEND PULSEQLIB_SOURCES src/cache/pulseqlib_cache_seqdesc.c)
    endif()
endif()

set(MD5_SOURCES     src/vendor/external_md5.c)
set(KISSFFT_SOURCES src/vendor/external_kiss_fft.c src/vendor/external_kiss_fftr.c)

add_library(pulseqlib STATIC ${PULSEQLIB_SOURCES} ${KISSFFT_SOURCES} ${MD5_SOURCES})
target_include_directories(pulseqlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(PULSEQLIB_BUILD_SEQDESC AND PULSEQLIB_BUILD_CACHE)
    target_compile_definitions(pulseqlib PUBLIC PULSEQLIB_HAVE_SEQDESC)
endif()

add_library(pulseqlib_bridge STATIC src/core/pulseqlib_bridge.c)
target_link_libraries(pulseqlib_bridge PUBLIC pulseqlib)
target_include_directories(pulseqlib_bridge PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
