set(PULSEG_SOURCES
    src/core/pulseg_error.c
    src/core/pulseg_math.c
    src/io/pulseg_parse.c
    src/structure/pulseg_dedup.c
    src/structure/pulseg_structure.c
    src/core/pulseg_core.c
    src/waveforms/pulseg_waveforms.c
    src/safety/pulseg_safety.c
    src/structure/pulseg_getters.c
    src/freqmod/pulseg_freqmod.c
    src/io/pulseg_protocol.c
    src/waveforms/pulseg_trajectory.c
    src/cache/pulseg_cache.c
    src/structure/pulseg_seqdesc.c
    src/cache/pulseg_cache_seqdesc.c
)

set(MD5_SOURCES     src/vendor/external_md5.c)
set(KISSFFT_SOURCES src/vendor/external_kiss_fft.c src/vendor/external_kiss_fftr.c)

add_library(pulseg STATIC ${PULSEG_SOURCES} ${KISSFFT_SOURCES} ${MD5_SOURCES})
target_include_directories(pulseg PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pulseg
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vendor
)

add_library(pulseg_bridge STATIC src/core/pulseg_bridge.c)
target_link_libraries(pulseg_bridge PUBLIC pulseg)
target_include_directories(pulseg_bridge PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/pulseg)
