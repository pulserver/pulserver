# ------------------------------------------------------------------
# pulseg_recon_reader: the pulseg-owned recon cache reader.
#
# Stage 4 (vendor-neutral refactor): trajectory_cache_reader.{h,cpp} is now
# a thin C++ wrapper over the C89 reader pulseg_recon_cache_read()
# (csrc/src/recon/pulseg_recon.c). This target is intentionally separate
# from GadgetronClient below -- the vendored Gadgetron ISMRMRD client (network
# client + NHLBI compression) is NOT part of the pulseg port; only cache
# serialization/deserialization is shared C/C++ code. Non-Gadgetron
# consumers can link this target alone.
# ------------------------------------------------------------------

add_library(pulseg_recon_reader STATIC
    trajectory_cache_reader.h
    trajectory_cache_reader.cpp)

target_compile_features(pulseg_recon_reader PUBLIC cxx_std_17)

target_include_directories(pulseg_recon_reader PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include/mrdserver>)

# Public: pulseg (C library, provides csrc/include/pulseg/*.h transitively
# incl. pulseg_recon.h) and ismrmrd (trajectory_cache_reader.h's class API
# returns ISMRMRD:: types).
target_link_libraries(pulseg_recon_reader PUBLIC pulseg_recon ismrmrd)

install(TARGETS pulseg_recon_reader EXPORT mrdserverTargets
    ARCHIVE DESTINATION lib)
install(FILES trajectory_cache_reader.h ../include/pulseg/recon.hpp
    DESTINATION include/mrdserver)

# ------------------------------------------------------------------
# GadgetronClient: vendored Gadgetron ISMRMRD network client + NHLBI
# compression. Depends on pulseg_recon_reader for the cache reader (used by
# GadgetronEmitter's OnPrep path) -- it no longer compiles the reader itself.
# ------------------------------------------------------------------

set(SOURCE_FILES
    GadgetronClient.h
    gadgetron_ismrmrd_client.h
    gadgetron_ismrmrd_client.cpp
    CompressedFloatBuffer.cpp
    CompressedFloatBufferAvx2.cpp
    CompressedFloatBufferSse41.cpp
    NHLBICompression.h
    cpuisa.h
    cpuisa.cpp)

add_library(GadgetronClient STATIC ${SOURCE_FILES})

target_compile_features(GadgetronClient PUBLIC cxx_std_17)
target_compile_options(GadgetronClient PRIVATE -mavx2 -mfma)

target_link_libraries(GadgetronClient PRIVATE Boost::headers)
target_link_libraries(GadgetronClient PRIVATE Boost::program_options)
target_link_libraries(GadgetronClient PUBLIC pulseg_recon_reader)

target_include_directories(GadgetronClient PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/mrdserver>)

install(TARGETS GadgetronClient EXPORT mrdserverTargets
    ARCHIVE DESTINATION lib)
install(FILES GadgetronClient.h gadgetron_ismrmrd_client.h NHLBICompression.h cpuisa.h
    DESTINATION include/mrdserver)
