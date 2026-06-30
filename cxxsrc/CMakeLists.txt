set(SOURCE_FILES
    GadgetronClient.h
    gadgetron_ismrmrd_client.h
    gadgetron_ismrmrd_client.cpp
    trajectory_cache_reader.h
    trajectory_cache_reader.cpp
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
target_link_libraries(GadgetronClient PRIVATE ismrmrd)

target_include_directories(GadgetronClient PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/mrdserver>)

install(TARGETS GadgetronClient EXPORT mrdserverTargets
    ARCHIVE DESTINATION lib)
install(FILES GadgetronClient.h gadgetron_ismrmrd_client.h trajectory_cache_reader.h NHLBICompression.h cpuisa.h
    DESTINATION include/mrdserver)
