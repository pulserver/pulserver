
set(SOURCE_FILES
    GadgetronClient.h
    gadgetron_ismrmrd_client.h
    gadgetron_ismrmrd_client.cpp
    reader_wait_state.cpp
    CompressedFloatBuffer.cpp
    CompressedFloatBufferAvx2.cpp
    CompressedFloatBufferSse41.cpp
    cpuisa.cpp)

add_library(GadgetronClient STATIC ${SOURCE_FILES})

target_compile_options(GadgetronClient PRIVATE -mavx2 -mfma)

target_link_libraries(GadgetronClient PRIVATE Boost::headers)
target_link_libraries(GadgetronClient PRIVATE Boost::program_options)
target_link_libraries(GadgetronClient PRIVATE ismrmrd)

