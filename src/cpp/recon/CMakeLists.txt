# The recon-side sequence reader: SequenceCache populated straight from the
# Pulseq seqfile chain (sequence_file_reader) plus its format-agnostic
# consumers (sequence_cache). This target is intentionally separate from the
# GadgetronClient build so recon consumers can link it alone.

add_library(sequence_reader STATIC
    sequence_cache.h
    sequence_cache.cpp
    sequence_file_reader.h
    sequence_file_reader.cpp)

target_compile_features(sequence_reader PUBLIC cxx_std_17)

target_include_directories(sequence_reader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(sequence_reader PUBLIC pulseqpp ismrmrd)

install(FILES sequence_cache.h sequence_file_reader.h
        DESTINATION include/mrdserver)

# ------------------------------------------------------------------
# GadgetronClient: the vendored Gadgetron ISMRMRD network client and its
# NHLBI compression.  Kept apart from sequence_reader above because the two
# answer different needs -- a recon that only wants the sequence model links
# that one alone, and nothing here is required to read a seqfile.
# ------------------------------------------------------------------

add_library(GadgetronClient STATIC
    GadgetronClient.h
    gadgetron_ismrmrd_client.h
    gadgetron_ismrmrd_client.cpp
    CompressedFloatBuffer.cpp
    CompressedFloatBufferAvx2.cpp
    CompressedFloatBufferSse41.cpp
    NHLBICompression.h
    cpuisa.h
    cpuisa.cpp)

target_compile_features(GadgetronClient PUBLIC cxx_std_17)

# The AVX2 and SSE4.1 buffers select themselves at run time through cpuisa,
# so each is compiled for the instruction set it names and nothing else is.
set_source_files_properties(CompressedFloatBufferAvx2.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
set_source_files_properties(CompressedFloatBufferSse41.cpp
    PROPERTIES COMPILE_OPTIONS "-msse4.1")

target_link_libraries(GadgetronClient PRIVATE Boost::headers Boost::program_options)
target_link_libraries(GadgetronClient PUBLIC sequence_reader)

target_include_directories(GadgetronClient PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})

install(FILES GadgetronClient.h gadgetron_ismrmrd_client.h NHLBICompression.h cpuisa.h
        DESTINATION include/mrdserver)
