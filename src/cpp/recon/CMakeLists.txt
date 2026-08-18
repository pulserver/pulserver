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
