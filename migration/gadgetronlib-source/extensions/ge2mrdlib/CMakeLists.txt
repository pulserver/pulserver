set(SOURCE_FILES
    GERawConverter.cpp
    GERawConverter.h
    ismrmrd_user_parameter_overrides.cpp
    ismrmrd_user_parameter_overrides.h
    GenericConverter.cpp
    GenericConverter.h
    SequenceConverter.h
    XMLWriter.h)

find_package(LibXml2 REQUIRED)
find_package(LibXslt REQUIRED)

add_library(GEToIsmrmrd STATIC ${SOURCE_FILES})

target_link_libraries(GEToIsmrmrd PRIVATE Boost::headers)
target_link_libraries(GEToIsmrmrd PRIVATE Boost::program_options)
target_link_libraries(GEToIsmrmrd PRIVATE LibXml2::LibXml2)
target_link_libraries(GEToIsmrmrd PRIVATE LibXslt::LibXslt)
target_link_libraries(GEToIsmrmrd PRIVATE ismrmrd)
target_link_libraries(GEToIsmrmrd PUBLIC ${OX_SDK_IMPORT_PREFIX}Acquisition)
target_link_libraries(GEToIsmrmrd PUBLIC ${OX_SDK_IMPORT_PREFIX}ProcessingControl)
target_link_libraries(GEToIsmrmrd PRIVATE ${OX_SDK_IMPORT_PREFIX}Common)
target_link_libraries(GEToIsmrmrd PRIVATE ${OX_SDK_IMPORT_PREFIX}Core)
target_link_libraries(GEToIsmrmrd PUBLIC ${OX_SDK_IMPORT_PREFIX}Dicom)
target_link_libraries(GEToIsmrmrd PUBLIC ${OX_SDK_IMPORT_PREFIX}Epi)
target_link_libraries(GEToIsmrmrd PRIVATE ${OX_SDK_IMPORT_PREFIX}Legacy)

