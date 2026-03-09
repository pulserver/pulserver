cmake_minimum_required(VERSION 3.18)
project(pulserver_extensions LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Development.Module REQUIRED)

# -------------------------
# Reliable pybind11 module helper
# -------------------------
function(add_pybind_module module_name output_subdir)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs SOURCES INCLUDES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Expand sources: support both explicit files and globs
    set(MODULE_SOURCES_LIST "")
    foreach(src_pattern IN LISTS ARG_SOURCES)
        if("${src_pattern}" MATCHES "\\*")
            file(GLOB EXPANDED ${src_pattern})
            if(EXPANDED STREQUAL "")
                message(FATAL_ERROR "Glob '${src_pattern}' for module '${module_name}' matched no files")
            endif()
            list(APPEND MODULE_SOURCES_LIST ${EXPANDED})
        else()
            if(NOT EXISTS ${src_pattern})
                message(FATAL_ERROR "Source file '${src_pattern}' does not exist for module '${module_name}'")
            endif()
            list(APPEND MODULE_SOURCES_LIST ${src_pattern})
        endif()
    endforeach()

    # Get the directory of the first source file
    list(GET MODULE_SOURCES_LIST 0 FIRST_SOURCE)
    get_filename_component(MODULE_OUTPUT_DIR "${FIRST_SOURCE}" DIRECTORY)

    pybind11_add_module(${module_name} ${MODULE_SOURCES_LIST})

    target_include_directories(${module_name} PRIVATE ${ARG_INCLUDES})

    # Output directory: use the directory of the first source file
    set_target_properties(${module_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${MODULE_OUTPUT_DIR}"
        OUTPUT_NAME "${module_name}"
    )
endfunction()

# -------------------------
# _cpd module
# -------------------------
# add_pybind_module(_cpd_wrapper "" 
#     SOURCES
#         "${CMAKE_SOURCE_DIR}/python/pulserver/pulseq/_extension/_cpd_wrapper.cpp"
#         "${CMAKE_SOURCE_DIR}/external/cpdlib/misc.c"
#         "${CMAKE_SOURCE_DIR}/external/cpdlib/udcpd.c"
#         "${CMAKE_SOURCE_DIR}/external/cpdlib/vdcpd.c"
#     INCLUDES
#         ${CMAKE_SOURCE_DIR}/external
# )

# # Install built extensions to the source tree for editable mode
# file(GLOB CPD_WRAPPERS "${CMAKE_CURRENT_BINARY_DIR}/python/pulserver/pulseq/_extension/_cpd_wrapper*${CMAKE_SHARED_LIBRARY_SUFFIX}")
# install(
#     FILES  ${CPD_WRAPPERS} DESTINATION ${CMAKE_SOURCE_DIR}/python/pulserver/pulseq/_extension
# )

# -------------------------
# _vds module
# -------------------------
# add_pybind_module(_vds_wrapper ""
#     SOURCES
#         "${CMAKE_SOURCE_DIR}/python/pulserver/pulseq/_extension/_vds_wrapper.cpp"
#         "${CMAKE_SOURCE_DIR}/external/vdslib/vds.c"
#     INCLUDES
#         ${CMAKE_SOURCE_DIR}/external
#         #${CMAKE_SOURCE_DIR}/python/pulserver/pulseq/_extension
# )

# # Install built extensions to the source tree for editable mode
# file(GLOB VDS_WRAPPERS "${CMAKE_CURRENT_BINARY_DIR}/python/pulserver/pulseq/_extension/_vds_wrapper*${CMAKE_SHARED_LIBRARY_SUFFIX}")
# install(
#     FILES  ${VDS_WRAPPERS} DESTINATION ${CMAKE_SOURCE_DIR}/python/pulserver/pulseq/_extension
# )

# -------------------------
# _pulseqlib module
# -------------------------
file(GLOB PULSEQLIB_C_SOURCES "${CMAKE_SOURCE_DIR}/csrc/*.c")

add_pybind_module(_pulseqlib_wrapper ""
    SOURCES
        "${CMAKE_SOURCE_DIR}/python/pulserver/core/_extension/_pulseqlib_wrapper.cpp"
        ${PULSEQLIB_C_SOURCES}
    INCLUDES
        ${CMAKE_SOURCE_DIR}/csrc
        ${CMAKE_SOURCE_DIR}/extensions
)

# Install built extensions to the source tree for editable mode
file (GLOB PULSEQLIB_WRAPPERS "${CMAKE_CURRENT_BINARY_DIR}/python/pulserver/core/_extension/_pulseqlib_wrapper*${CMAKE_SHARED_LIBRARY_SUFFIX}")
install(
    FILES  ${PULSEQLIB_WRAPPERS} DESTINATION ${CMAKE_SOURCE_DIR}/python/pulserver/core/_extension
)

option(PULSERVER_BUILD_CPP_TESTS "Build lightweight C++ unit tests" OFF)
if(PULSERVER_BUILD_CPP_TESTS)
    enable_testing()
    add_subdirectory(${CMAKE_SOURCE_DIR}/tests/cpptests ${CMAKE_BINARY_DIR}/cpptests_build)
endif()