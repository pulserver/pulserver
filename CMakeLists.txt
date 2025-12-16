cmake_minimum_required(VERSION 3.18)
project(pulserver_extensions LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Development.Module REQUIRED)

set(PYTHON_EXT_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/python/pulserver/tools/_design")

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
        # If pattern contains '*', treat as glob
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

    # Create the pybind11 module
    pybind11_add_module(${module_name} ${MODULE_SOURCES_LIST})

    # Include directories
    target_include_directories(${module_name} PRIVATE ${ARG_INCLUDES})

    # Output directory
    set_target_properties(${module_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${PYTHON_EXT_OUTPUT_DIR}/${output_subdir}"
        OUTPUT_NAME "${module_name}"
    )
endfunction()

# -------------------------
# _cpd module
# -------------------------
add_pybind_module(_cpd_wrapper "" 
    SOURCES
        "${CMAKE_SOURCE_DIR}/python/pulserver/tools/_design/_cpd_wrapper.cpp"
        "${CMAKE_SOURCE_DIR}/external/cpdlib/misc.c"
        "${CMAKE_SOURCE_DIR}/external/cpdlib/udcpd.c"
        "${CMAKE_SOURCE_DIR}/external/cpdlib/vdcpd.c"
    INCLUDES
        ${CMAKE_SOURCE_DIR}/external
)

# -------------------------
# _vds module
# -------------------------
add_pybind_module(_vds_wrapper "_grad"
    SOURCES
        "${CMAKE_SOURCE_DIR}/python/pulserver/tools/_design/_grad/_vds_wrapper.cpp"
        "${CMAKE_SOURCE_DIR}/external/vds/vds.c"
    INCLUDES
        ${CMAKE_SOURCE_DIR}/external
        ${CMAKE_SOURCE_DIR}/python/pulserver/tools/_design/_grad
)