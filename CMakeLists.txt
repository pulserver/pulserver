cmake_minimum_required(VERSION 3.18)
project(pulserver_extensions LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Development.Module REQUIRED)

set(PYTHON_EXT_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/python/pulserver/tools/_design")

# -------------------------
# Helper function
# -------------------------
function(add_pybind_module module_name output_subdir)
    set(options)
    set(oneValueArgs SOURCES_GLOB)
    set(multiValueArgs INCLUDES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Collect sources using GLOB
    file(GLOB MODULE_SOURCES ${ARG_SOURCES_GLOB})

    # Safety check
    if(MODULE_SOURCES STREQUAL "")
        message(FATAL_ERROR "No sources found for module '${module_name}' matching: ${ARG_SOURCES_GLOB}")
    endif()

    # Create the pybind11 module
    pybind11_add_module(${module_name} ${MODULE_SOURCES})

    target_include_directories(${module_name} PRIVATE ${ARG_INCLUDES})

    # Output directory
    set_target_properties(${module_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${PYTHON_EXT_OUTPUT_DIR}/${output_subdir}"
        OUTPUT_NAME "${module_name}"
    )
endfunction()

# -------------------------
# _cpd module (automatic sources)
# -------------------------
add_pybind_module(_cpd "" 
    SOURCES_GLOB
        "${CMAKE_SOURCE_DIR}/python/pulserver/tools/_design/_cpd.cpp"
        "${CMAKE_SOURCE_DIR}/external/cpdlib/*.c"
    INCLUDES
        ${CMAKE_SOURCE_DIR}/external
)

# -------------------------
# _vds module
# -------------------------
add_pybind_module(_vds "_grad"
    SOURCES_GLOB
        "${CMAKE_SOURCE_DIR}/python/pulserver/tools/_design/_grad/_vds.cpp"
        "${CMAKE_SOURCE_DIR}/external/vds/*.c"
    INCLUDES
        ${CMAKE_SOURCE_DIR}/external
        ${CMAKE_SOURCE_DIR}/python/pulserver/tools/_design/_grad
)