cmake_minimum_required(VERSION 3.18)
project(pulserver_extensions LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Development.Module REQUIRED)

function(add_pybind_module module_name output_subdir)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs SOURCES INCLUDES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(MODULE_SOURCES_LIST "")
    foreach(src_pattern IN LISTS ARG_SOURCES)
        if("${src_pattern}" MATCHES "\\*")
            file(GLOB EXPANDED ${src_pattern})
            if(EXPANDED STREQUAL "")
                message(FATAL_ERROR "Glob '${src_pattern}' matched no files")
            endif()
            list(APPEND MODULE_SOURCES_LIST ${EXPANDED})
        else()
            if(NOT EXISTS ${src_pattern})
                message(FATAL_ERROR "Source file '${src_pattern}' does not exist")
            endif()
            list(APPEND MODULE_SOURCES_LIST ${src_pattern})
        endif()
    endforeach()

    list(GET MODULE_SOURCES_LIST 0 FIRST_SOURCE)
    get_filename_component(MODULE_OUTPUT_DIR "${FIRST_SOURCE}" DIRECTORY)

    pybind11_add_module(${module_name} ${MODULE_SOURCES_LIST})
    target_include_directories(${module_name} PRIVATE ${ARG_INCLUDES})
    set_target_properties(${module_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${MODULE_OUTPUT_DIR}"
        OUTPUT_NAME "${module_name}"
    )
endfunction()

# _pulseqlib_wrapper: pge C extension (needs full pulseqlib + seqdesc + cache)
set(PULSEQLIB_C_SOURCES
    "${CMAKE_SOURCE_DIR}/csrc/src/core/pulseqlib_error.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/core/pulseqlib_math.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/io/pulseqlib_parse.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/structure/pulseqlib_dedup.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/structure/pulseqlib_structure.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/core/pulseqlib_core.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/cache/pulseqlib_cache.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/waveforms/pulseqlib_waveforms.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/safety/pulseqlib_safety.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/structure/pulseqlib_getters.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/freqmod/pulseqlib_freqmod.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/io/pulseqlib_protocol.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/waveforms/pulseqlib_trajectory.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/structure/pulseqlib_seqdesc.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/cache/pulseqlib_cache_seqdesc.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/core/pulseqlib_bridge.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/vendor/external_kiss_fft.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/vendor/external_kiss_fftr.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/vendor/external_md5.c"
)

add_pybind_module(_pulseqlib_wrapper ""
    SOURCES
        "${CMAKE_SOURCE_DIR}/python/pge/core/_extension/_pulseqlib_wrapper.cpp"
        ${PULSEQLIB_C_SOURCES}
    INCLUDES
        ${CMAKE_SOURCE_DIR}/csrc/include
        ${CMAKE_SOURCE_DIR}/csrc/src/vendor
        ${CMAKE_SOURCE_DIR}/extensions
)

file(GLOB PULSEQLIB_WRAPPERS
    "${CMAKE_CURRENT_BINARY_DIR}/python/pge/core/_extension/_pulseqlib_wrapper*${CMAKE_SHARED_LIBRARY_SUFFIX}")
install(FILES ${PULSEQLIB_WRAPPERS}
    DESTINATION ${CMAKE_SOURCE_DIR}/python/pge/core/_extension)

option(PULSERVER_BUILD_CPP_TESTS "Build lightweight C++ unit tests" OFF)
if(PULSERVER_BUILD_CPP_TESTS)
    enable_testing()
    add_subdirectory(${CMAKE_SOURCE_DIR}/tests/cpptests ${CMAKE_BINARY_DIR}/cpptests_build)
endif()
