cmake_minimum_required(VERSION 3.18)
project(pulserver_extensions LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Development.Module REQUIRED)

# The standalone C++ Pulseq library.  Built before the extension modules
# because _pulseqpp_wrapper links it.
add_subdirectory(${CMAKE_SOURCE_DIR}/cxx/pulseq ${CMAKE_BINARY_DIR}/pulseqpp_build)

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

# _pulseg_wrapper: pulserver.analysis C extension (needs full pulseg + seqdesc + cache)
set(PULSEG_C_SOURCES
    "${CMAKE_SOURCE_DIR}/csrc/src/pulseq/pulseq_parse.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/pulseq/pulseq_binary.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/pulseq/pulseq_names.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/core/pulseg_error.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/core/pulseg_math.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/io/pulseg_opts.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/structure/pulseg_dedup.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/structure/pulseg_structure.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/core/pulseg_core.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/cache/pulseg_cache.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/waveforms/pulseg_waveforms.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/safety/pulseg_safety.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/safety/pulseg_pns_memo.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/structure/pulseg_getters.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/freqmod/pulseg_freqmod.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/io/pulseg_protocol.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/cache/pulseg_trajectory.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/structure/pulseg_seqdesc.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/cache/pulseg_cache_seqdesc.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/bridge/pulseg_bridge.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/vendor/external_kiss_fft.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/vendor/external_kiss_fftr.c"
    # The same kissfft a second time, in double precision, for the mechanical
    # resonance chirp-z. See csrc/CMakeLists.txt, which has always had it.
    "${CMAKE_SOURCE_DIR}/csrc/src/vendor/external_kiss_fft_double.c"
    "${CMAKE_SOURCE_DIR}/csrc/src/vendor/external_md5.c"
)

add_pybind_module(_pulseg_wrapper ""
    SOURCES
        "${CMAKE_SOURCE_DIR}/python/pulserver/_ext/_pulseg_wrapper.cpp"
        ${PULSEG_C_SOURCES}
    INCLUDES
        ${CMAKE_SOURCE_DIR}/csrc/include
        ${CMAKE_SOURCE_DIR}/csrc/include/pulseg
        ${CMAKE_SOURCE_DIR}/csrc/include/pulseq
        ${CMAKE_SOURCE_DIR}/csrc/src
        ${CMAKE_SOURCE_DIR}/csrc/src/pulseq
        ${CMAKE_SOURCE_DIR}/csrc/src/vendor
        ${CMAKE_SOURCE_DIR}/cxx/include/pulseg
)

file(GLOB PULSEG_WRAPPERS
    "${CMAKE_CURRENT_BINARY_DIR}/python/pulserver/_ext/_pulseg_wrapper*${CMAKE_SHARED_LIBRARY_SUFFIX}")
install(FILES ${PULSEG_WRAPPERS}
    DESTINATION ${CMAKE_SOURCE_DIR}/python/pulserver/_ext)

# _pulseqpp_wrapper: the binding for pulseq::Sequence (cxx/pulseq). Thin --
# the public API, its docstrings and its type hints are Python-side, in
# pulserver/pypulseq/_sequence.py.
add_pybind_module(_pulseqpp_wrapper ""
    SOURCES
        "${CMAKE_SOURCE_DIR}/python/pulserver/_ext/_pulseqpp_wrapper.cpp"
        "${CMAKE_SOURCE_DIR}/python/pulserver/_ext/_pulseqpp_eventtypes.cpp"
    INCLUDES
        ${CMAKE_SOURCE_DIR}/cxx/include
)
target_link_libraries(_pulseqpp_wrapper PRIVATE pulseqpp)

file(GLOB PULSEQPP_WRAPPERS
    "${CMAKE_CURRENT_BINARY_DIR}/python/pulserver/_ext/_pulseqpp_wrapper*${CMAKE_SHARED_LIBRARY_SUFFIX}")
install(FILES ${PULSEQPP_WRAPPERS}
    DESTINATION ${CMAKE_SOURCE_DIR}/python/pulserver/_ext)

# _arbgrad_wrapper: pulserver.arbgrad C++ extension (vendored MRArbGrad core;
# see cxx/arbgrad/vendor/mrarbgrad/NOTICE.md)
add_pybind_module(_arbgrad_wrapper ""
    SOURCES
        "${CMAKE_SOURCE_DIR}/python/pulserver/_ext/_arbgrad_wrapper.cpp"
        "${CMAKE_SOURCE_DIR}/cxx/arbgrad/vendor/mrarbgrad/mag/Mag.cpp"
        "${CMAKE_SOURCE_DIR}/cxx/arbgrad/vendor/mrarbgrad/utility/global.cpp"
        "${CMAKE_SOURCE_DIR}/cxx/arbgrad/vendor/mrarbgrad/utility/v3.cpp"
    INCLUDES
        ${CMAKE_SOURCE_DIR}/cxx/arbgrad/vendor/mrarbgrad
)

file(GLOB ARBGRAD_WRAPPERS
    "${CMAKE_CURRENT_BINARY_DIR}/python/pulserver/_ext/_arbgrad_wrapper*${CMAKE_SHARED_LIBRARY_SUFFIX}")
install(FILES ${ARBGRAD_WRAPPERS}
    DESTINATION ${CMAKE_SOURCE_DIR}/python/pulserver/_ext)

# _sampling_wrapper: deterministic variable-density sampling kernels.
add_pybind_module(_sampling_wrapper ""
    SOURCES
        "${CMAKE_SOURCE_DIR}/python/pulserver/_ext/_sampling_wrapper.cpp"
)

file(GLOB SAMPLING_WRAPPERS
    "${CMAKE_CURRENT_BINARY_DIR}/python/pulserver/_ext/_sampling_wrapper*${CMAKE_SHARED_LIBRARY_SUFFIX}")
install(FILES ${SAMPLING_WRAPPERS}
    DESTINATION ${CMAKE_SOURCE_DIR}/python/pulserver/_ext)

# _fastseq_wrapper: the fastseq block-writing loop plus its rounding and row
# deduplication kernels. Optional at import time -- `fastseq.modules` falls back
# to its pure-Python reference implementations when this is not present, so a
# source checkout stays usable without a build.
add_pybind_module(_fastseq_wrapper ""
    SOURCES
        "${CMAKE_SOURCE_DIR}/fastseq/_fastseq_wrapper.cpp"
)

file(GLOB FASTSEQ_WRAPPERS
    "${CMAKE_CURRENT_BINARY_DIR}/fastseq/_fastseq_wrapper*${CMAKE_SHARED_LIBRARY_SUFFIX}")
install(FILES ${FASTSEQ_WRAPPERS}
    DESTINATION ${CMAKE_SOURCE_DIR}/fastseq)

option(PULSERVER_BUILD_CPP_TESTS "Build lightweight C++ unit tests" OFF)
if(PULSERVER_BUILD_CPP_TESTS)
    enable_testing()
    add_subdirectory(${CMAKE_SOURCE_DIR}/tests/cpptests ${CMAKE_BINARY_DIR}/cpptests_build)
endif()
