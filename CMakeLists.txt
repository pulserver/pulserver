cmake_minimum_required(VERSION 3.18)
project(pulserver_extensions LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Development.Module REQUIRED)

pybind11_add_module(_cpd
    python/pulserver/tools/_design/_cpd.cpp
    external/cpdlib/misc.c
    external/cpdlib/udcpd.c
    external/cpdlib/vdcpd.c
)

target_include_directories(_cpd PRIVATE
    ${CMAKE_SOURCE_DIR}/external
)

set_target_properties(_cpd PROPERTIES
    OUTPUT_NAME "_cpd"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/pulserver/tools/_design"
)
