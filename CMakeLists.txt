cmake_minimum_required(VERSION 3.18)
project(pulserver_cpd LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Development.Module REQUIRED)

pybind11_add_module(_cpd
    python/pulserver/design/_sampling/_cpd.cpp
    csrc/cpdlib/misc.c
    csrc/cpdlib/udcpd.c
    csrc/cpdlib/vdcpd.c
)

target_include_directories(_cpd PRIVATE
    ${CMAKE_SOURCE_DIR}/csrc
)

set_target_properties(_cpd PROPERTIES
    OUTPUT_NAME "_cpd"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/pulserver/design/_sampling"
)