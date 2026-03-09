cmake_minimum_required(VERSION 3.16)
project(pulserver_cpp_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

add_executable(test_reader_wait_state
    test_reader_wait_state.cpp
    ../../extensions/gadgetronlib/reader_wait_state.cpp
)
target_include_directories(test_reader_wait_state PRIVATE ../../extensions/gadgetronlib)
target_link_libraries(test_reader_wait_state PRIVATE pthread)
add_test(NAME test_reader_wait_state COMMAND test_reader_wait_state)

add_executable(test_ismrmrd_user_parameter_overrides
    test_ismrmrd_user_parameter_overrides.cpp
    ../../extensions/ge2mrdlib/ismrmrd_user_parameter_overrides.cpp
)
target_include_directories(test_ismrmrd_user_parameter_overrides PRIVATE ../../extensions/ge2mrdlib)
add_test(NAME test_ismrmrd_user_parameter_overrides COMMAND test_ismrmrd_user_parameter_overrides)
