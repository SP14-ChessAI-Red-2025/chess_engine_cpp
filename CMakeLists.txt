cmake_minimum_required(VERSION 3.25.0)

project(chess_cpp VERSION 0.0.0.1 LANGUAGES CXX)

add_subdirectory(src)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND "test" IN_LIST VCPKG_MANIFEST_FEATURES)
    add_subdirectory(tests)
endif()
