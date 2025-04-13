cmake_minimum_required(VERSION 3.25.0)
project(chess_cpp VERSION 0.0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Enforce C++20

add_subdirectory(src)

# Conditionally add tests only if the "test" feature is enabled in vcpkg
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND "test" IN_LIST VCPKG_MANIFEST_FEATURES)
    # Ensure GTest is found before adding the subdirectory if tests are enabled
    find_package(GTest CONFIG REQUIRED)
    if(GTest_FOUND)
        enable_testing() # Enable CTest support
        add_subdirectory(tests)
    else()
        message(WARNING "GTest not found, tests subdirectory will be skipped.")
    endif()
endif()