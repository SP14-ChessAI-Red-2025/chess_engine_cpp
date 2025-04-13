find_package(onnxruntime REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

# Define the static library 'chess_cpp'
add_library(chess_cpp STATIC
    chess_rules.cpp
    chess_ai.cpp
    # minimax.cpp # Incompatible file removed
    heuristic_evaluator.cpp # Still included, but content is commented out
    nnue_evaluator.cpp
    threefold_repetition.cpp
)

# --- Add this line ---
# Ensure the static library is compiled with Position Independent Code (-fPIC)
# so it can be linked into the shared library (Python module) below.
set_target_properties(chess_cpp PROPERTIES POSITION_INDEPENDENT_CODE ON)
# --- End Add ---


# Define the Python module using pybind11
pybind11_add_module(chess_cpp_pybind SHARED python_api.cpp)

# --- Add include directories for the pybind target ---
target_include_directories(chess_cpp_pybind PRIVATE
    "../include/chess_cpp"
)
target_include_directories(chess_cpp_pybind PRIVATE
    "/home/bailey/Documents/onnxruntime-linux-x64-gpu-1.21.0/include"
)
# --- End Pybind Includes ---


# Add specific compile options (architecture-specific optimizations)
target_compile_options(chess_cpp PRIVATE -mbmi -mbmi2 -mpopcnt)

# Add platform-specific compile options/definitions
if(MSVC)
    target_compile_definitions(chess_cpp PRIVATE NOMINMAX)
    target_compile_options(chess_cpp PRIVATE "/W4")
    target_compile_options(chess_cpp PRIVATE "/Zc:__cplusplus")
else()
    target_compile_options(chess_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Specify C++ standard requirement
target_compile_features(chess_cpp PUBLIC c_std_11 cxx_std_20)

# --- Include Directories for chess_cpp library ---
target_include_directories(chess_cpp PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>" # For source tree builds
    "$<INSTALL_INTERFACE:include>" # For installed package
)
target_include_directories(chess_cpp PRIVATE
    "../include/chess_cpp" # Allows #include <chess_rules.hpp> etc. directly
    "/home/bailey/Documents/onnxruntime-linux-x64-gpu-1.21.0/include" # ONNX Runtime include path
)
target_include_directories(chess_cpp PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}/include # For generated version.hpp
)


# --- Executable ---
add_executable(chess_cpp_main chess_cpp_main.cpp)


# --- Version Header ---
configure_file(../include/chess_cpp/version.hpp.in include/chess_cpp/version.hpp @ONLY)


# --- Linking ---
target_link_libraries(chess_cpp PRIVATE onnxruntime::onnxruntime)
target_link_libraries(chess_cpp_main PRIVATE chess_cpp)
# Link the Python module against the chess library, pybind11, and Threads
target_link_libraries(chess_cpp_pybind PRIVATE chess_cpp pybind11::module Threads::Threads)


# --- Debug Flags ---
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(chess_cpp PRIVATE DEBUG)
endif()

