find_package(onnxruntime REQUIRED)
find_package(pybind11 CONFIG REQUIRED) # Keep finding it, even if API uses Ctypes for now
find_package(Threads REQUIRED)

add_library(chess_cpp STATIC
    chess_rules.cpp
    chess_ai.cpp
    nnue_evaluator.cpp
    threefold_repetition.cpp
)

# Ensure static library is compiled with PIC for linking into shared lib
set_target_properties(chess_cpp PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Define the Python module (even if using Ctypes, keep target name consistent)
pybind11_add_module(chess_cpp_pybind SHARED python_api.cpp)

target_include_directories(chess_cpp_pybind PRIVATE
    "../include/chess_cpp"
    ${onnxruntime_INCLUDE_DIRS} # Use variable from find_package
)

# Add architecture-specific optimizations conditionally
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Applying BMI/POPCNT flags for GCC/Clang compiler.")
    target_compile_options(chess_cpp PRIVATE -mbmi -mbmi2 -mpopcnt)
endif()

# Add platform-specific compile options/definitions
if(MSVC)
    target_compile_definitions(chess_cpp PRIVATE NOMINMAX)
    target_compile_options(chess_cpp PRIVATE "/W4" "/Zc:__cplusplus")
else()
    target_compile_options(chess_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_compile_features(chess_cpp PUBLIC cxx_std_20)

target_include_directories(chess_cpp PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>" # For source tree builds
    "$<INSTALL_INTERFACE:include>" # For installed package
)
target_include_directories(chess_cpp PRIVATE
    "../include/chess_cpp"
    ${onnxruntime_INCLUDE_DIRS} # Use variable from find_package
)
target_include_directories(chess_cpp PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}/include # For generated version.hpp
)

add_executable(chess_cpp_main chess_cpp_main.cpp)

# --- Version Header ---
configure_file(../include/chess_cpp/version.hpp.in include/chess_cpp/version.hpp @ONLY)

# --- Linking ---
target_link_libraries(chess_cpp PRIVATE onnxruntime::onnxruntime)
target_link_libraries(chess_cpp_main PRIVATE chess_cpp)
# Link Python module against chess lib, onnxruntime, and Threads
target_link_libraries(chess_cpp_pybind PRIVATE chess_cpp onnxruntime::onnxruntime Threads::Threads)

# --- Debug Flags ---
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(chess_cpp PRIVATE DEBUG)
endif()