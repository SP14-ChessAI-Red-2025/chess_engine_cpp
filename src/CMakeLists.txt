# Bailey build cmd so I don't lose.
# cmake -B build -S . -DNNUE_ENABLED=ON -DORT_INSTALL_DIR=$(pwd)/ort_install -DCMAKE_PREFIX_PATH=/home/bailey/Documents/onnxruntime-linux-x64-gpu-1.21.0 -DCMAKE_BUILD_TYPE=Release

# Define the core sources (excluding NNUE-specific ones initially)
set(CHESS_CPP_SOURCES
    chess_rules.cpp
    python_api.cpp
    chess_ai.cpp
    threefold_repetition.cpp
)

# Add the shared library target using the core sources
add_library(chess_cpp SHARED ${CHESS_CPP_SOURCES})

# --- Conditionally Add NNUE ---
# This definition controls whether NNUE code paths are active in C++
# It should be set via cmake -D command line, e.g., cmake -DNNUE_ENABLED=ON ..
option(NNUE_ENABLED "Enable NNUE evaluation" OFF) # Default to OFF unless specified

if(NNUE_ENABLED)
    message(STATUS "NNUE Evaluation Enabled.")

    # Define ORT_INSTALL_DIR - expects path passed via -DORT_INSTALL_DIR=... from build.sh
    if(NOT DEFINED ORT_INSTALL_DIR)
      set(ORT_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ort_install" CACHE PATH "Path to ONNX Runtime install directory")
      message(WARNING "ORT_INSTALL_DIR not passed via command line, using default: ${ORT_INSTALL_DIR}")
    endif()
    message(STATUS "Using ONNX Runtime from: ${ORT_INSTALL_DIR}")

    # --- Use paths from downloaded ORT ---
    # Add include directory needed for #include <onnxruntime_cxx_api.h>
    target_include_directories(chess_cpp PRIVATE ${ORT_INSTALL_DIR}/include)

    # Find the actual ONNX Runtime library file (.so) in the downloaded lib folder
    find_library(ONNXRUNTIME_LIBRARY NAMES libonnxruntime.so onnxruntime
                 PATHS ${ORT_INSTALL_DIR}/lib
                 REQUIRED) # REQUIRED makes CMake error out if not found
    message(STATUS "Found ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

    # Link the chess_cpp library against the found ONNX Runtime library file
    target_link_libraries(chess_cpp PRIVATE ${ONNXRUNTIME_LIBRARY})
    # --- End of needed additions for downloaded ORT ---


    # --- Conditionally add NNUE source file ---
    target_sources(chess_cpp PRIVATE nnue_evaluator.cpp) # Assuming path relative to CMakeLists.txt
    target_compile_definitions(chess_cpp PUBLIC NNUE_ENABLED)

else()
    message(STATUS "NNUE Evaluation Disabled.")
endif()
# --- End Conditional NNUE ---


# --- Executable ---
add_executable(chess_cpp_main chess_cpp_main.cpp)
# Link the main executable against the library (which may or may not include NNUE parts)
target_link_libraries(chess_cpp_main PRIVATE chess_cpp)


# --- Include Directories (Core) ---
# These are needed regardless of NNUE status
target_include_directories(chess_cpp PUBLIC
    # Directory containing headers within the source tree (e.g., ../include)
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>"
    # Directory where headers will be installed
    "$<INSTALL_INTERFACE:include>"
)


# --- Compile Options / Features ---

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

    # Add extra checks for Debug builds
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(chess_cpp PRIVATE _GLIBCXX_ASSERTIONS _LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE)
        target_compile_definitions(chess_cpp_main PRIVATE _GLIBCXX_ASSERTIONS _LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE)
        target_compile_definitions(chess_cpp PRIVATE DEBUG) # Add DEBUG definition
    endif()
endif()

# Set C++ standard requirement
target_compile_features(chess_cpp PUBLIC cxx_std_20)

# --- Version Header Generation ---
# Ensure the .in file exists in the specified location relative to this CMakeLists.txt
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/chess_cpp/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/chess_cpp/version.hpp" # Output to build dir
    @ONLY
)
# Add the build directory include path so the generated header can be found
target_include_directories(chess_cpp PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include")
target_include_directories(chess_cpp_main PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include")

