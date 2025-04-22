add_library(chess_cpp SHARED
        chess_rules.cpp ../include/chess_cpp/chess_rules.hpp
        python_api.cpp ../include/chess_cpp/python_api.hpp
        chess_ai.cpp ../include/chess_cpp/chess_ai.hpp
        threefold_repetition.cpp
)

if(NNUE_ENABLED)
    target_sources(chess_cpp PRIVATE
            nnue_evaluator.cpp ../include/chess_cpp/nnue_evaluator.hpp
    )

    if(NOT DEFINED ONNX_LIBRARY_PATH OR NOT DEFINED ONNX_INCLUDE_PATH)
        message(FATAL_ERROR "Must specify ONNX_LIBRARY_PATH and ONNX_INCLUDE_PATH")
    endif()

    target_link_directories(chess_cpp PRIVATE ${ONNX_LIBRARY_PATH})
    target_link_libraries(chess_cpp PRIVATE onnxruntime)

    target_compile_definitions(chess_cpp PUBLIC NNUE_ENABLED NNUE_DISABLE_CUDA)

    if(NNUE_DISABLE_CUDA)
        target_compile_definitions(chess_cpp PRIVATE NNUE_DISABLE_CUDA)
    endif()

    target_include_directories(chess_cpp PRIVATE ${ONNX_INCLUDE_PATH})
endif()

add_executable(chess_cpp_main chess_cpp_main.cpp)

target_link_libraries(chess_cpp_main PRIVATE chess_cpp)

include(CheckIPOSupported)
check_ipo_supported(RESULT result)
if(result AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_target_properties(chess_cpp PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_target_properties(chess_cpp_main PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

if(MSVC)
    target_compile_definitions(chess_cpp PRIVATE NOMINMAX)
    target_compile_options(chess_cpp PRIVATE "/W4")
    target_compile_options(chess_cpp PRIVATE "/Zc:__cplusplus") # Use the correct value for the __cplusplus macro
else()
    target_compile_options(chess_cpp PRIVATE -Wall -Wextra -Wpedantic)

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        # Enable extra runtime checks
        target_compile_definitions(chess_cpp PRIVATE _GLIBCXX_ASSERTIONS _LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE)
        target_compile_definitions(chess_cpp_main PRIVATE _GLIBCXX_ASSERTIONS _LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE)
    endif()
endif()

target_compile_features(chess_cpp PUBLIC c_std_11 cxx_std_20)

# Projects linking to this library need to explicitly specify the subfolder
# That isn't necessary within the project, though
target_include_directories(chess_cpp PUBLIC "../include")
target_include_directories(chess_cpp PRIVATE "../include/chess_cpp")

configure_file(../include/chess_cpp/version.hpp.in include/chess_cpp/version.hpp)
target_include_directories(chess_cpp PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)

#Debug flags
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(chess_cpp PRIVATE DEBUG)
endif()
