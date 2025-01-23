add_library(chess_cpp SHARED chess_cpp.cpp ../include/chess_cpp/chess_cpp.hpp)

add_executable(chess_cpp_main chess_cpp_main.cpp)

target_link_libraries(chess_cpp_main PRIVATE chess_cpp)

if(MSVC)
    target_compile_definitions(chess_cpp PRIVATE NOMINMAX)
    target_compile_options(chess_cpp PRIVATE "/W4")
    target_compile_options(chess_cpp PRIVATE "/Zc:__cplusplus") # Use the correct value for the __cplusplus macro
else()
    target_compile_options(chess_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_compile_features(chess_cpp PUBLIC c_std_11 cxx_std_17)

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
