find_package(GTest CONFIG REQUIRED)

add_executable(chess_cpp_test chess_cpp_test.cpp)

target_link_libraries(chess_cpp_test PRIVATE chess_cpp GTest::gtest_main)

include(CTest)
include(GoogleTest)
gtest_discover_tests(chess_cpp_test)
