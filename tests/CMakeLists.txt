find_package(GTest CONFIG REQUIRED)

add_executable(chess_cpp_test chess_cpp_test.cpp)

target_link_libraries(chess_cpp_test PRIVATE chess_cpp GTest::gtest_main)

add_custom_command(TARGET chess_cpp_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:chess_cpp_test> $<TARGET_RUNTIME_DLLS:chess_cpp_test>
        COMMAND_EXPAND_LISTS
)

include(CTest)
include(GoogleTest)
gtest_discover_tests(chess_cpp_test)
