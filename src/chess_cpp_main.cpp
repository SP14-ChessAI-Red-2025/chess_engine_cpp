#include <chess_cpp/python_api.hpp>

#include <iostream>
#include <format>


int main() {
    auto board = chess::python::get_initial_board_state();

    std::size_t num_moves = 0;

    auto moves = chess::python::get_valid_moves(board, &num_moves);

    std::cout << std::format("num_moves: {}", num_moves) << std::endl;

    chess::python::free_moves(moves);

}
