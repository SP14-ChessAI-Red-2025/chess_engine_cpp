#include "chess_cpp.hpp"

#include <cstdlib>

chess_move* get_valid_moves(board_state board_state, std::size_t* num_moves) {
    std::size_t size = 10;

    auto* result = new chess_move[size];

    for(std::size_t i = 0; i < size; i++) {
        result[i] = {
            .start_position = {.rank = static_cast<std::uint8_t>(i), .file = 213},
            .target_position = {.rank = static_cast<std::uint8_t>(board_state.i), .file = 2}
        };
    }

    *num_moves = size;

    return result;
}

void free_moves(chess_move* moves) {
    delete[] moves;
}