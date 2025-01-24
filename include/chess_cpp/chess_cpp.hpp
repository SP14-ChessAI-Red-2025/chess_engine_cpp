#pragma once

#include "chess_cpp/version.hpp"

#include <cstddef>
#include <cstdint>
namespace chess {
enum class piece {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5
};

struct board_position {
    std::uint8_t rank;
    std::uint8_t file;
};

struct chess_move {
    board_position start_position;

    board_position target_position;
};

struct board_state {
    piece pieces[8][8];
};

extern "C" chess_move* get_valid_moves(board_state board_state, std::size_t* num_moves);
extern "C" void free_moves(chess_move* moves);