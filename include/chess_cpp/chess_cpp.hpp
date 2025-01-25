#pragma once

#include "chess_cpp/version.hpp"

#include <cstddef>
#include <cstdint>

namespace chess {
enum class piece_type {
    None = 0, // Used to indicate that a square is empty
    Pawn = 1,
    Knight = 2,
    Bishop = 3,
    Rook = 4,
    Queen = 5,
    King = 6
};

enum class player {
    white = 0,
    black = 1
};

struct board_position {
    std::uint8_t rank;
    std::uint8_t file;
};

struct piece {
    piece_type type;

    player player;
};

struct chess_move {
    board_position start_position;

    board_position target_position;
};

struct board_state {
    piece pieces[8][8] = {};

    // Whether the players have castled yet
    // White is at index 0, black at index 1
    bool has_castled[2] = {};

    // Whether the kings are currently in check
    bool in_check[2] = {};

    player current_player = player::white;
};

// Functions in this namespace will be called from Python
namespace python {
// Returns an array of valid moves for a given board state
// Writes the number of valid moves to *num_moves
// The returned array must be freed with free_moves
extern "C" chess_move* get_valid_moves(board_state board_state, std::size_t* num_moves);

// Free a list of moves allocated by get_valid_moves
extern "C" void free_moves(chess_move* moves);

extern "C" board_state get_initial_board_state();
}
}
