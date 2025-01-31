#pragma once

#include "chess_cpp/version.hpp"

#include <cstddef>
#include <cstdint>

namespace chess {
enum class piece_type : int {
    none = 0, // Used to indicate that a square is empty
    pawn = 1,
    knight = 2,
    bishop = 3,
    rook = 4,
    queen = 5,
    king = 6
};

enum class player : int {
    white = 0,
    black = 1
};

enum class move_type : int {
    normal_move = 0,
    capture = 1,
    en_passant = 2,
    castle = 3,
    promotion = 4
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
    move_type type;

    board_position start_position;

    board_position target_position;

    piece_type promotion_target;
};

struct board_state {
    // pieces[0] is rank 1, and pieces[7] is rank 8
    // pieces[2][3] is rank 3, file D
    piece pieces[8][8];

    // Whether the players are allowed to castle
    // Kingside white castling is at index 0, queenside white castling at index 1
    // Kingside black castling is at index 2, queenside black castling at index 3
    bool can_castle[4];

    // Whether the kings are currently in check
    // White is at index 0, black at index 1
    bool in_check[2];

    // Whether a pawn is able to be captured via en passant
    // This is only true if the pawn has just moved 2 squares on the previous turn
    // Indices 0-7 represent the white pawns, on files A-H
    // Indices 8-15 represent the black pawns, on files A-H
    bool en_passant_valid[16];

    // Turns since a capture has been made or a pawn has benn moved
    // It this reaches 50, the game is a draw
    int turns_since_last_capture_or_pawn;

    player current_player;

    static board_state initial_board_state();
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
