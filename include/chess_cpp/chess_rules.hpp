#pragma once

#include <cstddef>
#include <cstdint>

#include <vector>

// Use existing value of DLLEXPORT if already defined
#ifndef DLLEXPORT
    #ifdef _MSC_VER
        // Windows needs __declspec(dllexport) to make a symbol available from a dll
        #define DLLEXPORT __declspec(dllexport)
    #else
        #define DLLEXPORT
    #endif
#endif


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
    promotion = 4,
    claim_draw = 5, // Used to claim a draw under the 50 move rule
    resign = 6
};

enum class game_status : int {
    normal = 0,
    draw = 1,
    checkmate = 2,
    resigned = 3
};

struct board_position {
    std::uint8_t rank;
    std::uint8_t file;

    bool operator==(const board_position& rhs) const = default;
};

struct piece {
    piece_type type;

    player piece_player;

    bool operator==(const piece& rhs) const = default;
};

struct chess_move {
    move_type type;

    // if type == castle, this refers to the location of the rook
    board_position start_position;

    // Meaningless if type == castle
    // If type == en_passant, this refers to the position where the capturing pawn will move to
    // The captured pawn's position has the same file, but with a rank 1 closer to the center
    board_position target_position;

    // Only meaningful if type == promotion
    piece_type promotion_target;

    bool operator==(const chess_move& rhs) const = default;
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

    // Status of the game: normal, a draw, or a checkmate
    // If status == checkmate or status == resigned, then current_player is the loser
    game_status status;

    bool can_claim_draw;

    DLLEXPORT static board_state initial_board_state() noexcept;
};


DLLEXPORT std::vector<chess_move> get_valid_moves(const board_state& board_state);

DLLEXPORT board_state apply_move(board_state board, chess_move move);
}