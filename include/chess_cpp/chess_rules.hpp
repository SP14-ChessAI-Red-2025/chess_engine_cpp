#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace chess {
enum class piece_type : int {
    none = 0,
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
    claim_draw = 5,
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
    move_type type{}; // Use {} for default initialization
    board_position start_position{};
    board_position target_position{};
    piece_type promotion_target = piece_type::none; // Default initialize
    bool operator==(const chess_move& rhs) const = default;
};

struct board_state {
    piece pieces[8][8]{}; // Default initialize array
    bool can_castle[4]{};
    bool in_check[2]{};
    bool en_passant_valid[16]{};
    int turns_since_last_capture_or_pawn = 0;
    player current_player = player::white;
    game_status status = game_status::normal;
    bool can_claim_draw = false;

    // (DLLEXPORT likely not needed if only called via python_api or internally)
    static board_state initial_board_state() noexcept;
};

// Functions typically called via python_api or internally
std::vector<chess_move> get_valid_moves(const board_state& board_state);
board_state apply_move(board_state board, chess_move move);

} // namespace chess