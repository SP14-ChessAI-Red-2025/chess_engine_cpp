#pragma once

#include "chess_rules.hpp"
#include <unordered_set> // Correct header for unordered_multiset

namespace chess {

// Hash function for board_state
struct board_state_hasher {
    std::size_t operator()(const board_state& board) const;
};

// Equality comparison for board_state
struct board_state_equality {
    bool operator()(const board_state& board1, const board_state& board2) const;
};

// Structure to track previous board states for detecting threefold repetition
struct previous_board_states {
    std::unordered_multiset<board_state, board_state_hasher, board_state_equality> encountered_board_states;
    bool draw_allowed = false;

    void add_board_state(const board_state& board_state) {
        encountered_board_states.insert(board_state);
        if (encountered_board_states.count(board_state) >= 3) {
            draw_allowed = true;
        }
    }

    void clear_history_on_irreversible_move(const chess_move& move, const piece& moved_piece) {
        if (moved_piece.type == piece_type::pawn || move.type == move_type::capture || move.type == move_type::en_passant) {
            encountered_board_states.clear();
            draw_allowed = false;
        }
    }
};

} // namespace chess