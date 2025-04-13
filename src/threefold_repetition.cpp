#include "chess_rules.hpp"

#include <unordered_set>
#include <cstddef>

namespace chess {
struct board_state_hasher {
    std::size_t operator()(const board_state& board) const {
        // TODO: Implement a proper hash function based on piece positions, castling rights, en passant target, and current player
        return 0; // Placeholder - This will make the set ineffective!
    }
};

struct board_state_equality {
    // Compares relevant parts of the board state for repetition detection
    bool operator()(const board_state& board1, const board_state& board2) const {
        for(std::uint8_t rank = 0; rank < 8; rank++) {
            for(std::uint8_t file = 0; file < 8; file++) {
                // Compare piece type and player
                if(board1.pieces[rank][file].type != board2.pieces[rank][file].type ||
                   board1.pieces[rank][file].piece_player != board2.pieces[rank][file].piece_player) {
                     return false;
                }
            }
        }
        // Compare castling rights
        for(int i=0; i<4; ++i) {
             if (board1.can_castle[i] != board2.can_castle[i]) return false;
        }
        // Compare en passant validity (crucial)
        for(int i=0; i<16; ++i) {
            if (board1.en_passant_valid[i] != board2.en_passant_valid[i]) return false;
        }

        // Compare player to move
        if (board1.current_player != board2.current_player) return false;

        // If all relevant parts match, the states are considered equal for repetition
        return true;
    }
};

// Structure to track previous board states for detecting threefold repetition
struct previous_board_states {
    std::unordered_multiset<board_state, board_state_hasher, board_state_equality> encountered_board_states;

    bool draw_allowed = false; // Flag if a draw can be claimed by repetition

    void add_board_state(const board_state& board_state) {
        encountered_board_states.insert(board_state);

        // Check if this state has now appeared 3 or more times
        if(encountered_board_states.count(board_state) >= 3) {
            draw_allowed = true;
        }
    }

    // Optional: Add a function to clear history if a pawn move or capture occurs
    void clear_history_on_irreversible_move(const chess_move& move, const piece& moved_piece) {
        if (moved_piece.type == piece_type::pawn ||
            move.type == move_type::capture ||
            move.type == move_type::en_passant) // En passant is a capture
        {
            encountered_board_states.clear();
            draw_allowed = false; // Reset draw claim status
        }
    }
};

} // namespace chess