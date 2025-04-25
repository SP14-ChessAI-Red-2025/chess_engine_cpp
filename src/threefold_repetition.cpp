// src/threefold_repetition.cpp
#include "chess_cpp/threefold_repetition.hpp"
#include "chess_cpp/chess_rules.hpp" // Needs access to move_type, piece_type enums

namespace chess {

    // Adds the current board state to the history count.
    // Updates the draw_allowed flag if the position has occurred 3 or more times.
    void previous_board_states::add_board_state(const board_state& current_board_state) {
        // Use the custom hasher and equality defined for the unordered_map
        int& count = position_counts[current_board_state];
        count++;

        // Check if draw *can* be claimed (position repeated >= 3 times)
        // Note: Doesn't automatically set game status to draw, just enables the claim.
        if (count >= 3) {
            draw_allowed = true;
        }
        // If count drops below 3 later (e.g., due to clearing history), draw_allowed remains true
        // until an irreversible move resets it via clear_history_on_irreversible_move.
    }

    // Clears the history if an irreversible move (pawn move or capture) occurred.
    void previous_board_states::clear_history_on_irreversible_move(const chess_move& move, const piece& moved_piece) {
        // Check if the move is a pawn move OR any type of capture (including en passant).
        // Promotions are implicitly pawn moves. Castling resets rights, but doesn't clear history here.
        if (moved_piece.type == piece_type::pawn ||
            move.type == move_type::capture ||
            move.type == move_type::en_passant) // Note: promotion check redundant if pawn check included
        {
            // Irreversible move occurred, reset the history and the draw claim possibility.
            position_counts.clear();
            draw_allowed = false;
            // Important: The state *after* this irreversible move needs to be added
            // back to the history by calling add_board_state separately.
        }
    }

} // namespace chess