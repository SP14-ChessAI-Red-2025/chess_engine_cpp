#ifndef CHESS_CPP_THREEFOLD_REPETITION_HPP
#define CHESS_CPP_THREEFOLD_REPETITION_HPP

#include "chess_cpp/chess_rules.hpp" // Includes board_state, piece, chess_move definitions
#include <unordered_map>            // For storing position counts with hashing
#include <functional>               // For std::hash
#include <cstddef>                  // For std::size_t

namespace chess {

    // Helper function for combining hashes (boost::hash_combine pattern)
    // Placed here for use by board_state_hasher
    template <class T>
    inline void hash_combine(std::size_t& seed, const T& v) {
        // Simple hash combination function
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    // Complete hasher for board_state
    // Defined in the header to be available for the unordered_map declaration
    struct board_state_hasher {
        std::size_t operator()(const board_state& state) const {
            std::size_t seed = 0;

            // Hash pieces
            for (int r = 0; r < 8; ++r) {
                for (int f = 0; f < 8; ++f) {
                     // Combine hashes for piece type and player
                     hash_combine(seed, static_cast<int>(state.pieces[r][f].type));
                     hash_combine(seed, static_cast<int>(state.pieces[r][f].piece_player));
                }
            }

            // Hash current player
            hash_combine(seed, static_cast<int>(state.current_player));

            // Hash castling rights
            for (int i = 0; i < 4; ++i) {
                hash_combine(seed, state.can_castle[i]);
            }

            // Hash en passant validity
            // Ensure the size (16 here) matches your board_state definition
            for (int i = 0; i < 16; ++i) {
                hash_combine(seed, state.en_passant_valid[i]);
            }

            return seed;
        }
    };

    // Explicit equality predicate for board_state
    // Defined in the header to be available for the unordered_map declaration
    struct board_state_equal {
        bool operator()(const board_state& lhs, const board_state& rhs) const {
            // Call the globally defined operator== for board_state
            // Ensure operator==(lhs, rhs) is declared in chess_rules.hpp
            // and defined in chess_rules.cpp
            return lhs == rhs; // Relies on operator== being defined for board_state
        }
    };


    // Class to track board state history for threefold repetition detection
    class previous_board_states {
    public:
        // Use unordered_map with the custom hasher AND the custom equality predicate
        std::unordered_map<board_state, int, board_state_hasher, board_state_equal> position_counts;

        // Flag indicating if a draw can be claimed based on repetition
        bool draw_allowed = false;

        /**
         * @brief Adds the current board state to the history count.
         * Updates the draw_allowed flag if the position has occurred 3 or more times.
         * @param current_board_state The board state to record.
         */
        void add_board_state(const board_state& current_board_state);

        /**
         * @brief Clears the history if an irreversible move (pawn move or capture) occurred.
         * This is necessary because such moves reset the conditions for repetition.
         * @param move The move that was just made.
         * @param moved_piece The piece that was moved.
         */
        void clear_history_on_irreversible_move(const chess_move& move, const piece& moved_piece);
    };

} // namespace chess

#endif // CHESS_CPP_THREEFOLD_REPETITION_HPP
