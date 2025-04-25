// include/chess_cpp/chess_ai.hpp
#pragma once // Use pragma once for include guard

#include "chess_rules.hpp" // Use the definitions from chess_rules.hpp
#include "nnue_evaluator.hpp"
#include "threefold_repetition.hpp" // Needed for previous_board_states, board_state_hasher, board_state_equal

#include <vector>
#include <memory>
#include <optional>
#include <cstdint>
#include <string>
#include <limits>
#include <map> // Add this to use std::map
#include <unordered_map> // For transposition table
#include <mutex>         // For transposition table mutex
#include <atomic>        // For cancellation flag
#include <stdexcept>     // For std::runtime_error

// Define the namespace for AI-related components
namespace chess::ai {

    // --- Constants ---
    // Define piece values for move ordering or simple evaluation fallback
    // Using const ensures these are compile-time constants
    const std::map<piece_type, int> piece_values = {
        {piece_type::pawn, 100},
        {piece_type::knight, 320},
        {piece_type::bishop, 330},
        {piece_type::rook, 500},
        {piece_type::queen, 900},
        {piece_type::king, 20000}, // High value for king safety considerations
        {piece_type::none, 0}
    };

    // --- Transposition Table Entry ---
    // Stores information about previously evaluated board states
    struct transposition_table_entry {
        enum class node_type { exact, lower_bound, upper_bound }; // Add enum
        double score = 0.0;
        std::size_t depth = 0;
        node_type type = node_type::exact; // Add type member
        // Potentially add best_move field here later for PV extraction
    };
    // --- Game Tree Node (Conceptual) ---
    // Represents a node in the search, used transiently during recursion
    struct game_tree {
        chess::board_state current_state; // The board state this node represents
        // Removed score, move, children as they are not strictly needed for the recursive function itself

        // Constructor
        game_tree(const chess::board_state& state);

        /**
         * @brief Performs the recursive Alpha-Beta search step.
         * @param depth Remaining search depth.
         * @param evaluator Reference to the evaluator (NNUE or basic).
         * @param alpha Best score maximizer can guarantee so far.
         * @param beta Best score minimizer can guarantee so far.
         * @param history Mutable history reference for checking repetitions and applying moves.
         * @return The evaluated score of this node/state from the perspective of the current player to move.
         * @throws std::runtime_error if the search is cancelled.
         */
        double deepen(std::size_t depth, NNUEEvaluator& evaluator, double alpha, double beta, previous_board_states& history);
    };

    // --- AI State ---
    // Encapsulates the AI's persistent state, primarily the evaluator
    struct chess_ai_state {
        NNUEEvaluator evaluator_; // The evaluator (NNUE or potentially basic)

        // Constructor: Initializes the evaluator from a model path
        explicit chess_ai_state(const std::string& model_path);

        /**
         * @brief Calculates and applies the AI's best move to the given board state.
         * @param board The current board state (will be modified by applying the chosen move).
         * @param difficulty Controls the search depth or time limit (interpretation depends on implementation).
         * @param history The game history object (will be modified by applying the chosen move).
         */
        void make_move(board_state& board, std::int32_t difficulty, previous_board_states& history);
    };

    // --- Helper Function Declarations ---
    // Determines if a move should be considered during search (e.g., ignore resign/draw)
    bool should_consider_move(const chess_move& move);

    // Checks if a specific square is attacked by the given player
    bool is_square_attacked(const board_state& board, const board_position& square, player attacking_player);

    // Provides a heuristic score for a move, used for move ordering
    int score_move(const board_state& state, const chess_move& move);

    // --- Transposition Table & Cancellation Flag (Declared as extern, defined in .cpp) ---
    // Using global variables for simplicity, could be encapsulated later if needed.
    // The unordered_map uses the board_state itself as the key, requiring hasher and equality predicates.
    extern std::unordered_map<chess::board_state, transposition_table_entry, chess::board_state_hasher, chess::board_state_equal> transposition_table;
    extern std::mutex transposition_table_mutex; // Mutex to protect concurrent access to the TT
    extern std::atomic<bool> search_cancelled;   // Atomic flag to signal search cancellation

    // Function to signal cancellation (definition in .cpp)
    void cancel_ai_search();

} // namespace chess::ai

// REMOVED the stray #endif that was here
