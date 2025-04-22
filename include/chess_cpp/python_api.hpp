#pragma once

#include "chess_rules.hpp"
// #include "chess_ai.hpp" // chess_rules.hpp likely includes necessary types, avoid circularity if possible
                           // Forward declare if needed, or ensure chess_ai.hpp doesn't include python_api.hpp

#include <cstddef> // For std::size_t
#include <cstdint> // For std::int32_t

// Forward declare AI state if possible to reduce header dependencies
namespace chess::ai { struct chess_ai_state; }


// Functions in this namespace will be called from Python (likely via ctypes)
namespace chess::python {

    /**
     * @brief Initialize the AI state, loading the NNUE model.
     * @param model_path Path to the .onnx model file.
     * @return Opaque pointer to the AI state, or nullptr on failure. Must be freed with free_ai_state.
     */
    extern "C" DLLEXPORT void* init_ai_state(const char* model_path) noexcept;

    /**
     * @brief Free resources associated with the AI state.
     * @param state Opaque pointer previously returned by init_ai_state.
     */
    extern "C" DLLEXPORT void free_ai_state(void* state) noexcept;

    /**
     * @brief Returns an array of valid moves for a given board state.
     * @param board_state The current state of the board.
     * @param num_moves Pointer to a size_t where the number of valid moves will be written.
     * @return Pointer to an array of chess_move objects, or nullptr on failure. The returned array must be freed with free_moves.
     */
    extern "C" DLLEXPORT chess_move* get_valid_moves(board_state board_state, std::size_t* num_moves) noexcept;

    /**
     * @brief Free a list of moves allocated by get_valid_moves.
     * @param moves Pointer previously returned by get_valid_moves.
     */
    extern "C" DLLEXPORT void free_moves(chess_move* moves) noexcept;

    /**
     * @brief Applies a move to the board_state, modifying it in place.
     * @param board_state Pointer to the board state to modify.
     * @param move The move to apply.
     */
    extern "C" DLLEXPORT void apply_move(board_state* board_state, chess_move move) noexcept;

    /**
     * @brief Have the AI calculate and make a move, modifying the board_state in place.
     * @param ai_state Opaque pointer to the AI state
     * @param board_state Pointer to the board state to modify.
     * @param difficulty Difficulty level (e.g., influencing search depth).
     */
    extern "C" DLLEXPORT void ai_move(ai::chess_ai_state* ai_state, board_state* board_state, std::int32_t difficulty) noexcept;

    /**
     * @brief Get a board_state representing a game that has not yet started.
     * @return The initial board state.
     */
    extern "C" DLLEXPORT board_state get_initial_board_state() noexcept;

} // namespace chess::python
