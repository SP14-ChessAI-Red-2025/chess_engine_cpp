#pragma once

#include "chess_rules.hpp"

#ifdef _MSC_VER
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

// Functions in this namespace will be called from Python
namespace chess::python {

// Initialize any long lived state used by the AI
extern "C" DLLEXPORT void* init_ai_state() noexcept;
// Free any resources associated with state
extern "C" DLLEXPORT void free_ai_state(void* state) noexcept;

// Returns an array of valid moves for a given board state
// Writes the number of valid moves to *num_moves
// The returned array must be freed with free_moves
extern "C" DLLEXPORT chess_move* get_valid_moves(board_state board_state, std::size_t* num_moves) noexcept;

// Free a list of moves allocated by get_valid_moves
extern "C" DLLEXPORT void free_moves(chess_move* moves) noexcept;

// Applies a move to board_state
// Updates board_state->pieces
// Updates board_state->en_passant_valid
// Updates board_state->can_castle if the move if a castle, or the move is of a king or rook that has not moved yet
// Increments board_state->turns_since_last_capture_or_pawn if necessary
// Toggles board_state->current_player
extern "C" DLLEXPORT void apply_move(board_state* board_state, chess_move move) noexcept;

// Have the AI make a move
extern "C" DLLEXPORT void ai_move(board_state* board_state, std::int32_t difficulty) noexcept;

// Get a board_state representing a game that has not yet started
extern "C" DLLEXPORT board_state get_initial_board_state() noexcept;

}
