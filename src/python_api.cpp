#include "python_api.hpp"

#include "chess_ai.hpp"

#include <new>
#include <algorithm>

namespace chess::python {

void* init_ai_state(const char* model_path) noexcept {
    try {
        return new(std::nothrow) ai::chess_ai_state{model_path};
    } catch(...) {
        // TODO: Return error to Python somehow
        return nullptr;
    }
}

void free_ai_state(void* state) noexcept {
    delete static_cast<ai::chess_ai_state*>(state);
}

chess_move* get_valid_moves(board_state board_state, std::size_t* num_moves) noexcept {
    std::vector<chess_move> valid_moves;

    try {
        valid_moves = get_valid_moves(board_state);
    } catch(...) {
        // TODO: Return error to Python somehow
        return nullptr;
    }

    *num_moves = valid_moves.size();

    auto* result = new(std::nothrow) chess_move[valid_moves.size()];

    if(result) std::ranges::copy(valid_moves, result);

    return result;
}

void free_moves(chess_move* moves) noexcept {
    delete[] moves;
}

void apply_move(board_state* board_state, chess_move move) noexcept {
    try {
        *board_state = apply_move(*board_state, move);
    } catch(...) {
        // TODO: Return error to Python somehow
    }
}

void ai_move(ai::chess_ai_state* ai_state, board_state* board_state, std::int32_t difficulty) noexcept {
    try {
        ai_state->make_move(*board_state, difficulty);
    } catch(...) {
        // TODO: Return error to Python somehow
    }
}

board_state get_initial_board_state() noexcept {
    return board_state::initial_board_state();
}
}
