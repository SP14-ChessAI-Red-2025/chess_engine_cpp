#include "chess_cpp/python_api.hpp"

#include "chess_cpp/chess_ai.hpp"
#include "chess_cpp/chess_rules.hpp"

#include <new>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace chess::python {

    // Updated function to accept model path
    void* init_ai_state(const char* model_path) noexcept {
        if (!model_path) {
            std::cerr << "Error: init_ai_state called with null model_path." << std::endl;
            return nullptr;
        }

        try {
            std::string model_path_str(model_path);
            // Use nothrow to avoid exceptions crossing the C boundary directly
            return new(std::nothrow) ai::chess_ai_state{model_path_str};
        } catch (const std::exception& e) {
             // Catch exceptions *during* construction within C++
             std::cerr << "Error initializing AI state: " << e.what() << std::endl;
            return nullptr;
        } catch (...) {
            std::cerr << "Unknown error initializing AI state." << std::endl;
            return nullptr;
        }
    }

    void free_ai_state(void* state) noexcept {
        delete static_cast<ai::chess_ai_state*>(state);
    }

    chess_move* get_valid_moves(board_state board_state, std::size_t* num_moves) noexcept {
        if (!num_moves) return nullptr;
        std::vector<chess_move> valid_moves_vec;
        try {
            valid_moves_vec = chess::get_valid_moves(board_state);
        } catch(...) {
            // Avoid letting exceptions cross the C boundary
            std::cerr << "Error getting valid moves." << std::endl;
            *num_moves = 0;
            return nullptr;
        }
        *num_moves = valid_moves_vec.size();
        if (valid_moves_vec.empty()) return nullptr;
        // Use nothrow to prevent allocation exceptions crossing the boundary
        auto* result = new(std::nothrow) chess_move[valid_moves_vec.size()];
        if (!result) {
            // Allocation failed
            std::cerr << "Error allocating memory for moves array." << std::endl;
            *num_moves = 0;
            return nullptr;
        }
        std::copy(valid_moves_vec.begin(), valid_moves_vec.end(), result);
        return result;
    }

    void free_moves(chess_move* moves) noexcept {
        delete[] moves;
    }

    void apply_move(board_state* board_state, chess_move move) noexcept {
        if (!board_state) return;
        try {
            *board_state = chess::apply_move(*board_state, move);
        } catch(...) {
            // Avoid letting exceptions cross the C boundary
            std::cerr << "Error applying move." << std::endl;
            // Optionally modify board_state to an error state if possible/needed
        }
    }

    // Accept void* and cast it internally
    void ai_move(void* ai_state_opaque, board_state* board_state, std::int32_t difficulty) noexcept {
        if (!ai_state_opaque || !board_state) return;

        ai::chess_ai_state* ai_state = static_cast<ai::chess_ai_state*>(ai_state_opaque);

        try {
            ai_state->make_move(*board_state, difficulty);
        } catch (const std::exception& e) {
             std::cerr << "Error during AI move calculation: " << e.what() << std::endl;
        } catch(...) {
             std::cerr << "Unknown error during AI move calculation." << std::endl;
        }
    }


    board_state get_initial_board_state() noexcept {
        return board_state::initial_board_state();
    }

} // namespace chess::python