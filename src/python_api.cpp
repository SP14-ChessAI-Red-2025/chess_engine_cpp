// src/python_api.cpp
#include "chess_cpp/python_api.hpp"
#include "chess_cpp/chess_ai.hpp"
#include "chess_cpp/chess_rules.hpp"
#include "chess_cpp/threefold_repetition.hpp"

#include <new>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <cstring>
#include <sstream>
#include <filesystem> // Added for model path conversion

// Define C-compatible types
using ProgressCallback = void (*)(int, int, int, uint64_t, const chess::chess_move*);

// Define EngineHandle struct
struct EngineHandle {
    chess::ai::chess_ai_state ai_state;
    chess::previous_board_states history;
    chess::board_state current_board;
    bool initialized_correctly = false;

    EngineHandle(const std::string& model_path) try :
        ai_state(model_path),
        history(),
        current_board(chess::board_state::initial_board_state())
    {
        history.add_board_state(current_board);
        initialized_correctly = true;
        std::cout << "[C++ EngineHandle] Constructor finished." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[C++ EngineHandle ERROR] Exception during construction: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[C++ EngineHandle ERROR] Unknown exception during construction." << std::endl;
    }

    EngineHandle() = delete;

    bool is_ai_state_initialized() const { return initialized_correctly; }
    void cancel_ai_search_internal() { chess::ai::cancel_ai_search(); }

    // Placeholder move to SAN (same as before)
    std::string convert_move_to_san(const chess::chess_move& move) const {
        std::stringstream ss;
        ss << static_cast<char>('a' + move.start_position.file) << (move.start_position.rank + 1);
        ss << static_cast<char>('a' + move.target_position.file) << (move.target_position.rank + 1);
        if (move.type == chess::move_type::promotion) {
            switch (move.promotion_target) {
                case chess::piece_type::queen: ss << "=Q"; break; // Changed char to string
                case chess::piece_type::rook: ss << "=R"; break;
                case chess::piece_type::bishop: ss << "=B"; break;
                case chess::piece_type::knight: ss << "=N"; break;
                default: ss << "=?"; break;
            }
        } else if (move.type == chess::move_type::castle) {
             if (move.target_position.file == 6) { ss.str(""); ss << "O-O"; }
             else if (move.target_position.file == 2) { ss.str(""); ss << "O-O-O"; }
        }
        return ss.str();
    }
};

// Define functions within extern "C" block
extern "C" {

    // --- Engine Handle Management --- 
    DLLEXPORT void* engine_create(const char* model_path) noexcept {
        if (!model_path) { /* ... error handling ... */ return nullptr; }
        EngineHandle* handle = nullptr;
        try {
            handle = new(std::nothrow) EngineHandle(model_path);
            if (!handle || !handle->is_ai_state_initialized()) { /* ... error handling ... */ delete handle; return nullptr; }
            std::cout << "[C API] engine_create successful." << std::endl;
            return handle;
        } catch (...) {
            delete handle; return nullptr; 
        }
    }

    DLLEXPORT void engine_destroy(void* engine_handle_opaque) noexcept {
        if (engine_handle_opaque) {
            std::cout << "[C API] engine_destroy called." << std::endl;
            delete static_cast<EngineHandle*>(engine_handle_opaque);
        } else { /* ... warning ... */ }
    }

    DLLEXPORT void engine_reset_engine(void* engine_handle_opaque) noexcept {
        if (!engine_handle_opaque) {
            std::cerr << "[C API ERROR] reset_engine called with null handle." << std::endl;
            return;
        }
    
        EngineHandle* handle = static_cast<EngineHandle*>(engine_handle_opaque);
        try {
            // Reset the board state
            handle->current_board = chess::board_state::initial_board_state();
    
            // Reset the history by replacing it with a new instance
            handle->history = chess::previous_board_states();
            handle->history.add_board_state(handle->current_board);
    
            std::cout << "[C API] engine_reset_engine: Engine successfully reset to initial state." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[C API ERROR] Exception during engine_reset_engine: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[C API ERROR] Unknown exception during engine_reset_engine." << std::endl;
        }
    }

    // --- Board State Access --- 
    DLLEXPORT chess::board_state* engine_get_board_state(void* engine_handle_opaque) noexcept {
        if (!engine_handle_opaque) { /* ... error handling ... */ return nullptr; }
        return &(static_cast<EngineHandle*>(engine_handle_opaque)->current_board);
    }

    // --- Get Valid Moves ---
    DLLEXPORT size_t engine_get_valid_moves(void* engine_handle_opaque,
                                            chess::chess_move* out_moves_buffer,
                                            size_t buffer_capacity) noexcept
    {
        if (!engine_handle_opaque) { /* ... error handling ... */ return 0; }
        EngineHandle* handle = static_cast<EngineHandle*>(engine_handle_opaque);
        std::vector<chess::chess_move> valid_moves_vec;
        try {
            valid_moves_vec = chess::get_valid_moves(handle->current_board);
        } catch(...) { /* ... error handling ... */ return 0; }
        size_t num_moves_found = valid_moves_vec.size();
        if (out_moves_buffer && buffer_capacity > 0) {
            size_t num_to_copy = std::min(num_moves_found, buffer_capacity);
            if (num_to_copy < num_moves_found) { /* ... warning ... */ }
            if (num_to_copy > 0) {
                std::copy(valid_moves_vec.begin(), valid_moves_vec.begin() + num_to_copy, out_moves_buffer);
            }
        }
        return num_moves_found;
    }

    // --- Apply Move ---
    DLLEXPORT chess::board_state* engine_apply_move(void* engine_handle_opaque,
                                                    const chess::chess_move* move) noexcept { // Removed out_board_state*, returns pointer
        if (!engine_handle_opaque || !move) {
             std::cerr << "[C API ERROR] engine_apply_move called with null handle or move." << std::endl;
             return nullptr; // Return null on error
        }
        EngineHandle* handle = static_cast<EngineHandle*>(engine_handle_opaque);

        try {
            // Apply the move - this MODIFIES handle->current_board by reference
            std::cerr << "DEBUG: Before move, current_player: " << static_cast<int>(handle->current_board.current_player) << std::endl;
            chess::apply_move(handle->current_board, *move, handle->history);
            std::cerr << "DEBUG: After move, current_player: " << static_cast<int>(handle->current_board.current_player) << std::endl;
            // Return pointer to the MODIFIED internal board state
            return &(handle->current_board);

        } catch(const std::exception& e) {
            std::cerr << "[C API ERROR] Exception applying move: " << e.what() << std::endl;
            return nullptr; // Return null on error
        } catch(...) {
            std::cerr << "[C API ERROR] Unknown error applying move." << std::endl;
            return nullptr; // Return null on error
        }
    }

    // --- AI Move Calculation ---
    DLLEXPORT chess::board_state* engine_ai_move(void* engine_handle_opaque, int difficulty, ProgressCallback /*callback*/) noexcept { // Returns pointer
        if (!engine_handle_opaque) {
            std::cerr << "[C API ERROR] engine_ai_move called with null handle." << std::endl;
            return nullptr; // Return null on error
        }
        EngineHandle* handle = static_cast<EngineHandle*>(engine_handle_opaque);

        std::cerr << "[C API] engine_ai_move called for player "
                  << static_cast<int>(handle->current_board.current_player)
                  << " with difficulty " << difficulty << std::endl;
        try {
            // Call make_move - this MODIFIES handle->current_board by reference
            handle->ai_state.make_move(handle->current_board, difficulty, handle->history);

            std::cerr << "[C API] engine_ai_move: C++ make_move completed. Final state player="
                      << static_cast<int>(handle->current_board.current_player)
                      << ", status=" << static_cast<int>(handle->current_board.status) << std::endl;

            // Return pointer to the MODIFIED internal board state
            return &(handle->current_board);

        } catch (const std::exception& e) {
            std::cerr << "[C API ERROR] Exception during AI move calculation: " << e.what() << std::endl;
            return nullptr; // Return null on error
        } catch(...) {
            std::cerr << "[C API ERROR] Unknown exception during AI move calculation." << std::endl;
            return nullptr; // Return null on error
        }
    }

    // --- Move to String ---
    DLLEXPORT bool engine_move_to_str(void* engine_handle_opaque, const chess::chess_move* move, char* buffer, size_t buffer_size) noexcept {
        if (!engine_handle_opaque || !move || !buffer || buffer_size == 0) return false;
        EngineHandle* handle = static_cast<EngineHandle*>(engine_handle_opaque);
        try {
            std::string move_str = handle->convert_move_to_san(*move);
            if (move_str.length() < buffer_size) {
                strncpy(buffer, move_str.c_str(), buffer_size);
                buffer[move_str.length()] = '\0';
                return true;
            } else {
                strncpy(buffer, move_str.c_str(), buffer_size - 1);
                buffer[buffer_size - 1] = '\0';
                std::cerr << "[C API WARNING] engine_move_to_str: Buffer too small for SAN string '" << move_str << "'" << std::endl;
                return false;
            }
        } catch (...) {
            std::cerr << "[C API ERROR] Unknown exception during move_to_str." << std::endl;
            if (buffer_size > 0) buffer[0] = '\0';
            return false;
        }
    }

    // --- Cancellation ---
     DLLEXPORT void engine_cancel_search(void* engine_handle_opaque) noexcept {
        if (!engine_handle_opaque) return;
        EngineHandle* handle = static_cast<EngineHandle*>(engine_handle_opaque);
        try {
            handle->cancel_ai_search_internal();
        } catch (...) { /* ... error handling ... */ }
     }

    // --- Evaluate Board ---
    DLLEXPORT double engine_evaluate_board(void* engine_handle_opaque) noexcept {
        if (!engine_handle_opaque) {
            std::cerr << "[C API ERROR] engine_evaluate_board called with null handle." << std::endl;
            return 0.0; // Return a default value on error
        }

        EngineHandle* handle = static_cast<EngineHandle*>(engine_handle_opaque);

#ifdef NNUE_ENABLED // Check if NNUE is enabled
        try {
            return handle->ai_state.evaluator_.evaluate(handle->current_board);
        } catch (const std::exception& e) {
            std::cerr << "[C API ERROR] Exception during evaluate_board: " << e.what() << std::endl;
            return 0.0; // Return a default value on error
        } catch (...) {
            std::cerr << "[C API ERROR] Unknown exception during evaluate_board." << std::endl;
            return 0.0; // Return a default value on error
        }
#else
        std::cerr << "[C API WARNING] NNUE evaluation is disabled in this build." << std::endl;
        return 0.0; // Return a default value if NNUE is disabled
#endif
    }

} // extern "C"