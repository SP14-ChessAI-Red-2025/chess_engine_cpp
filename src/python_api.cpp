// src/python_api.cpp
#include "chess_cpp/python_api.hpp" // Include the header defining the C API
#include "chess_cpp/chess_ai.hpp"   // Include AI state definition
#include "chess_cpp/chess_rules.hpp" // Include C++ rule functions and types
#include "chess_cpp/threefold_repetition.hpp" // Include history state definition

#include <new>       // For std::nothrow
#include <vector>
#include <string>
#include <algorithm> // For std::copy, std::min
#include <stdexcept>
#include <iostream>  // For error logging
#include <memory>    // For std::unique_ptr
#include <cstring>   // For strncpy
#include <sstream>   // For placeholder move_to_san

// --- Define C-compatible types ---
// Matches Python's ctypes definition
// Note: This callback type is defined but not used by the current C++ ai_move implementation
using ProgressCallback = void (*)(int, int, int, uint64_t, const chess::chess_move*);

// Define a structure to hold all necessary engine components
struct EngineHandle {
    chess::ai::chess_ai_state ai_state;
    chess::previous_board_states history;
    chess::board_state current_board; // Store the current board state here
    bool initialized_correctly = false; // Flag for successful init

    // Constructor to initialize components
    EngineHandle(const std::string& model_path) try : // Use function-try-block for constructor safety
        ai_state(model_path), // Initialize AI state
        history(),            // Initialize history
        current_board(chess::board_state::initial_board_state()) // Initialize board
    {
        history.add_board_state(current_board);
        // Basic check: Assume if ai_state constructor didn't throw, it's okay.
        initialized_correctly = true; // Set flag
        std::cout << "[C++ EngineHandle] Constructor finished. Initial player: "
                  << static_cast<int>(current_board.current_player) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[C++ EngineHandle ERROR] Exception during construction: " << e.what() << std::endl;
        // initialized_correctly remains false
    } catch (...) {
        std::cerr << "[C++ EngineHandle ERROR] Unknown exception during construction." << std::endl;
        // initialized_correctly remains false
    }


    // Default constructor deleted
    EngineHandle() = delete;

    // --- Placeholder/Wrapper Methods ---
    bool is_ai_state_initialized() const {
        // Rely on the flag set (or not set) in the constructor
        return initialized_correctly;
    }

    void cancel_ai_search_internal() {
        // Call the global cancel function from chess_ai.hpp
        std::cerr << "[C++ EngineHandle] Calling global chess::ai::cancel_ai_search()." << std::endl;
        chess::ai::cancel_ai_search();
    }

    // Placeholder for move_to_san
    std::string convert_move_to_san(const chess::chess_move& move) const {
        std::stringstream ss;
        // Use static_cast<char> for clarity, ensure file is 0-7 ('a'-'h') and rank is 0-7 ('1'-'8')
        ss << static_cast<char>('a' + move.start_position.file) << (move.start_position.rank + 1);
        ss << static_cast<char>('a' + move.target_position.file) << (move.target_position.rank + 1);

        // Use correct enum names from chess_rules.hpp
        if (move.type == chess::move_type::promotion) {
            switch (move.promotion_target) {
                case chess::piece_type::queen: ss << "=Q"; break;
                case chess::piece_type::rook: ss << "=R"; break;
                case chess::piece_type::bishop: ss << "=B"; break;
                case chess::piece_type::knight: ss << "=N"; break;
                default: ss << "=?"; break; // Handle unknown/none promotion type
            }
        }
        // Basic castle notation (doesn't check actual castling move type, only coordinates)
        // A proper SAN generator would use move.type == chess::move_type::castle
        else if (move.type == chess::move_type::castle) {
             // Kingside castle: King moves e1g1 or e8g8 (file 4 -> 6)
             // Queenside castle: King moves e1c1 or e8c8 (file 4 -> 2)
             if (move.target_position.file == 6) { // Kingside target file is 6 (g-file)
                 ss.str(""); // Clear coordinate string
                 ss << "O-O";
             } else if (move.target_position.file == 2) { // Queenside target file is 2 (c-file)
                 ss.str(""); // Clear coordinate string
                 ss << "O-O-O";
             }
        }
        // Add check/checkmate symbols would require re-evaluating the resulting board state
        return ss.str();
    }
};


// Define functions within extern "C" block to ensure C linkage
extern "C" {

    // --- Engine Handle Management ---
    DLLEXPORT void* engine_create(const char* model_path) noexcept {
        if (!model_path) {
            std::cerr << "[C API ERROR] engine_create called with null model_path." << std::endl;
            return nullptr;
        }
        EngineHandle* handle = nullptr;
        try {
            handle = new(std::nothrow) EngineHandle(model_path);
            if (!handle) {
                std::cerr << "[C API ERROR] Failed to allocate memory for EngineHandle." << std::endl;
                return nullptr;
            }
            if (!handle->is_ai_state_initialized()) {
                std::cerr << "[C API ERROR] EngineHandle initialization failed (check constructor logs)." << std::endl;
                delete handle;
                return nullptr;
            }
            std::cout << "[C API] engine_create successful." << std::endl;
            return handle;
        } catch (...) {
            // Catch potential exceptions during EngineHandle construction if not caught by function-try-block
            std::cerr << "[C API ERROR] Unknown exception during engine_create." << std::endl;
            delete handle; // Clean up potentially partially constructed object
            return nullptr;
        }
    }

    DLLEXPORT void engine_destroy(void* engine_handle_opaque) noexcept {
        if (engine_handle_opaque) {
            std::cout << "[C API] engine_destroy called." << std::endl;
            delete static_cast<EngineHandle*>(engine_handle_opaque);
        } else {
            std::cerr << "[C API WARNING] engine_destroy called with null handle." << std::endl;
        }
    }

    // --- Board State Access ---
    DLLEXPORT chess::board_state* engine_get_board_state(void* engine_handle_opaque) noexcept {
        if (!engine_handle_opaque) {
            std::cerr << "[C API ERROR] engine_get_board_state called with null handle." << std::endl;
            return nullptr;
        }
        EngineHandle* handle = static_cast<EngineHandle*>(engine_handle_opaque);
        // Return pointer to the board state stored within the handle
        return &(handle->current_board);
    }

    // --- Get Valid Moves ---
    DLLEXPORT size_t engine_get_valid_moves(void* engine_handle_opaque,
                                            chess::chess_move* out_moves_buffer,
                                            size_t buffer_capacity) noexcept
    {
        if (!engine_handle_opaque) {
            std::cerr << "[C API ERROR] engine_get_valid_moves called with null handle." << std::endl;
            return 0;
        }
        EngineHandle* handle = static_cast<EngineHandle*>(engine_handle_opaque);
        std::vector<chess::chess_move> valid_moves_vec;
        try {
            // Get moves for the board state stored in the handle
            valid_moves_vec = chess::get_valid_moves(handle->current_board);
        } catch(const std::exception& e) {
            std::cerr << "[C API ERROR] Exception getting valid moves from C++ rules: " << e.what() << std::endl;
            return 0;
        } catch(...) {
            std::cerr << "[C API ERROR] Unknown error getting valid moves from C++ rules." << std::endl;
            return 0;
        }
        size_t num_moves_found = valid_moves_vec.size();

        // If buffer provided, copy moves (up to capacity)
        if (out_moves_buffer && buffer_capacity > 0) {
            size_t num_to_copy = std::min(num_moves_found, buffer_capacity);
            if (num_to_copy < num_moves_found) {
                std::cerr << "[C API WARNING] engine_get_valid_moves: Buffer capacity (" << buffer_capacity
                          << ") < moves found (" << num_moves_found << "). Truncating." << std::endl;
            }
            if (num_to_copy > 0) { // Only copy if there's something to copy
                std::copy(valid_moves_vec.begin(), valid_moves_vec.begin() + num_to_copy, out_moves_buffer);
            }
            // Return the number actually copied (or that would have been copied if buffer was large enough)
            // It's often more useful for the caller to know how many moves *exist* in total.
        }
        // Return the total number of moves found, regardless of buffer size.
        return num_moves_found;
    }

    // --- Apply Move ---
    DLLEXPORT bool engine_apply_move(void* engine_handle_opaque, const chess::chess_move* move,
        chess::board_state* out_board_state) noexcept {
        if (!engine_handle_opaque || !move || !out_board_state) {
             std::cerr << "[C API ERROR] engine_apply_move called with null handle, move, or output buffer." << std::endl;
             return false;
        }
        EngineHandle* handle = static_cast<EngineHandle*>(engine_handle_opaque);

        try {
            // Apply the move to the board state stored WITHIN the handle
            chess::apply_move(handle->current_board, *move, handle->history);

            // Copy the UPDATED board state from the handle to the output parameter
            *out_board_state = handle->current_board;

            return true; // Indicate success

        } catch(const std::exception& e) {
            std::cerr << "[C API ERROR] Exception applying move: " << e.what() << std::endl;
            return false;
        } catch(...) {
            std::cerr << "[C API ERROR] Unknown error applying move." << std::endl;
            return false;
        }
    }

    // --- AI Move Calculation ---
    DLLEXPORT bool engine_ai_move(void* engine_handle_opaque, int difficulty, ProgressCallback /*callback*/) noexcept {
        if (!engine_handle_opaque) {
            std::cerr << "[C API ERROR] engine_ai_move called with null handle." << std::endl;
            return false;
        }
        EngineHandle* handle = static_cast<EngineHandle*>(engine_handle_opaque);

        std::cerr << "[C API] engine_ai_move called for player "
                  << static_cast<int>(handle->current_board.current_player)
                  << " with difficulty " << difficulty << std::endl;
        try {
            // Call the AI state's make_move method, passing the handle's board and history
            handle->ai_state.make_move(handle->current_board, difficulty, handle->history);

            // Assumed success if no exception
            std::cerr << "[C API] engine_ai_move: C++ make_move (void) completed. Final state player="
                      << static_cast<int>(handle->current_board.current_player)
                      << ", status=" << static_cast<int>(handle->current_board.status) << std::endl;
            return true;

        } catch (const std::exception& e) {
            std::cerr << "[C API ERROR] Exception during AI move calculation: " << e.what() << std::endl;
            return false;
        } catch(...) {
            std::cerr << "[C API ERROR] Unknown exception during AI move calculation." << std::endl;
            return false;
        }
    }

    // --- Move to String ---
    DLLEXPORT bool engine_move_to_str(void* engine_handle_opaque, const chess::chess_move* move, char* buffer, size_t buffer_size) noexcept {
        if (!engine_handle_opaque || !move || !buffer || buffer_size == 0) return false;
        EngineHandle* handle = static_cast<EngineHandle*>(engine_handle_opaque);
        try {
            std::string move_str = handle->convert_move_to_san(*move); // Use handle's helper

            if (move_str.length() < buffer_size) {
                strncpy(buffer, move_str.c_str(), buffer_size); // Copy includes null terminator if space
                buffer[move_str.length()] = '\0'; // Ensure null termination
                return true;
            } else {
                // Buffer too small, copy truncated string and return false
                strncpy(buffer, move_str.c_str(), buffer_size - 1);
                buffer[buffer_size - 1] = '\0'; // Null terminate truncated string
                std::cerr << "[C API WARNING] engine_move_to_str: Buffer too small for SAN string '" << move_str << "'" << std::endl;
                return false;
            }
        } catch (...) {
            std::cerr << "[C API ERROR] Unknown exception during move_to_str." << std::endl;
            if (buffer_size > 0) buffer[0] = '\0'; // Ensure buffer is safe on error
            return false;
        }
    }

    // --- Cancellation ---
     DLLEXPORT void engine_cancel_search(void* engine_handle_opaque) noexcept {
        if (!engine_handle_opaque) return;
        EngineHandle* handle = static_cast<EngineHandle*>(engine_handle_opaque);
        try {
            std::cerr << "[C API] engine_cancel_search called." << std::endl;
            handle->cancel_ai_search_internal(); // Calls the handle method
        } catch (...) {
            std::cerr << "[C API ERROR] Unknown error during cancel_search." << std::endl;
        }
     }

} // extern "C"