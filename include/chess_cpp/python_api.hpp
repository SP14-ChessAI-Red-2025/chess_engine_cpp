// src/python_api.hpp
#ifndef CHESS_CPP_PYTHON_API_HPP
#define CHESS_CPP_PYTHON_API_HPP

#include "chess_rules.hpp" // Include C++ type definitions
#include <cstdint>         // For std::int32_t, std::size_t
#include <cstddef>         // For std::size_t on some compilers

// Define DLLEXPORT based on platform (simplified)
#ifdef _WIN32
    #define DLLEXPORT __declspec(dllexport)
#else
    #define DLLEXPORT __attribute__((visibility("default")))
#endif

// Use extern "C" to prevent C++ name mangling for C API functions
extern "C" {

    // --- Engine Handle Management ---
    DLLEXPORT void* engine_create(const char* model_path) noexcept;
    DLLEXPORT void engine_destroy(void* engine_handle_opaque) noexcept;

    // --- Board State Access ---
    DLLEXPORT chess::board_state* engine_get_board_state(void* engine_handle_opaque) noexcept;

    // --- Get Valid Moves ---
    DLLEXPORT size_t engine_get_valid_moves(void* engine_handle_opaque,
                                            chess::chess_move* out_moves_buffer,
                                            size_t buffer_capacity) noexcept;

    // --- Apply Move (MODIFIED) ---
    /**
     * @brief Applies a move using the engine handle, updating internal state and history.
     * @param engine_handle_opaque Opaque pointer to the EngineHandle.
     * @param move Pointer to the move to apply.
     * @return Pointer to the updated internal board_state within the handle on success, nullptr on failure.
     */
    DLLEXPORT chess::board_state* engine_apply_move(void* engine_handle_opaque,
                                                    const chess::chess_move* move) noexcept; // Removed out_board_state*, returns pointer

    // --- AI Move Calculation (MODIFIED) ---
    using ProgressCallback = void (*)(int, int, int, uint64_t, const chess::chess_move*);
    /**
     * @brief Asks the AI within the engine handle to make a move. Modifies the internal board state.
     * @param engine_handle_opaque Opaque pointer to the EngineHandle.
     * @param difficulty Difficulty level (e.g., search depth).
     * @param callback Pointer to a progress callback function (currently ignored by C++).
     * @return Pointer to the updated internal board_state within the handle on success, nullptr on failure.
     */
    DLLEXPORT chess::board_state* engine_ai_move(void* engine_handle_opaque,
                                                  int difficulty,
                                                  ProgressCallback callback) noexcept; // Returns pointer

    // --- Move to String ---
    DLLEXPORT bool engine_move_to_str(void* engine_handle_opaque,
                                       const chess::chess_move* move,
                                       char* buffer, size_t buffer_size) noexcept;

    // --- Cancellation ---
    DLLEXPORT void engine_cancel_search(void* engine_handle_opaque) noexcept;

} // extern "C"

#endif // CHESS_CPP_PYTHON_API_HPP