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
    /** @brief Creates an engine handle, initializing AI state and board. */
    DLLEXPORT void* engine_create(const char* model_path) noexcept;
    /** @brief Destroys the engine handle and frees associated resources. */
    DLLEXPORT void engine_destroy(void* engine_handle_opaque) noexcept;

    // --- Board State Access ---
    /** @brief Returns a pointer to the current board state held within the engine handle. */
    DLLEXPORT chess::board_state* engine_get_board_state(void* engine_handle_opaque) noexcept;

    // --- Get Valid Moves ---
    /**
     * @brief Gets the valid moves for the current board state within the handle.
     *
     * If out_moves_buffer is NULL or buffer_capacity is 0, returns the number
     * of moves needed. Otherwise, attempts to fill the buffer.
     *
     * @param engine_handle_opaque Opaque pointer to the EngineHandle.
     * @param out_moves_buffer A pointer to a buffer allocated by the caller (Python)
     * where the valid moves will be copied. Can be NULL.
     * @param buffer_capacity The maximum number of chess_move elements the buffer can hold.
     * @return The number of valid moves found (which is also the number required/copied).
     * Returns 0 on error or if no moves are available.
     */
    DLLEXPORT size_t engine_get_valid_moves(void* engine_handle_opaque,
                                            chess::chess_move* out_moves_buffer,
                                            size_t buffer_capacity) noexcept;

    // --- Apply Move ---
    /**
     * @brief Applies a move using the engine handle, updating internal state and history.
     * Copies the resulting board state back to the caller.
     * @param engine_handle_opaque Opaque pointer to the EngineHandle.
     * @param move Pointer to the move to apply.
     * @param out_board_state Pointer to a BoardState struct provided by the caller
     * where the resulting state will be copied.
     * @return true if the move was applied successfully, false otherwise.
     */
    DLLEXPORT bool engine_apply_move(void* engine_handle_opaque,
        const chess::chess_move* move,
        chess::board_state* out_board_state) noexcept;

    // --- AI Move Calculation ---
    // Define callback type matching Python's expectation if needed, though unused in C++ now.
    using ProgressCallback = void (*)(int, int, int, uint64_t, const chess::chess_move*);
    /**
     * @brief Asks the AI within the engine handle to make a move. Modifies the internal board state.
     * @param engine_handle_opaque Opaque pointer to the EngineHandle.
     * @param difficulty Difficulty level (e.g., search depth).
     * @param callback Pointer to a progress callback function (currently ignored by C++).
     * @return true if the AI move calculation and application succeeded, false otherwise.
     */
    DLLEXPORT bool engine_ai_move(void* engine_handle_opaque, int difficulty, ProgressCallback callback) noexcept;


    // --- Move to String ---
    /**
     * @brief Converts a chess move to a string representation (e.g., SAN).
     * @param engine_handle_opaque Opaque pointer to the EngineHandle.
     * @param move Pointer to the move to convert.
     * @param buffer Caller-provided buffer to store the string.
     * @param buffer_size Size of the caller-provided buffer.
     * @return true if conversion succeeded and fit in buffer, false otherwise.
     */
    DLLEXPORT bool engine_move_to_str(void* engine_handle_opaque, const chess::chess_move* move, char* buffer, size_t buffer_size) noexcept;

    // --- Cancellation ---
    /** @brief Signals the AI search associated with the engine handle to stop. */
    DLLEXPORT void engine_cancel_search(void* engine_handle_opaque) noexcept;

    // --- REMOVED OLD/CONFLICTING DECLARATIONS ---
    // DLLEXPORT void* init_ai_state(const char* model_path) noexcept;
    // DLLEXPORT void free_ai_state(void* state) noexcept;
    // DLLEXPORT void* create_history() noexcept;
    // DLLEXPORT void free_history(void* history_ptr) noexcept;
    // DLLEXPORT chess::board_state get_initial_board_state() noexcept;
    // DLLEXPORT bool get_valid_moves(...) noexcept; // Conflicting overload removed
    // DLLEXPORT void apply_move(...) noexcept;
    // DLLEXPORT void ai_move(...) noexcept;
    // DLLEXPORT double evaluate_board(...) noexcept;
    // DLLEXPORT void cancel_search() noexcept; // Use engine_cancel_search instead
    // DLLEXPORT bool engine_get_valid_moves(...) noexcept; // Conflicting overload removed
    // DLLEXPORT bool engine_get_board_state(...) noexcept; // Conflicting overload removed

} // extern "C"

#endif // CHESS_CPP_PYTHON_API_HPP
