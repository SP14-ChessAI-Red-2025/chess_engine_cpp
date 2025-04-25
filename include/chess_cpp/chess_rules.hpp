#ifndef CHESS_CPP_CHESS_RULES_HPP
#define CHESS_CPP_CHESS_RULES_HPP

#include <vector>
#include <cstdint>
#include <optional>
#include <string>
#include <stdexcept>

// Forward declaration needed here because previous_board_states uses board_state
// and apply_move/update_status below use previous_board_states
namespace chess { class previous_board_states; }

namespace chess {

    // --- Enums (player, piece_type, move_type, game_status) ---
    enum class player : std::int8_t { white, black };
    enum class piece_type : std::int8_t { none, pawn, knight, bishop, rook, queen, king };
    enum class move_type : std::int8_t {
        normal_move, capture, en_passant, castle, promotion, claim_draw, resign
    };
     enum class game_status : std::int8_t { normal, draw, checkmate, resigned, draw_by_repetition };


    // --- Structs ---
    struct board_position { /* ... as before ... */
        std::uint8_t rank;
        std::uint8_t file;
        bool operator==(const board_position& other) const;
        bool operator<(const board_position& other) const;
    };
    struct piece { /* ... as before ... */
        piece_type type = piece_type::none;
        player piece_player = player::white;
        bool operator==(const piece& other) const;
    };
    struct chess_move { /* ... as before ... */
        move_type type = move_type::normal_move;
        board_position start_position{};
        board_position target_position{};
        piece_type promotion_target = piece_type::none;
    };
    struct board_state { /* ... as before ... */
        piece pieces[8][8]{};
        bool can_castle[4]{};
        bool in_check[2]{};
        bool en_passant_valid[16]{};
        int turns_since_last_capture_or_pawn = 0;
        player current_player = player::white;
        game_status status = game_status::normal;
        bool can_claim_draw = false;
        static board_state initial_board_state() noexcept;
    };
    struct board_offset { /* ... as before ... */
        int rank_offset;
        int file_offset;
    };

    // --- Function Declarations ---
    bool in_bounds(int rank, int file);
    std::optional<board_position> apply_offset(board_position position, board_offset offset);
    std::vector<chess_move> get_valid_moves(const board_state& board_state);

    /**
     * @brief Applies a given move to the board state, updates history, and updates the game status.
     * Assumes the move is valid.
     * @param board The board state to modify.
     * @param move The move to apply.
     * @param history The game history object to update and check for repetitions.
     * @return The board state after the move has been applied and status updated.
     */
    // <<<--- MODIFIED: Added history parameter ---<<<
    void apply_move(board_state& board, chess_move move, previous_board_states& history);

    // <<<--- OVERLOAD (Optional): Keep original apply_move if needed for contexts without history ---<<<
    // board_state apply_move(board_state board, chess_move move);


    bool operator==(const board_state& lhs, const board_state& rhs);
    bool is_player_in_check(const board_state& board, player player_in_check);

    /**
     * @brief Updates the game status (checkmate, stalemate, draw by rule) based on the current board state and history.
     * @param board The board state to check and potentially modify.
     * @param history The game history object to check for repetitions.
     */
    void update_status(board_state& board, previous_board_states& history);

} // namespace chess

#endif // CHESS_CPP_CHESS_RULES_HPP
