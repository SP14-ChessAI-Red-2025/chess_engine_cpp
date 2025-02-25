#include <gtest/gtest.h>

#include "chess_cpp/chess_rules.hpp"

#include <cstdint>
#include <algorithm>

// Check that the possible first moves are correctly calculated
TEST(ChessRules, FirstMoveTest) {
    auto board_state = chess::board_state::initial_board_state();
    auto valid_moves = get_valid_moves(board_state);

    ASSERT_EQ(valid_moves.size(), 21);

    ASSERT_NE(std::ranges::find(valid_moves, chess::chess_move{.type = chess::move_type::resign}), valid_moves.end());

    // Test that all valid pawn moves are produced
    for(std::uint8_t file = 0; file < 8; file++) {
        for(std::uint8_t target_rank = 2; target_rank <= 3; target_rank++) {
            chess::chess_move move = {
                .start_position = {1, file},
                .target_position = {target_rank, file},
                .type = chess::move_type::normal_move
            };

            ASSERT_NE(std::ranges::find(valid_moves, move), valid_moves.end());
        }
    }

    // Test that all valid knight moves are produced

    chess::chess_move move = {
        .start_position = {0, 1},
        .target_position = {2, 0},
        .type = chess::move_type::normal_move
    };

    ASSERT_NE(std::ranges::find(valid_moves, move), valid_moves.end());

    move = {
        .start_position = {0, 1},
        .target_position = {2, 2},
        .type = chess::move_type::normal_move
    };

    ASSERT_NE(std::ranges::find(valid_moves, move), valid_moves.end());

    move = {
        .start_position = {0, 6},
        .target_position = {2, 5},
        .type = chess::move_type::normal_move
    };

    ASSERT_NE(std::ranges::find(valid_moves, move), valid_moves.end());

    move = {
        .start_position = {0, 6},
        .target_position = {2, 7},
        .type = chess::move_type::normal_move
    };

    ASSERT_NE(std::ranges::find(valid_moves, move), valid_moves.end());
}


