#include <gtest/gtest.h>

#include "chess_cpp/chess_rules.hpp"

#include <cstdint>
#include <algorithm>
#include <ranges>

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
                .type = chess::move_type::normal_move,
                .start_position = {1, file},
                .target_position = {target_rank, file}
            };

            ASSERT_NE(std::ranges::find(valid_moves, move), valid_moves.end());
        }
    }

    // Test that all valid knight moves are produced

    chess::chess_move move = {
        .type = chess::move_type::normal_move,
        .start_position = {0, 1},
        .target_position = {2, 0}
    };

    ASSERT_NE(std::ranges::find(valid_moves, move), valid_moves.end());

    move = {
        .type = chess::move_type::normal_move,
        .start_position = {0, 1},
        .target_position = {2, 2}
    };

    ASSERT_NE(std::ranges::find(valid_moves, move), valid_moves.end());

    move = {
        .type = chess::move_type::normal_move,
        .start_position = {0, 6},
        .target_position = {2, 5}
    };

    ASSERT_NE(std::ranges::find(valid_moves, move), valid_moves.end());

    move = {
        .type = chess::move_type::normal_move,
        .start_position = {0, 6},
        .target_position = {2, 7}
    };

    ASSERT_NE(std::ranges::find(valid_moves, move), valid_moves.end());
}

void apply_moves(const std::ranges::range auto& indices, chess::board_state& board_state) {
    auto valid_moves = get_valid_moves(board_state);

    for(auto i : indices) {
        board_state = apply_move(board_state, valid_moves[i]);

        valid_moves = get_valid_moves(board_state);
    }
}

// Test that a fool's mate is properly detected
TEST(ChessRules, FoolsMateTest) {
    std::size_t indices[] = {14, 8, 15, 20};

    auto board_state = chess::board_state::initial_board_state();

    apply_moves(indices, board_state);

    ASSERT_EQ(board_state.status, chess::game_status::checkmate);
    ASSERT_EQ(board_state.current_player, chess::player::white);
}

// Test that draws are properly detected
TEST(ChessRules, DrawTest) {
    std::size_t indices[] = {
        12, 1,
        5, 13,
        43, 23,
        28, 13,
        25, 14,
        41, 3,
        43, 24,
        29, 3,
        39, 4,
        28
    };

    auto board_state = chess::board_state::initial_board_state();

    apply_moves(indices, board_state);

    ASSERT_EQ(board_state.status, chess::game_status::draw);
    ASSERT_EQ(board_state.current_player, chess::player::black);
}

// Test the 50 move rule and 75 move rule
TEST(ChessRules, FiftyMoveRule) {
    std::size_t indices[] = {
        4, 0
    };

    auto board_state = chess::board_state::initial_board_state();

    apply_moves(indices, board_state);

    for(int i = 0; i < 24; i++) {
        // Moves rooks forward and backward, ending up with the same positions
        std::size_t rook_moves[] = {
            0, 15,
            3, 1
        };

        apply_moves(rook_moves, board_state);
    }

    std::size_t indices2[] = {
        0, 15,
        3
    };

    apply_moves(indices2, board_state);

    // We should be one turn shy of triggering the 50 move rule
    ASSERT_FALSE(board_state.can_claim_draw);

    int indices3[] = {1};

    apply_moves(indices3, board_state);

    ASSERT_TRUE(board_state.can_claim_draw);

    for(int i = 0; i < 12; i++) {
        // Moves rooks forward and backward, ending up with the same positions
        std::size_t rook_moves[] = {
            0, 15,
            3, 1
        };

        apply_moves(rook_moves, board_state);
    }

    int indices4[] = {0};

    apply_moves(indices4, board_state);

    // We should be one turn shy of triggering the 75 move rule
    ASSERT_NE(board_state.status, chess::game_status::draw);

    int indices5[] = {15};

    apply_moves(indices5, board_state);

    ASSERT_EQ(board_state.status, chess::game_status::draw);
}

