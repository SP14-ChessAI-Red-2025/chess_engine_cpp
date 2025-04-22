#pragma once

#include "chess_rules.hpp" // Use the definitions from chess_rules.hpp
#include "nnue_evaluator.hpp"
#include <vector>
#include <memory>
#include <optional>
#include <cstdint>
#include <string>
#include <limits>
#include <map> // Add this to use std::map

#ifndef CHESS_AI_HPP
#define CHESS_AI_HPP

namespace chess::ai {

    extern const std::map<piece_type, int> piece_values; // Declare as extern

    struct game_tree;

    struct game_tree {
        chess::board_state current_state;
        double score;
        std::optional<chess::chess_move> move;
        std::vector<std::unique_ptr<game_tree>> children;

        game_tree(const chess::board_state& state);

        /**
         * @brief Deepen the search tree using Alpha-Beta pruning.
         * @param depth Remaining search depth.
         * @param evaluator Reference to the evaluator.
         * @param alpha Best score maximizer can guarantee.
         * @param beta Best score minimizer can guarantee.
         * @return Score of this node.
         */
        double deepen(std::size_t depth, NNUEEvaluator& evaluator, double alpha, double beta);

        void add_child(const chess_move& m, const board_state& next_state);
    };

    struct chess_ai_state {
        std::unique_ptr<game_tree> root;
        NNUEEvaluator evaluator_;

        explicit chess_ai_state(const std::string& model_path);
        void make_move(board_state& board_state, std::int32_t difficulty);
    };

    bool should_consider_move(const chess_move& move);
    bool are_states_equal(const board_state& s1, const board_state& s2);
    bool is_square_attacked(const board_state& board, const board_position& square, player opponent_color);
    bool threatens_high_value_piece(const board_state& board, const chess_move& move);

} // namespace chess::ai

#endif // CHESS_AI_HPP