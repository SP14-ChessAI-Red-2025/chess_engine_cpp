#include "chess_ai.hpp"

#include <ranges>
#include <utility>
#include <limits>
#include <algorithm>
#include <optional>
#include <cassert>

namespace chess::ai {

#ifdef NNUE_ENABLED
chess_ai_state::chess_ai_state(const char* model_path) : nnue_evaluator{model_path} {

}
#endif

using score_t = double;

// Whether the AI should consider the move
// The AI currently ignores resignations and claiming draws
bool should_consider_move(chess_move move) {
    return move.type != move_type::resign && move.type != move_type::claim_draw;
};

score_t rank_board_nnue(chess_ai_state& ai_state, const board_state& board, player player) {
    return ai_state.nnue_evaluator.evaluate(board);
}

// A positive result is good for player, negative is bad for player
score_t rank_board_old(const chess_ai_state& ai_state, const board_state& board, player player) {
    score_t board_value = 0;

    for(const auto& rank : board.pieces) {
        for(const auto& piece : rank) {
            score_t piece_value = 0;

            using enum piece_type;

            switch(piece.type) {
            case pawn:
                piece_value = 1;
                break;
            case knight:
                piece_value = 3;
                break;
            case bishop:
                piece_value = 3;
                break;
            case rook:
                piece_value = 5;
                break;
            case queen:
                piece_value = 9;
                break;
            default:
                break;
            }

            if(piece.piece_player == player) {
                board_value += piece_value;
            } else {
                board_value -= piece_value;
            }
        }
    }

    return board_value;
}

score_t rank_board(chess_ai_state& ai_state, const board_state& board, player player) {
    return rank_board_nnue(ai_state, board, player);
}

struct game_tree {
    chess_ai_state* ai_state;

    board_state current_state;

    enum player player; // The maximizing player

    std::optional<chess_move> move = {}; // The move that resulted in the current_state

    std::vector<game_tree> get_children() {
        std::vector<game_tree> children = {};

        for(auto& move : get_valid_moves(current_state) | std::views::filter(should_consider_move)) {
            auto board = apply_move(current_state, move);

            children.emplace_back(ai_state, board, player, move);
        }

        return children;
    }

    // Rank the current state according to the minimax algorithm
    score_t minimax(std::size_t depth, bool maximizing, score_t alpha, score_t beta) {
        if(depth == 0) {
            return rank_board(*ai_state, current_state, player);
        }

        auto children = get_children();

        if(maximizing) {
            score_t max_score = std::numeric_limits<decltype(max_score)>::min();

            for(auto child : children) {
                auto score = child.minimax(depth - 1, false, alpha, beta);

                max_score = std::max(score, max_score);

                if(score >= beta) {
                    break;
                }

                alpha = std::max(score, alpha);
            }

            return max_score;
        } else {
            score_t min_score = std::numeric_limits<decltype(min_score)>::max();

            for(auto child : children) {
                auto score = child.minimax(depth - 1, true, alpha, beta);

                min_score = std::min(score, min_score);

                if(score <= alpha) {
                    break;
                }

                beta = std::min(score, beta);
            }

            return min_score;
        }
    }

    board_state get_best_move(std::size_t depth) {
        assert(depth != 0);

        score_t alpha = std::numeric_limits<decltype(alpha)>::min();
        score_t beta = std::numeric_limits<decltype(beta)>::max();

        auto children = get_children();

        std::vector<std::pair<game_tree*, score_t>> children_with_scores;
        children_with_scores.reserve(children.size());

        std::ranges::transform(children, std::back_inserter(children_with_scores), [=](game_tree& child) {
            return std::make_pair(&child, child.minimax(depth - 1, false, alpha, beta));
        });

        auto projection = &std::pair<game_tree*, score_t>::second;

        return std::ranges::max_element(children_with_scores, {}, projection)->first->current_state;
    }
};

void chess_ai_state::make_move(board_state& board, std::int32_t difficulty) {
    if(board.status != game_status::normal) throw std::runtime_error{"Game is over"};
    if(difficulty == 0) throw std::runtime_error{"Difficulty must be at least 1"};

    game_tree tree{this, board, board.current_player};

    board = tree.get_best_move(difficulty);
}

} // namespace chess::ai
