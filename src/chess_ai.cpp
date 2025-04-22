#include "chess_ai.hpp"

#include <ranges>
#include <utility>
#include <limits>
#include <algorithm>
#include <optional>
#include <cassert>

namespace chess::ai {

// Whether the AI should consider the move
// The AI currently ignores resignations and claiming draws
bool should_consider_move(chess_move move) {
    return move.type != move_type::resign && move.type != move_type::claim_draw;
};

// A positive result is good for player, negative is bad for player
std::int32_t rank_board(const board_state& board, player player) {
    std::int32_t board_value = 0;

    for(const auto& rank : board.pieces) {
        for(const auto& piece : rank) {
            std::int32_t piece_value = 0;

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

struct game_tree {
    board_state current_state;

    enum player player; // The maximizing player

    std::optional<chess_move> move = {}; // The move that resulted in the current_state

    std::vector<game_tree> get_children() {
        std::vector<game_tree> children = {};

        for(auto& move : get_valid_moves(current_state) | std::views::filter(should_consider_move)) {
            auto board = apply_move(current_state, move);

            children.emplace_back(board, player, move);
        }

        return children;
    }

    // Rank the current state according to the minimax algorithm
    std::int32_t minimax(std::size_t depth, bool maximizing, std::int32_t alpha, std::int32_t beta) {
        if(depth == 0) {
            return rank_board(current_state, player);
        }

        auto children = get_children();

        if(maximizing) {
            std::int32_t max_score = std::numeric_limits<decltype(max_score)>::min();

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
            std::int32_t min_score = std::numeric_limits<decltype(min_score)>::max();

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

        std::int32_t alpha = std::numeric_limits<decltype(alpha)>::min();
        std::int32_t beta = std::numeric_limits<decltype(beta)>::max();

        auto children = get_children();

        std::vector<std::pair<game_tree*, std::int32_t>> children_with_scores;
        children_with_scores.reserve(children.size());

        std::ranges::transform(children, std::back_inserter(children_with_scores), [=](game_tree& child) {
            return std::make_pair(&child, child.minimax(depth - 1, false, alpha, beta));
        });

        auto projection = &std::pair<game_tree*, std::int32_t>::second;

        return std::ranges::max_element(children_with_scores, {}, projection)->first->current_state;
    }
};

void chess_ai_state::make_move(board_state& board, std::int32_t difficulty) {
    if(board.status != game_status::normal) throw std::runtime_error{"Game is over"};
    if(difficulty == 0) throw std::runtime_error{"Difficulty must be at least 1"};

    game_tree tree{board, board.current_player};

    board = tree.get_best_move(difficulty);
}

} // namespace chess::ai
