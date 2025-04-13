#include "chess_ai.hpp"

#include <ranges>
#include <utility>
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

    std::optional<chess_move> move; // The move that resulted in the current_state

    std::vector<game_tree> children;

    void deepen(std::size_t depth) {
        if(depth == 0) return;

        for(auto& move : get_valid_moves(current_state) | std::views::filter(should_consider_move)) {
            auto board = apply_move(current_state, move);

            children.emplace_back(board, move).deepen(depth - 1);
        }
    }

    // Rank the current state according to the minimax algorithm
    std::int32_t minimax(std::size_t depth, bool maximizing, player player) {
        if(depth == 0 || children.empty()) {
            return rank_board(current_state, player);
        }

        auto scores = children | std::views::transform([=](game_tree& child) {
            return child.minimax(depth - 1, !maximizing, player);
        });

        if(maximizing) {
            return *std::ranges::max_element(scores);
        } else {
            return *std::ranges::min_element(scores);
        }
    }

    game_tree* get_best_move(std::size_t depth) {
        assert(depth != 0);
        assert(!children.empty());

        auto children_with_scores = children | std::views::transform([=, this](game_tree& child) {
            return std::make_pair(&child, child.minimax(depth - 1, false, this->current_state.current_player));
        });

        auto projection = &decltype(children_with_scores[0])::second;

        return (*std::ranges::max_element(children_with_scores, {}, projection)).first;
    }
};

void chess_ai_state::make_move(board_state& board, std::int32_t difficulty) {
    game_tree tree{board};

    tree.deepen(3);

    auto move = tree.get_best_move(3)->move;
    assert(move.has_value());

    board = apply_move(board, *move);
}

} // namespace chess::ai