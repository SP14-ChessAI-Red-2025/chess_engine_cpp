#include "chess_ai.hpp"

#include <ranges>
#include <functional>
#include <cassert>

namespace chess::ai {

std::int32_t rank_board(const board_state& board) {
    std::int32_t board_value = 0;

    for(const auto& rank : board.pieces) {
        for(const auto& piece : rank) {
            std::int32_t piece_value = 0;

            using enum piece_type;

            switch (piece.type) {
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

            if(piece.piece_player == board.current_player) {
                board_value += piece_value;
            }
        }
    }

    return board_value;
}

void chess_ai_state::make_move(board_state& board, std::int32_t difficulty) {
    auto valid_moves = get_valid_moves(board);

    auto next_states = valid_moves | std::views::filter([](auto move) {
        return move.type != move_type::resign;
    }) | std::views::transform(std::bind_front(apply_move, board));

    std::int32_t max_score = 0;
    const board_state* best_board = nullptr;

    for(const auto& next_state : next_states) {
        std::int32_t score = rank_board(next_state);

        if(score >= max_score) {
            max_score = score;
            best_board = &next_state;
        }
    }

    // Resignation is always a valid option, so this should never be null
    assert(best_board != nullptr);

    board = *best_board;
}

}