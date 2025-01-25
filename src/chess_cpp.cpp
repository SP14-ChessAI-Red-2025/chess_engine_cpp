#include "chess_cpp.hpp"

#include <cstdlib>

#include <vector>
#include <algorithm>
#include <stdexcept>

namespace chess {

std::vector<chess_move> get_rook_moves(const board_state& board, board_position position, std::size_t limit = 7) {
    // TODO: implement
    return {};
}

std::vector<chess_move> get_bishop_moves(const board_state& board, board_position position, std::size_t limit = 7) {
    // TODO: implement
    return {};
}

std::vector<chess_move> get_knight_moves(const board_state& board, board_position position) {
    // TODO: implement
    return {};
}

std::vector<chess_move> get_pawn_moves(const board_state& board, board_position position) {
    // TODO: implement
    return {};
}

std::vector<chess_move> get_moves_for_piece_type(const board_state& board, piece piece, board_position position) {
    switch (piece.type) {
    case piece_type::Pawn:
        return get_pawn_moves(board, position);
    case piece_type::Knight:
        return get_knight_moves(board, position);
    case piece_type::Bishop:
        return get_bishop_moves(board, position);
    case piece_type::Rook:
        return get_rook_moves(board, position);
    case piece_type::Queen: {
        auto horizontal_moves = get_rook_moves(board, position);
        auto diagonal_moves = get_bishop_moves(board, position);

        std::ranges::copy(horizontal_moves, std::back_inserter(diagonal_moves));

        return diagonal_moves;
    }
    case piece_type::King: {
        auto horizontal_moves = get_rook_moves(board, position, 1);
        auto diagonal_moves = get_bishop_moves(board, position, 1);

        std::ranges::copy(horizontal_moves, std::back_inserter(diagonal_moves));

        return diagonal_moves;
    }
    default:
        // Invalid piece type
        throw std::invalid_argument{"Invalid piece type"};
    }
}


namespace python {
chess_move* get_valid_moves(board_state board_state, std::size_t* num_moves) {
    std::vector<chess_move> valid_moves = {chess::chess_move{
        .start_position = {1, 2},
        .target_position = {3, 4}
    }};

    for(std::uint8_t rank = 0; rank < 8; rank++) {
        for(std::uint8_t file = 0; file < 8; file++) {
            auto piece = board_state.pieces[rank][file];
            board_position position = {
                .rank = rank,
                .file = file
            };
            if(piece.type != piece_type::None && board_state.pieces[rank][file].player == board_state.current_player) {
                std::ranges::copy(get_moves_for_piece_type(board_state, piece, position), std::back_inserter(valid_moves));
            }
        }
    }

    *num_moves = valid_moves.size();

    auto* result = new chess_move[valid_moves.size()];

    std::ranges::copy(valid_moves, result);

    return result;
}

void free_moves(chess_move* moves) {
    delete[] moves;
}

board_state get_initial_board_state() {
    return {};
}
}
}