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
    std::vector<chess_move> moves;

    struct board_offset {
        int rank_offset;
        int file_offset;
    };

    board_offset offsets[] = {
        {1, 2},
        {1, -2},
        {-1, 2},
        {-1, -2},
        {2, 1},
        {2, -1},
        {-2, 1},
        {-2, -1}
    };

    for(auto offset : offsets) {
        int rank = position.rank + offset.rank_offset;
        int file = position.file + offset.file_offset;

        // Target position would be out of bounds
        if(rank < 0 || rank >= 8 || file < 0 || file >= 8) continue;

        board_position target_position = {
            static_cast<std::uint8_t>(rank),
            static_cast<std::uint8_t>(file)
        };

        auto player = board.pieces[position.rank][position.file].player;

        auto target_piece = board.pieces[target_position.rank][target_position.file];

        if(target_piece.type == piece_type::none) {
            moves.push_back(chess_move{
                .type = move_type::normal_move,
                .start_position = position,
                .target_position = target_position,
            });

            continue;
        }

        if(target_piece.player != player) {
            moves.push_back(chess_move{
                .type = move_type::capture,
                .start_position = position,
                .target_position = target_position
            });
        }
    }


    return moves;
}

std::vector<chess_move> get_pawn_moves(const board_state& board, board_position position) {
    // TODO: implement
    return {};
}

std::vector<chess_move> get_moves_for_piece_type(const board_state& board, piece piece, board_position position) {
    switch (piece.type) {
    case piece_type::pawn:
        return get_pawn_moves(board, position);
    case piece_type::knight:
        return get_knight_moves(board, position);
    case piece_type::bishop:
        return get_bishop_moves(board, position);
    case piece_type::rook:
        return get_rook_moves(board, position);
    case piece_type::queen: {
        auto horizontal_moves = get_rook_moves(board, position);
        auto diagonal_moves = get_bishop_moves(board, position);

        std::ranges::copy(horizontal_moves, std::back_inserter(diagonal_moves));

        return diagonal_moves;
    }
    case piece_type::king: {
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

board_state board_state::initial_board_state() {
    board_state board = {};

    for(bool& b : board.can_castle) {
        b = true;
    }

    using enum piece_type;
    using enum player;

    piece_type rank1and8[8] = {rook, knight, bishop, queen, king, bishop, knight, rook};

    for(std::size_t i = 0; i < 8; i++) {
        board.pieces[0][i] = {rank1and8[i], white};
    }

    for(auto& piece : board.pieces[1]) {
        piece = {pawn, white};
    }

    for(auto& piece : board.pieces[6]) {
        piece = {pawn, black};
    }

    for(std::size_t i = 0; i < 8; i++) {
        board.pieces[7][i] = {rank1and8[i], black};
    }

    return board;
}



namespace python {
chess_move* get_valid_moves(board_state board_state, std::size_t* num_moves) {
    std::vector<chess_move> valid_moves = {};

    for(std::uint8_t rank = 0; rank < 8; rank++) {
        for(std::uint8_t file = 0; file < 8; file++) {
            auto piece = board_state.pieces[rank][file];
            board_position position = {
                .rank = rank,
                .file = file
            };
            if(piece.type != piece_type::none && board_state.pieces[rank][file].player == board_state.current_player) {
                auto moves = get_moves_for_piece_type(board_state, piece, position);

                std::ranges::copy(moves, std::back_inserter(valid_moves));
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

void apply_move(board_state* board_state, chess_move move) {
    // TODO: Implement
}

board_state get_initial_board_state() {
    return board_state::initial_board_state();
}
}
}