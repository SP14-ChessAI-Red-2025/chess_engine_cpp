#include "chess_cpp.hpp"

#include <cstdlib>

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iterator>

namespace chess {

struct board_offset {
    int rank_offset;
    int file_offset;
};

bool in_bounds(int rank, int file) {
    return !(rank < 0 || rank >= 8 || file < 0 || file >= 8);
}

// Checks if the position determined by position + offset is in bounds
// If it is, writes the move to the iterator it
// Returns true if the target position is empty, otherwise returns false
bool check_position(const board_state& board, player player, board_position position, board_offset offset, std::back_insert_iterator<std::vector<chess_move>> it) {
    bool can_continue = true;

    int rank = position.rank + offset.rank_offset;
    int file = position.file + offset.file_offset;

    if(!in_bounds(rank, file)) return false;

    auto type = move_type::normal_move;

    if(board.pieces[rank][file].type != piece_type::none) {
        // Encountered one of our own pieces, cannot move further
        if(board.pieces[rank][file].player == player) return false;

        // Encountered an enemy piece
        // We can capture it, but cannot move past it
        type = move_type::capture;

        can_continue = false;
    }

    *it = {
        .type = type,
        .start_position = position,
        .target_position = board_position(rank, file), // Rank and file were already bounds-checked, so narrowing conversion is ok
    };

    return can_continue;
}

std::vector<chess_move> get_rook_moves(const board_state& board, board_position position, std::size_t limit = 7) {
    std::vector<chess_move> moves;

    auto player = board.pieces[position.rank][position.file].player;

    auto it = std::back_inserter(moves);

    for(int i = 1; i <= limit; i++) {
        bool can_continue = check_position(board, player, position, {i, 0}, it);

        if(!can_continue) break;
    }

    for(int i = 1; i <= limit; i++) {
        bool can_continue = check_position(board, player, position, {-i, 0}, it);

        if(!can_continue) break;
    }

    for(int i = 1; i <= limit; i++) {
        bool can_continue = check_position(board, player, position, {0, i}, it);

        if(!can_continue) break;
    }

    for(int i = 1; i <= limit; i++) {
        bool can_continue = check_position(board, player, position, {0, -i}, it);

        if(!can_continue) break;
    }

    return moves;
}

std::vector<chess_move> get_bishop_moves(const board_state& board, board_position position, std::size_t limit = 7) {
    std::vector<chess_move> moves;

    auto player = board.pieces[position.rank][position.file].player;

    auto it = std::back_inserter(moves);

    for(int i = 1; i <= limit; i++) {
        bool can_continue = check_position(board, player, position, {i, i}, it);

        if(!can_continue) break;
    }

    for(int i = 1; i <= limit; i++) {
        bool can_continue = check_position(board, player, position, {-i, i}, it);

        if(!can_continue) break;
    }

    for(int i = 1; i <= limit; i++) {
        bool can_continue = check_position(board, player, position, {i, -i}, it);

        if(!can_continue) break;
    }

    for(int i = 1; i <= limit; i++) {
        bool can_continue = check_position(board, player, position, {-i, -i}, it);

        if(!can_continue) break;
    }

    return moves;
}

std::vector<chess_move> get_knight_moves(const board_state& board, board_position position) {
    std::vector<chess_move> moves;

    auto player = board.pieces[position.rank][position.file].player;

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

        auto target_piece = board.pieces[target_position.rank][target_position.file];

        if(target_piece.type == piece_type::none) {
            moves.push_back({
                .type = move_type::normal_move,
                .start_position = position,
                .target_position = target_position,
            });

            continue;
        }

        if(target_piece.player != player) {
            moves.push_back({
                .type = move_type::capture,
                .start_position = position,
                .target_position = target_position
            });
        }
    }


    return moves;
}

std::vector<chess_move> get_pawn_moves(const board_state& board, board_position position) {
    std::vector<chess_move> moves;

    auto player = board.pieces[position.rank][position.file].player;

    auto it = std::back_inserter(moves);

    // Moving 2 spaces is only allowed if the pawn is at its starting position
    bool double_move_allowed = (position.rank == 1 && player == player::white) || (position.rank == 6 && player == player::black);

    int offset_multiplier = player == player::white ? 1 : -1;  // Used to change the direction of the move if it is black's move

    for(int i = 1; i <= 2; i++) {
        board_offset offset = {
            .rank_offset = i * offset_multiplier
        };

        bool can_continue = check_position(board, player, position, offset, it);

        if(!can_continue || !double_move_allowed) break;
    }

    return moves;
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