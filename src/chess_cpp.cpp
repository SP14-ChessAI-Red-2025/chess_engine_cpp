#include "chess_cpp.hpp"

#include <cstdlib>

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iterator>
#include <optional>
#include <new>

namespace chess {

// An invalid board state should not be possible to create without manually editing the board state
class invalid_board_state_error : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

struct board_offset {
    int rank_offset;
    int file_offset;
};

bool in_bounds(int rank, int file) {
    return !(rank < 0 || rank >= 8 || file < 0 || file >= 8);
}

std::optional<board_position> apply_offset(board_position position, board_offset offset) {
    int rank = position.rank + offset.rank_offset;
    int file = position.file + offset.file_offset;

    if(in_bounds(rank, file)) {
        return {{
            .rank = static_cast<std::uint8_t>(rank),
            .file = static_cast<std::uint8_t>(file)
        }};
    }

    return std::nullopt;
}

// Checks if the position determined by position + offset is in bounds
// If it is, writes the move to the iterator it
// Returns true if the target position is empty, otherwise returns false
bool check_position(const board_state& board, player player, board_position position, board_offset offset, std::back_insert_iterator<std::vector<chess_move>> it) {
    bool can_continue = true;

    auto target_position = apply_offset(position, offset);

    if(!target_position) return false;

    auto [rank, file] = *target_position;

    auto type = move_type::normal_move;

    if(board.pieces[rank][file].type != piece_type::none) {
        // Encountered one of our own pieces, cannot move further
        if(board.pieces[rank][file].piece_player == player) return false;

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

// Checks if the pawn at start_position can perform any en passant captures, and writes them to it
void get_en_passant_moves(const board_state& board, board_position start_position, std::back_insert_iterator<std::vector<chess_move>> it) {
    auto player = board.pieces[start_position.rank][start_position.file].piece_player;

    // Check if pawn is on the correct rank to perform an en passant capture
    if(start_position.rank != (player == player::white ? 4 : 3)) return;

    std::uint8_t target_rank = player == player::white ? 5 : 2; // The rank we move to

    int target_files[] = {
        start_position.file - 1,
        start_position.file + 1
    };

    for(auto file : target_files) {
        if(file < 0 || file > 7) continue;

        if(!board.en_passant_valid[file + (player == player::white ? 8 : 0)]) continue;

        // en_passant_valid is only set to true immediately after the enemy pawn moves,
        // so we don't need to check if pieces[capture_rank][file] contains an enemy piece

        *it = {
            .type = move_type::en_passant,
            .start_position = start_position,
            .target_position = {
                .rank = target_rank,
                .file = static_cast<std::uint8_t>(file)
            }
        };
    }
}

// Writes 4 chess_move instances, one for each possible promotion target, to it
void get_promotion_moves(board_position start_position, board_position target_position, std::back_insert_iterator<std::vector<chess_move>> it) {
    piece_type promotion_targets[] = {piece_type::queen, piece_type::rook, piece_type::bishop, piece_type::knight};

    for(auto promotion_target : promotion_targets) {
        *it = {
            .type = move_type::promotion,
            .start_position = start_position,
            .target_position = target_position,
            .promotion_target = promotion_target
        };
    }
}

std::vector<chess_move> get_castle_moves(const board_state& board, board_position position) {
    // Can only castle if the rook is in the starting file
    if(position.file != 0 && position.file != 7) return {};

    auto player = board.pieces[position.rank][position.file].piece_player;

    bool kingside = position.file != 0;

    bool castling_allowed = board.can_castle[player == player::white ? (kingside ? 0 : 1) : (kingside ? 2 : 3)];

    if(!castling_allowed) return {};

    std::vector<chess_move> moves;

    std::uint8_t home_rank = player == player::white ? 0 : 7;

    if(kingside) {
        for(std::uint8_t file = 5; file <= 6; file++) {
            if(board.pieces[home_rank][file].type != piece_type::none) {
                castling_allowed = false;
                break;
            }
        }
    } else {
        for(std::uint8_t file = 1; file <= 3; file++) {
            if(board.pieces[home_rank][file].type != piece_type::none) {
                castling_allowed = false;
                break;
            }
        }
    }

    if(castling_allowed) {
        moves.push_back({
            .type = move_type::castle,
            .start_position = position
        });
    }

    return moves;
}

std::vector<chess_move> get_rook_moves(const board_state& board, board_position position, std::size_t limit = 7) {
    std::vector<chess_move> moves;

    auto player = board.pieces[position.rank][position.file].piece_player;

    auto it = std::back_inserter(moves);

    board_offset offsets[] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    for(auto offset : offsets) {
        for(int i = 1; i <= limit; i++) {
            bool can_continue = check_position(board, player, position, {offset.rank_offset * i, offset.file_offset * i}, it);

            if(!can_continue) break;
        }
    }

    return moves;
}

std::vector<chess_move> get_bishop_moves(const board_state& board, board_position position, std::size_t limit = 7) {
    std::vector<chess_move> moves;

    auto player = board.pieces[position.rank][position.file].piece_player;

    auto it = std::back_inserter(moves);

    board_offset offsets[] = {{1, 1}, {-1, 1}, {1, -1}, {-1, -1}};

    for(auto offset : offsets) {
        for(int i = 1; i <= limit; i++) {
            bool can_continue = check_position(board, player, position, {i * offset.rank_offset, i * offset.file_offset}, it);

            if(!can_continue) break;
        }
    }

    return moves;
}

std::vector<chess_move> get_knight_moves(const board_state& board, board_position position) {
    std::vector<chess_move> moves;

    auto player = board.pieces[position.rank][position.file].piece_player;

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
        auto target_position = apply_offset(position, offset);

        if(!target_position) continue;

        auto [rank, file] = *target_position;

        auto target_piece = board.pieces[rank][file];

        move_type type;

        if(target_piece.type == piece_type::none) {
            type = move_type::normal_move;
        } else if(target_piece.piece_player != player) {
            type = move_type::capture;
        } else {
            continue;
        }

        moves.push_back({
            .type = type,
            .start_position = position,
            .target_position = *target_position
        });
    }


    return moves;
}

std::vector<chess_move> get_pawn_moves(const board_state& board, board_position position) {
    std::vector<chess_move> moves;

    auto player = board.pieces[position.rank][position.file].piece_player;

    auto it = std::back_inserter(moves);

    // Whether the move in question is a pawn promotion
    bool is_promotion = (position.rank == 6 && player == player::white) || (position.rank == 1 && player == player::black);

    // Moving 2 spaces is only allowed if the pawn is at its starting position
    bool double_move_allowed = (position.rank == 1 && player == player::white) || (position.rank == 6 && player == player::black);

    int offset_multiplier = player == player::white ? 1 : -1;  // Used to change the direction of the move if it is black's move

    if(is_promotion) {
        board_position target_position = {
            // position.rank + offset_multiplier can only be 0 or 7 here, so this cast is always safe
            .rank = static_cast<std::uint8_t>(position.rank + offset_multiplier),
            .file = position.file
        };

        if(board.pieces[target_position.rank][target_position.file].type == piece_type::none) {
            get_promotion_moves(position, target_position, it);
        }
    } else {
        for(int i = 1; i <= 2; i++) {
            board_offset offset = {
                .rank_offset = i * offset_multiplier
            };

            bool can_continue = check_position(board, player, position, offset, it);

            if(!can_continue || !double_move_allowed) break;
        }
    }

    board_offset capture_offsets[] = {
        {
            .rank_offset = offset_multiplier,
            .file_offset = 1
        }, {
            .rank_offset = offset_multiplier,
            .file_offset = -1
        }
    };

    for(auto offset : capture_offsets) {
        auto target_position = apply_offset(position, offset);

        if(!target_position) continue;

        auto [rank, file] = *target_position;

        if(board.pieces[rank][file].type != piece_type::none && board.pieces[rank][file].piece_player != player) {
            moves.push_back({
                .type = is_promotion ? move_type::promotion : move_type::capture,
                .start_position = position,
                .target_position = *target_position
            });
        }
    }

    get_en_passant_moves(board, position, it);

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
    case piece_type::rook: {
        auto rook_moves = get_rook_moves(board, position);
        auto castle_moves = get_castle_moves(board, position);

        std::ranges::copy(castle_moves, std::back_inserter(rook_moves));

        return rook_moves;
    }
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

board_state board_state::initial_board_state() noexcept {
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

std::vector<chess_move> get_valid_moves(const board_state& board_state) {
    std::vector<chess_move> valid_moves = {};

    for(std::uint8_t rank = 0; rank < 8; rank++) {
        for(std::uint8_t file = 0; file < 8; file++) {
            auto piece = board_state.pieces[rank][file];

            board_position position = {
                .rank = rank,
                .file = file
            };

            if(piece.type != piece_type::none && board_state.pieces[rank][file].piece_player == board_state.current_player) {
                auto moves = get_moves_for_piece_type(board_state, piece, position);

                std::ranges::copy(moves, std::back_inserter(valid_moves));
            }
        }
    }

    return valid_moves;
}



namespace python {
void* init_ai_state() noexcept {
    return new(std::nothrow) chess_ai_state{};
}

void free_ai_state(void* state) noexcept {
    delete static_cast<chess_ai_state*>(state);
}

chess_move* get_valid_moves(board_state board_state, std::size_t* num_moves) noexcept {
    std::vector<chess_move> valid_moves;

    try {
        valid_moves = get_valid_moves(board_state);
    } catch(...) {
        // TODO: Return error to Python somehow
        return nullptr;
    }

    *num_moves = valid_moves.size();

    auto* result = new(std::nothrow) chess_move[valid_moves.size()];

    if(result) std::ranges::copy(valid_moves, result);

    return result;
}

void free_moves(chess_move* moves) noexcept {
    delete[] moves;
}

void apply_move(board_state* board_state, chess_move move) noexcept {
    // TODO: Implement
}

void ai_move(board_state* board_state, std::int32_t difficulty) noexcept {
    // TODO: Implement
}

board_state get_initial_board_state() noexcept {
    return board_state::initial_board_state();
}
}
}