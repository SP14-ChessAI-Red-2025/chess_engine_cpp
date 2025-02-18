#include "chess_cpp.hpp"

#include <cstdlib>

#include <vector>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <iterator>
#include <ranges>
#include <functional>
#include <optional>
#include <new>

namespace chess {

// An invalid board state should not be possible to create without manually editing the board state
class invalid_board_state_error : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

// An invalid move should not be possible to create without manually modifying a chess_move
class invalid_move_error : public std::logic_error {
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

    return {};
}

// Checks if the position determined by position + offset is in bounds
// If it is, writes the move to the iterator it
// is_pawn_move is true if the piece being moved is a pawn
// Returns true if the target position is empty, otherwise returns false
bool check_position(const board_state& board, player player, board_position position, board_offset offset, bool is_pawn_move, std::back_insert_iterator<std::vector<chess_move>> it) {
    bool can_continue = true;

    auto target_position = apply_offset(position, offset);

    if(!target_position) return false;

    auto [rank, file] = *target_position;

    auto type = move_type::normal_move;

    if(board.pieces[rank][file].type != piece_type::none) {
        // Encountered one of our own pieces, cannot move further
        if(board.pieces[rank][file].piece_player == player) return false;

        // This function is only called for pawns when they are moving forward, but pawns cannot capture while moving forward
        if(is_pawn_move) return false;

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
            bool can_continue = check_position(board, player, position, {offset.rank_offset * i, offset.file_offset * i}, false, it);

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
            bool can_continue = check_position(board, player, position, {i * offset.rank_offset, i * offset.file_offset}, false, it);

            if(!can_continue) break;
        }
    }

    return moves;
}

// Get moves for pieces that move like a queen (queen and king)
std::vector<chess_move> get_queen_moves(const board_state& board, board_position position, std::size_t limit = 7) {
    auto horizontal_moves = get_rook_moves(board, position, limit);
    auto diagonal_moves = get_bishop_moves(board, position, limit);

    std::ranges::copy(horizontal_moves, std::back_inserter(diagonal_moves));

    return diagonal_moves;
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

            bool can_continue = check_position(board, player, position, offset, true, it);

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
        return get_queen_moves(board, position);
    }
    case piece_type::king: {
        return get_queen_moves(board, position, 1);
    }
    default:
        // Invalid piece type
        throw invalid_board_state_error{"Invalid piece type"};
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

std::optional<board_position> get_king_position(const board_state& board, player player) {
    for(std::uint8_t rank = 0; rank <= 7; rank++) {
        for(std::uint8_t file = 0; file <= 7; file++) {
            auto& piece = board.pieces[rank][file];

            if(piece.type == piece_type::king && piece.piece_player == player) {
                return {{rank, file}};

            }
        }
    }

    return {};
}

// Whether player is in check
bool is_player_in_check(board_state board, player player) {
    std::optional<board_position> king_position = get_king_position(board, player);

    if(!king_position) throw invalid_board_state_error{"Invalid board state: no king"};

    std::vector<chess_move> threatening_moves; // Moves that would threaten the king

    auto it = std::back_inserter(threatening_moves);

    // Now we check if any enemy knights, bishops, rooks, queens, or kings will be able to capture us
    auto knight_moves = get_knight_moves(board, *king_position);
    auto bishop_moves = get_bishop_moves(board, *king_position);
    auto rook_moves = get_rook_moves(board, *king_position);
    auto queen_moves = get_queen_moves(board, *king_position);
    auto king_moves = get_queen_moves(board, *king_position, 1);


    // If a knight in this position would be able to capture an enemy knight, then we are vulnerable to that knight
    std::ranges::copy(knight_moves | std::views::filter([&board](chess_move move) {
        auto [rank, file] = move.target_position;
        return move.type == move_type::capture && board.pieces[rank][file].type == piece_type::knight;
    }), it);
    // Same for the other piece types, apart from pawns, which can only capture in one direction
    std::ranges::copy(bishop_moves | std::views::filter([&board](chess_move move) {
        auto [rank, file] = move.target_position;
        return move.type == move_type::capture && board.pieces[rank][file].type == piece_type::bishop;
    }), it);
    std::ranges::copy(rook_moves | std::views::filter([&board](chess_move move) {
        auto [rank, file] = move.target_position;
        return move.type == move_type::capture && board.pieces[rank][file].type == piece_type::rook;
    }), it);
    std::ranges::copy(queen_moves | std::views::filter([&board](chess_move move) {
        auto [rank, file] = move.target_position;
        return move.type == move_type::capture && board.pieces[rank][file].type == piece_type::queen;
    }), it);
    std::ranges::copy(king_moves | std::views::filter([&board](chess_move move) {
        auto [rank, file] = move.target_position;
        return move.type == move_type::capture && board.pieces[rank][file].type == piece_type::king;
    }), it);

    // Now check if any pawns can capture the king

    int offset_multiplier = player == player::white ? 1 : -1;

    // The locations that pawns could potentially attack us from
    board_offset pawn_offsets[] = {{offset_multiplier, 1}, {offset_multiplier, -1}};

    for(auto offset : pawn_offsets) {
        auto position = apply_offset(*king_position, offset);

        if(!position) continue;

        auto [rank, file] = *position;

        auto piece = board.pieces[rank][file];

        if(piece.type == piece_type::pawn && piece.piece_player != player) {
            return true;
        }
    }

    return !threatening_moves.empty();
}

board_state apply_move(board_state board, chess_move move) {
    bool is_capture_or_pawn_move = false;

    auto piece = board.pieces[move.start_position.rank][move.start_position.file];

    if(piece.type == piece_type::king || move.type == move_type::castle) {
        int offset = piece.piece_player == player::white ? 0 : 2;

        // Disable both kingside and queenside castling
        board.can_castle[offset] = false;
        board.can_castle[offset + 1] = false;
    }

    if(piece.type == piece_type::rook) {
        int offset = piece.piece_player == player::white ? 0 : 2;

        if(move.start_position.file == 0) {
            board.can_castle[offset + 1] = false; // Disable queenside castling
        } else if(move.start_position.file == 7) {
            board.can_castle[offset] = false; // Disable kingside castling
        }
    }

    std::ranges::fill(board.en_passant_valid, false);

    switch(move.type) {
        case move_type::resign:
            board.status = game_status::resigned;
            return board;
        case move_type::claim_draw:
            board.status = game_status::draw;
            return board;
        case move_type::capture:
            is_capture_or_pawn_move = true;
            [[fallthrough]];
        case move_type::normal_move: {
            board.pieces[move.start_position.rank][move.start_position.file] = {};
            board.pieces[move.target_position.rank][move.target_position.file] = piece;

            if(piece.type == piece_type::pawn) {
                if(move.start_position.rank == 1 && move.target_position.rank == 3) {
                    board.en_passant_valid[move.start_position.file] = true;
                }
                if(move.start_position.rank == 6 && move.target_position.rank == 4) {
                    board.en_passant_valid[8 + move.start_position.file] = true;
                }
            }

            break;
        }
        case move_type::castle: {
            auto [rank, file] = move.start_position;

            auto king_position = get_king_position(board, board.current_player);

            if(!king_position) throw invalid_board_state_error{"Invalid board state: no king"};

            assert(rank == king_position->rank);

            std::uint8_t king_target_file = file == 0 ? 2 : 6;
            std::uint8_t rook_target_file = file == 0 ? 3 : 5;

            board.pieces[rank][file] = {};
            board.pieces[rank][rook_target_file] = piece;
            board.pieces[rank][king_position->file] = {};
            board.pieces[rank][king_target_file] = {
                .type = piece_type::king,
                .piece_player = board.current_player
            };

            break;
        }
        case move_type::en_passant: {
            is_capture_or_pawn_move = true;

            board.pieces[move.start_position.rank][move.start_position.file] = {};
            board.pieces[move.target_position.rank][move.target_position.file] = piece;

            int capture_offset = board.current_player == player::white ? -1 : 1;

            board.pieces[move.target_position.rank + capture_offset][move.target_position.file] = {};

            break;
        }
        case move_type::promotion: {
            is_capture_or_pawn_move = true;

            board.pieces[move.start_position.rank][move.start_position.file] = {};
            board.pieces[move.target_position.rank][move.target_position.file] = {
                .type = move.promotion_target,
                .piece_player = board.current_player
            };
            break;
        }
        default: throw invalid_move_error{"Invalid move"};
    }

    // Toggle the current player
    board.current_player = board.current_player == player::white ? player::black : player::white;

    if(is_capture_or_pawn_move) {
        board.turns_since_last_capture_or_pawn = 0;
        board.can_claim_draw = false;
    } else {
        // A turn is finished when black makes their move
        // We already toggled the player, however, so we need to compare with player::white
        if(board.current_player == player::white) {
            board.turns_since_last_capture_or_pawn++;
        }
    }

    return board;
}

// Whether a particular move would put the player making it in check
bool puts_player_in_check(board_state board, chess_move move) {
    auto player = board.pieces[move.start_position.rank][move.start_position.file].piece_player;

    board = apply_move(board, move);

    return is_player_in_check(board, player);
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

    std::erase_if(valid_moves, std::bind_front(puts_player_in_check, board_state));

    valid_moves.push_back({
        .type = move_type::resign
    });

    if(board_state.can_claim_draw) {
        valid_moves.push_back({
            .type = move_type::claim_draw
        });
    }

    return valid_moves;
}

void update_status(board_state& board) {
    // get_valid_moves always returns at least 1 move of type move_type::resign
    if(get_valid_moves(board).size() == 1) {
        if(is_player_in_check(board, board.current_player)) {
            board.status = game_status::checkmate;
        } else {
            board.status = game_status::draw;
        }

        return;
    }

    if(board.turns_since_last_capture_or_pawn >= 50) {
        board.can_claim_draw = true;
    }

    // 75 move rule
    if(board.turns_since_last_capture_or_pawn >= 75) {
        board.status = game_status::draw;
    }

    // TODO: Support threefold repetition rule
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
    try {
        *board_state = apply_move(*board_state, move);
        update_status(*board_state);
    } catch(...) {
        // TODO: Return error to Python somehow
    }
}

void ai_move(board_state* board_state, std::int32_t difficulty) noexcept {
    // TODO: Implement
}

board_state get_initial_board_state() noexcept {
    return board_state::initial_board_state();
}
}
}