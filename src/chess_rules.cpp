#include "chess_rules.hpp"

#include <cstdlib>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <iterator>
#include <ranges>
#include <functional>
#include <optional>

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

    auto target_position_opt = apply_offset(position, offset);

    if(!target_position_opt) return false;

    auto [rank, file] = *target_position_opt;

    auto type = move_type::normal_move;

    if(board.pieces[rank][file].type != piece_type::none) {
        if(board.pieces[rank][file].piece_player == player) return false;
        if(is_pawn_move) return false; // Pawns capture diagonally, not forward
        type = move_type::capture;
        can_continue = false;
    }

    *it = {
        .type = type,
        .start_position = position,
        .target_position = *target_position_opt
    };

    return can_continue;
}

// Checks if the pawn at start_position can perform any en passant captures, and writes them to it
void get_en_passant_moves(const board_state& board, board_position start_position, std::back_insert_iterator<std::vector<chess_move>> it) {
    auto player = board.pieces[start_position.rank][start_position.file].piece_player;

    if(start_position.rank != (player == player::white ? 4 : 3)) return;

    std::uint8_t target_rank = player == player::white ? 5 : 2;

    int target_files[] = {
        start_position.file - 1,
        start_position.file + 1
    };

    for(auto file : target_files) {
        if(file < 0 || file > 7) continue;

        if(!board.en_passant_valid[file + (player == player::white ? 8 : 0)]) continue;

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
    if(position.file != 0 && position.file != 7) return {};

    auto player = board.pieces[position.rank][position.file].piece_player;

    bool kingside = position.file != 0;
    bool castling_allowed = board.can_castle[player == player::white ? (kingside ? 0 : 1) : (kingside ? 2 : 3)];

    if(!castling_allowed) return {};

    std::vector<chess_move> moves;
    std::uint8_t home_rank = player == player::white ? 0 : 7;

    if(kingside) {
        for(std::uint8_t file = 5; file <= 6; file++) {
            if(board.pieces[home_rank][file].type != piece_type::none) return {};
        }
    } else {
        for(std::uint8_t file = 1; file <= 3; file++) {
            if(board.pieces[home_rank][file].type != piece_type::none) return {};
        }
    }

    moves.push_back({
        .type = move_type::castle,
        .start_position = position
    });

    return moves;
}

std::vector<chess_move> get_rook_moves(const board_state& board, board_position position, std::size_t limit = 7) {
    std::vector<chess_move> moves;
    auto player = board.pieces[position.rank][position.file].piece_player;
    auto it = std::back_inserter(moves);
    board_offset offsets[] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    for(auto offset : offsets) {
        for(std::size_t i = 1; i <= limit; i++) {
            bool can_continue = check_position(
                board, player, position,
                {offset.rank_offset * static_cast<int>(i), offset.file_offset * static_cast<int>(i)}, false, it);
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
        for(std::size_t i = 1; i <= limit; i++) {
            bool can_continue = check_position(
                board, player, position,
                {static_cast<int>(i) * offset.rank_offset, static_cast<int>(i) * offset.file_offset}, false, it);
            if(!can_continue) break;
        }
    }
    return moves;
}

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
        {1, 2}, {1, -2}, {-1, 2}, {-1, -2},
        {2, 1}, {2, -1}, {-2, 1}, {-2, -1}
    };

    for(auto offset : offsets) {
        auto target_position_opt = apply_offset(position, offset);
        if(!target_position_opt) continue;
        auto [rank, file] = *target_position_opt;
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
            .target_position = *target_position_opt
        });
    }
    return moves;
}

std::vector<chess_move> get_pawn_moves(const board_state& board, board_position position) {
    std::vector<chess_move> moves;
    auto player = board.pieces[position.rank][position.file].piece_player;
    auto it = std::back_inserter(moves);
    bool is_promotion = (position.rank == 6 && player == player::white) || (position.rank == 1 && player == player::black);
    bool double_move_allowed = (position.rank == 1 && player == player::white) || (position.rank == 6 && player == player::black);
    int offset_multiplier = player == player::white ? 1 : -1;

    if(is_promotion) {
        board_position target_position_fwd = {
            .rank = static_cast<std::uint8_t>(position.rank + offset_multiplier),
            .file = position.file
        };
        if(in_bounds(target_position_fwd.rank, target_position_fwd.file) && board.pieces[target_position_fwd.rank][target_position_fwd.file].type == piece_type::none) {
            get_promotion_moves(position, target_position_fwd, it);
        }
    } else {
        for(int i = 1; i <= 2; i++) {
            board_offset offset = { .rank_offset = i * offset_multiplier, .file_offset = 0 };
            bool can_continue = check_position(board, player, position, offset, true, it);
            if(!can_continue || !double_move_allowed || i == 2) break;
        }
    }

    board_offset capture_offsets[] = {
        {.rank_offset = offset_multiplier, .file_offset = 1},
        {.rank_offset = offset_multiplier, .file_offset = -1}
    };

    for(auto offset : capture_offsets) {
        auto target_position_cap = apply_offset(position, offset);
        if(!target_position_cap) continue;
        auto [rank, file] = *target_position_cap;

        if(board.pieces[rank][file].type != piece_type::none && board.pieces[rank][file].piece_player != player) {
            if(is_promotion) {
                get_promotion_moves(position, *target_position_cap, it);
            } else {
                moves.push_back({
                    .type = move_type::capture,
                    .start_position = position,
                    .target_position = *target_position_cap
                });
            }
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
        if (piece.type == piece_type::rook) {
             auto castle_moves = get_castle_moves(board, position);
             std::ranges::copy(castle_moves, std::back_inserter(rook_moves));
        }
        return rook_moves;
    }
    case piece_type::queen: {
        return get_queen_moves(board, position);
    }
    case piece_type::king: {
        // King moves like Queen with limit 1
        return get_queen_moves(board, position, 1);
    }
    default:
        throw invalid_board_state_error{"Invalid piece type"};
    }
}

board_state board_state::initial_board_state() noexcept {
    board_state board = {};
    board.current_player = player::white;
    board.status = game_status::normal;
    board.turns_since_last_capture_or_pawn = 0;
    board.can_claim_draw = false;

    for(bool& b : board.can_castle) { b = true; }

    using enum piece_type;
    using enum player;
    piece_type rank1and8[8] = {rook, knight, bishop, queen, king, bishop, knight, rook};
    for(std::size_t i = 0; i < 8; i++) { board.pieces[0][i] = {rank1and8[i], white}; }
    for(auto& piece : board.pieces[1]) { piece = {pawn, white}; }
    for(auto& piece : board.pieces[6]) { piece = {pawn, black}; }
    for(std::size_t i = 0; i < 8; i++) { board.pieces[7][i] = {rank1and8[i], black}; }

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

    // Temporarily switch player to check attacks from opponent's perspective
    board.current_player = (player == player::white) ? player::black : player::white;

    // Check attacks from opponent's perspective
    for(std::uint8_t r = 0; r < 8; ++r) {
        for(std::uint8_t f = 0; f < 8; ++f) {
            piece p = board.pieces[r][f];
            if (p.type != piece_type::none && p.piece_player == board.current_player) {
                 auto moves_for_p = get_moves_for_piece_type(board, p, {r, f});
                 for(const auto& move : moves_for_p) {
                     if(move.type == move_type::capture || move.type == move_type::en_passant) { // Need to consider en passant revealing check indirectly too
                         if (move.target_position == *king_position) {
                              return true;
                         }
                     }
                 }
            }
        }
    }
    return false;
}


// Applies the move, but doesn't update the game status
board_state apply_move_impl(board_state board, chess_move move) {
    if (move.type < move_type::normal_move || move.type > move_type::resign) {
         throw invalid_move_error{"Invalid move type"};
    }

    if (move.type == move_type::resign) {
        board.status = game_status::resigned;
        board.current_player = (board.current_player == player::white) ? player::black : player::white;
        return board;
    }
    if (move.type == move_type::claim_draw) {
        if (!board.can_claim_draw) {
             return board; // Ignore invalid claim
        }
        board.status = game_status::draw;
        board.current_player = (board.current_player == player::white) ? player::black : player::white;
        return board;
    }


    bool is_capture_or_pawn_move = (move.type == move_type::capture ||
                                   move.type == move_type::en_passant ||
                                   move.type == move_type::promotion ||
                                   board.pieces[move.start_position.rank][move.start_position.file].type == piece_type::pawn);


    piece piece_to_move = board.pieces[move.start_position.rank][move.start_position.file];

    // --- Update Castling Rights ---
    if (piece_to_move.type == piece_type::king) {
        int offset = piece_to_move.piece_player == player::white ? 0 : 2;
        board.can_castle[offset] = false;
        board.can_castle[offset + 1] = false;
    } else if (piece_to_move.type == piece_type::rook) {
        int offset = piece_to_move.piece_player == player::white ? 0 : 2;
        std::uint8_t home_rank = (piece_to_move.piece_player == player::white) ? 0 : 7;
        if (move.start_position.rank == home_rank) {
            if (move.start_position.file == 0) board.can_castle[offset + 1] = false;
            else if (move.start_position.file == 7) board.can_castle[offset] = false;
        }
    }
    if (move.type == move_type::capture) {
         piece captured_piece = board.pieces[move.target_position.rank][move.target_position.file];
         if (captured_piece.type == piece_type::rook) {
             int offset = captured_piece.piece_player == player::white ? 0 : 2;
             std::uint8_t home_rank = (captured_piece.piece_player == player::white) ? 0 : 7;
              if (move.target_position.rank == home_rank) {
                   if (move.target_position.file == 0) board.can_castle[offset + 1] = false;
                   else if (move.target_position.file == 7) board.can_castle[offset] = false;
              }
         }
    }

    std::ranges::fill(board.en_passant_valid, false);

    // --- Apply Move Logic ---
    switch(move.type) {
        case move_type::capture: // Fallthrough
        case move_type::normal_move: {
            board.pieces[move.target_position.rank][move.target_position.file] = piece_to_move;
            board.pieces[move.start_position.rank][move.start_position.file] = {};

            if(piece_to_move.type == piece_type::pawn) {
                int rank_diff = move.target_position.rank - move.start_position.rank;
                if (std::abs(rank_diff) == 2) {
                    int index = move.start_position.file + (piece_to_move.piece_player == player::white ? 0 : 8);
                    board.en_passant_valid[index] = true;
                }
            }
            break;
        }
        case move_type::castle: {
            auto [rook_rank, rook_file] = move.start_position;
            player current_p = board.current_player;
            std::uint8_t king_rank = (current_p == player::white) ? 0 : 7;
            std::uint8_t king_start_file = 4;

            bool kingside = (rook_file == 7);
            std::uint8_t king_target_file = kingside ? 6 : 2;
            std::uint8_t rook_target_file = kingside ? 5 : 3;

            board.pieces[king_rank][king_target_file] = board.pieces[king_rank][king_start_file];
            board.pieces[king_rank][king_start_file] = {};
            board.pieces[rook_rank][rook_target_file] = board.pieces[rook_rank][rook_file];
            board.pieces[rook_rank][rook_file] = {};
            break;
        }
        case move_type::en_passant: {
            board.pieces[move.target_position.rank][move.target_position.file] = piece_to_move;
            board.pieces[move.start_position.rank][move.start_position.file] = {};

            int capture_rank_offset = (board.current_player == player::white) ? -1 : 1;
            std::uint8_t captured_pawn_rank = move.target_position.rank + capture_rank_offset;
            std::uint8_t captured_pawn_file = move.target_position.file;

            if (in_bounds(captured_pawn_rank, captured_pawn_file)) {
                 board.pieces[captured_pawn_rank][captured_pawn_file] = {};
            } else {
                 throw invalid_move_error{"Invalid en passant capture position"};
            }
            break;
        }
        case move_type::promotion: {
            if (move.promotion_target == piece_type::none || move.promotion_target == piece_type::pawn || move.promotion_target == piece_type::king) {
                 throw invalid_move_error{"Invalid promotion target piece type"};
            }
            board.pieces[move.target_position.rank][move.target_position.file] = {
                .type = move.promotion_target,
                .piece_player = board.current_player
            };
            board.pieces[move.start_position.rank][move.start_position.file] = {};
            break;
        }
        case move_type::resign:
        case move_type::claim_draw:
             break; // Should have been handled earlier
        default: throw invalid_move_error{"Invalid move type in apply_move_impl switch"};
    }

    board.current_player = (board.current_player == player::white) ? player::black : player::white;

    if(is_capture_or_pawn_move) {
        board.turns_since_last_capture_or_pawn = 0;
    } else {
        board.turns_since_last_capture_or_pawn++;
    }

    board.can_claim_draw = false;

    board.in_check[static_cast<int>(board.current_player)] = is_player_in_check(board, board.current_player);
    board.in_check[static_cast<int>(piece_to_move.piece_player)] = false; // Clear check for player who moved


    return board;
}

// Whether a particular move would put the player making it in check
bool puts_player_in_check(board_state board, chess_move move) {
    if (move.type == move_type::resign || move.type == move_type::claim_draw) return false;

    player player_making_move = board.current_player;
    board_state temp_board = apply_move_impl(board, move); // Simulate the move
    return is_player_in_check(temp_board, player_making_move);
}

// Updates the game status based on the current board state
void update_status(board_state& board) {
    if (board.status != game_status::normal) return;

    // --- Check for 50/75 Move Rule ---
    if (board.turns_since_last_capture_or_pawn >= 150) {
        board.status = game_status::draw;
        return;
    }
    if (board.turns_since_last_capture_or_pawn >= 100) {
        board.can_claim_draw = true;
    }

    // --- Check for Checkmate / Stalemate ---
    std::vector<chess_move> valid_moves_for_current_player = get_valid_moves(board);

    bool has_legal_move = false;
    for(const auto& mv : valid_moves_for_current_player) {
        if (mv.type != move_type::resign && mv.type != move_type::claim_draw) {
            has_legal_move = true;
            break;
        }
    }

    if (!has_legal_move) {
        if (board.in_check[static_cast<int>(board.current_player)]) {
            board.status = game_status::checkmate;
        } else {
            board.status = game_status::draw; // Stalemate
        }
        return;
    }

    // TODO: Check for threefold repetition (requires history)
    // TODO: Check for insufficient material (requires analyzing remaining pieces)
}


std::vector<chess_move> get_valid_moves(const board_state& board_state) {
    std::vector<chess_move> pseudo_legal_moves;

    for(std::uint8_t rank = 0; rank < 8; rank++) {
        for(std::uint8_t file = 0; file < 8; file++) {
            auto piece = board_state.pieces[rank][file];
            if(piece.type != piece_type::none && piece.piece_player == board_state.current_player) {
                auto moves = get_moves_for_piece_type(board_state, piece, {rank, file});
                pseudo_legal_moves.insert(pseudo_legal_moves.end(), moves.begin(), moves.end());
            }
        }
    }

    std::vector<chess_move> legal_moves;
    std::copy_if(pseudo_legal_moves.begin(), pseudo_legal_moves.end(),
                 std::back_inserter(legal_moves),
                 [&](const chess_move& move) {
                     return !puts_player_in_check(board_state, move);
                 });

    legal_moves.push_back({ .type = move_type::resign });
    if(board_state.can_claim_draw) {
        legal_moves.push_back({ .type = move_type::claim_draw });
    }

    return legal_moves;
}


board_state apply_move(board_state board, chess_move move) {
    board = apply_move_impl(board, move);
    update_status(board); // Update status after the move is fully applied
    return board;
}

} // namespace chess