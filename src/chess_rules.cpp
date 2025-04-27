// src/chess_rules.cpp
#include "chess_cpp/chess_rules.hpp"
#include "chess_cpp/chess_ai.hpp" // For is_square_attacked
#include "chess_cpp/threefold_repetition.hpp"

#include <cstdlib>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <iterator>
#include <vector>
#include <cmath>
#include <optional>
#include <iostream>
#include <functional>
#include <set>

// Forward declaration (already in header, but safe)
namespace chess::ai {
    bool is_square_attacked(const board_state& board, const board_position& square, player attacking_player);
}

namespace chess {

    // --- Helper Functions ---
    bool in_bounds(int rank, int file) {
        return rank >= 0 && rank < 8 && file >= 0 && file < 8;
    }

    std::optional<board_position> apply_offset(board_position position, board_offset offset) {
        int new_rank = static_cast<int>(position.rank) + offset.rank_offset;
        int new_file = static_cast<int>(position.file) + offset.file_offset;
        if (in_bounds(new_rank, new_file)) {
            return board_position{static_cast<uint8_t>(new_rank), static_cast<uint8_t>(new_file)};
        }
        return std::nullopt;
    }

    bool is_player_in_check(const board_state& board, player p) {
        board_position king_pos = {0, 0};
        bool king_found = false;
        for (uint8_t r = 0; r < 8 && !king_found; ++r) {
            for (uint8_t f = 0; f < 8; ++f) {
                if (board.pieces[r][f].type == piece_type::king && board.pieces[r][f].piece_player == p) {
                    king_pos = {r, f};
                    king_found = true;
                    break;
                }
            }
        }
        if (!king_found) return false; // Should not happen in valid state
        return chess::ai::is_square_attacked(board, king_pos, (p == player::white) ? player::black : player::white);
    }

    // --- apply_move_impl ---
    board_state apply_move_impl(board_state board, chess_move move) {
         if (!in_bounds(move.start_position.rank, move.start_position.file) ||
             !in_bounds(move.target_position.rank, move.target_position.file)) {
              throw std::out_of_range("Move position out of bounds in apply_move_impl");
         }
        piece moved_piece = board.pieces[move.start_position.rank][move.start_position.file];
        piece captured_piece = board.pieces[move.target_position.rank][move.target_position.file];

        if (move.type == move_type::resign || move.type == move_type::claim_draw) {
             throw std::logic_error("apply_move_impl called with resign/claim_draw");
        }
        else if (move.type == move_type::castle) {
            int king_start_file = 4;
            int king_target_file = (move.target_position.file > king_start_file) ? 6 : 2;
            board_position king_start = {move.start_position.rank, static_cast<uint8_t>(king_start_file)};
            board_position king_target = {move.start_position.rank, static_cast<uint8_t>(king_target_file)};
            board.pieces[king_target.rank][king_target.file] = board.pieces[king_start.rank][king_start.file];
            board.pieces[king_start.rank][king_start.file] = {};
            int rook_start_file = (king_target_file == 6) ? 7 : 0;
            int rook_target_file = (king_target_file == 6) ? 5 : 3;
            board_position rook_start = {move.start_position.rank, static_cast<uint8_t>(rook_start_file)};
            board_position rook_target = {move.start_position.rank, static_cast<uint8_t>(rook_target_file)};
            board.pieces[rook_target.rank][rook_target.file] = board.pieces[rook_start.rank][rook_start.file];
            board.pieces[rook_start.rank][rook_start.file] = {};
            int castle_idx_offset = (moved_piece.piece_player == player::white) ? 0 : 2;
            board.can_castle[castle_idx_offset] = false;
            board.can_castle[castle_idx_offset + 1] = false;
        } else { // Normal, Capture, Promotion, En Passant
            for(auto& valid : board.en_passant_valid) { valid = false; } // Reset EP flags first
            board.pieces[move.start_position.rank][move.start_position.file] = {}; // Clear start
            // Place piece (handle promotion)
            board.pieces[move.target_position.rank][move.target_position.file] =
                (move.type == move_type::promotion) ? piece{move.promotion_target, moved_piece.piece_player} : moved_piece;
            // Handle EP capture
            if (move.type == move_type::en_passant) {
                int captured_pawn_rank = (moved_piece.piece_player == player::white) ? 4 : 3;
                board.pieces[captured_pawn_rank][move.target_position.file] = {};
                captured_piece = {piece_type::pawn, (moved_piece.piece_player == player::white ? player::black : player::white)};
            }
            // Set new EP validity if double pawn push
            if (moved_piece.type == piece_type::pawn && std::abs(static_cast<int>(move.target_position.rank) - static_cast<int>(move.start_position.rank)) == 2) {
                 int en_passant_target_rank = (moved_piece.piece_player == player::white) ? 2 : 5;
                 int en_passant_index = (en_passant_target_rank == 2) ? move.target_position.file : (8 + move.target_position.file);
                 if (en_passant_index >= 0 && en_passant_index < 16) board.en_passant_valid[en_passant_index] = true;
            }
            // Update castling rights if King/Rook moved or captured
            if (moved_piece.type == piece_type::king) {
                 int castle_idx_offset = (moved_piece.piece_player == player::white) ? 0 : 2;
                 board.can_castle[castle_idx_offset] = false; board.can_castle[castle_idx_offset + 1] = false;
            } else if (moved_piece.type == piece_type::rook) {
                int start_rank = (moved_piece.piece_player == player::white ? 0 : 7);
                if (move.start_position.rank == start_rank) {
                    int castle_idx_offset = (moved_piece.piece_player == player::white) ? 0 : 2;
                    if (move.start_position.file == 7) board.can_castle[castle_idx_offset] = false;
                    if (move.start_position.file == 0) board.can_castle[castle_idx_offset + 1] = false;
                }
            }
             if (captured_piece.type == piece_type::rook) {
                  int captured_start_rank = (captured_piece.piece_player == player::white ? 0 : 7);
                   if (move.target_position.rank == captured_start_rank) {
                       int captured_castle_idx_offset = (captured_piece.piece_player == player::white) ? 0 : 2;
                       if (move.target_position.file == 7) board.can_castle[captured_castle_idx_offset] = false;
                       if (move.target_position.file == 0) board.can_castle[captured_castle_idx_offset + 1] = false;
                   }
             }
        }

        // Update 50-move counter
        if (moved_piece.type == piece_type::pawn || captured_piece.type != piece_type::none) {
            board.turns_since_last_capture_or_pawn = 0;
        } else {
            board.turns_since_last_capture_or_pawn++;
        }
        // Switch player
        // std::cerr << "DEBUG C++ Rules: current_player: " << static_cast<int>(board.current_player) << std::endl;
        board.current_player = (board.current_player == player::white) ? player::black : player::white;
        // std::cerr << "DEBUG C++ Rules: current_player: " << static_cast<int>(board.current_player) << std::endl;
        // Update check status for both players
        board.in_check[static_cast<int>(player::white)] = is_player_in_check(board, player::white);
        board.in_check[static_cast<int>(player::black)] = is_player_in_check(board, player::black);

        return board;
    }

    // --- operator== for board_state ---
    bool operator==(const board_state& lhs, const board_state& rhs) {
        for (int r = 0; r < 8; ++r) for (int f = 0; f < 8; ++f) {
            if (lhs.pieces[r][f].type != rhs.pieces[r][f].type || lhs.pieces[r][f].piece_player != rhs.pieces[r][f].piece_player) return false;
        }
        if (lhs.current_player != rhs.current_player) return false;
        for (int i = 0; i < 4; ++i) if (lhs.can_castle[i] != rhs.can_castle[i]) return false;
        for (int i = 0; i < 16; ++i) if (lhs.en_passant_valid[i] != rhs.en_passant_valid[i]) return false;
        return true;
    }

    // --- update_status Function (with History) ---
    void update_status(board_state& board, previous_board_states& history) {
         if (board.status != game_status::normal) return;
         if (board.turns_since_last_capture_or_pawn >= 150) { board.status = game_status::draw; return; }
         auto it_rep = history.position_counts.find(board);
         if (it_rep != history.position_counts.end() && it_rep->second >= 5) { board.status = game_status::draw_by_repetition; return; }
         board.can_claim_draw = (board.turns_since_last_capture_or_pawn >= 100) || (history.draw_allowed && it_rep != history.position_counts.end() && it_rep->second >= 3);

         std::vector<chess_move> valid_moves = get_valid_moves(board);
         bool has_legal_gameplay_move = false;
         for(const auto& mv : valid_moves) { if (mv.type != move_type::resign && mv.type != move_type::claim_draw) { has_legal_gameplay_move = true; break; } }

         if (!has_legal_gameplay_move) {
             board.status = board.in_check[static_cast<int>(board.current_player)] ? game_status::checkmate : game_status::draw; // Stalemate
             board.can_claim_draw = false;
         }
    }

    // --- apply_move (with History) ---
    void apply_move(board_state& board, chess_move move, previous_board_states& history) {
        if (move.type == move_type::claim_draw) {
            if (board.can_claim_draw) board.status = game_status::draw; // Use generic draw type
            return;
        }
         if (move.type == move_type::resign) {
              board.status = game_status::resigned; board.can_claim_draw = false;
              return;
         }
        // Get info before clearing history
        piece moved_piece = {}; bool is_capture = false;
        if (in_bounds(move.start_position.rank, move.start_position.file)) moved_piece = board.pieces[move.start_position.rank][move.start_position.file]; else throw std::logic_error("apply_move invalid start");
        if (in_bounds(move.target_position.rank, move.target_position.file)) is_capture = (board.pieces[move.target_position.rank][move.target_position.file].type != piece_type::none) || (move.type == move_type::en_passant); else throw std::logic_error("apply_move invalid target");

        if (moved_piece.type == piece_type::pawn || is_capture) {
            history.clear_history_on_irreversible_move(move, moved_piece);
        }
        // Apply move using internal helper (modifies board by reference now)
        board = apply_move_impl(board, move); // Overwrite board with result
        // std::cerr << "DEBUG C++ Rules: current_player: " << static_cast<int>(board.current_player) << std::endl;

        // Add state to history AFTER move is applied
        if (board.status == game_status::normal) { // Add only if game continues
            history.add_board_state(board);
        }
        // Update status based on new state and history
        update_status(board, history);
   }

    // --- Generate Valid Moves ---
    std::vector<chess_move> get_valid_moves(const board_state& board) {
        std::vector<chess_move> pseudo_legal_moves;
        player current_p = board.current_player;
        player opponent_p = (current_p == player::white) ? player::black : player::white;
        // std::cerr << "DEBUG C++ Rules: get_valid_moves called for player " << static_cast<int>(current_p) << std::endl;

        auto add_promotions = [&](const board_position& start, const board_position& target, move_type base_type) {
            pseudo_legal_moves.push_back({base_type, start, target, piece_type::queen}); pseudo_legal_moves.push_back({base_type, start, target, piece_type::rook});
            pseudo_legal_moves.push_back({base_type, start, target, piece_type::bishop}); pseudo_legal_moves.push_back({base_type, start, target, piece_type::knight});
        };
        auto generate_sliding_moves = [&](const board_position& start, const std::vector<board_offset>& directions) {
            for (const auto& dir : directions) for (int i = 1; ; ++i) {
                auto target_opt = apply_offset(start, {dir.rank_offset * i, dir.file_offset * i}); if (!target_opt) break;
                piece target_piece = board.pieces[target_opt->rank][target_opt->file];
                if (target_piece.type == piece_type::none) pseudo_legal_moves.push_back({move_type::normal_move, start, *target_opt});
                else { if (target_piece.piece_player == opponent_p) pseudo_legal_moves.push_back({move_type::capture, start, *target_opt}); break; }
            }
        };

        for (uint8_t r = 0; r < 8; ++r) for (uint8_t f = 0; f < 8; ++f) {
            piece p = board.pieces[r][f];
            if (p.type == piece_type::none || p.piece_player != current_p) continue;
            board_position start_pos = {r, f};

            if (p.type == piece_type::pawn) {
                 int dir = (current_p == player::white) ? 1 : -1; int start_rank = (current_p == player::white) ? 1 : 6; int promotion_rank = (current_p == player::white) ? 7 : 0;
                 auto target1_opt = apply_offset(start_pos, {dir, 0});
                 if (target1_opt && board.pieces[target1_opt->rank][target1_opt->file].type == piece_type::none) {
                      if (target1_opt->rank == promotion_rank) add_promotions(start_pos, *target1_opt, move_type::promotion);
                      else pseudo_legal_moves.push_back({move_type::normal_move, start_pos, *target1_opt});
                      if (r == start_rank) { auto target2_opt = apply_offset(start_pos, {2 * dir, 0}); if (target2_opt && board.pieces[target2_opt->rank][target2_opt->file].type == piece_type::none) pseudo_legal_moves.push_back({move_type::normal_move, start_pos, *target2_opt}); }
                 }
                 for (int file_offset : {-1, 1}) { auto capture_target_opt = apply_offset(start_pos, {dir, file_offset}); if (capture_target_opt) { piece target_piece = board.pieces[capture_target_opt->rank][capture_target_opt->file]; if (target_piece.type != piece_type::none && target_piece.piece_player == opponent_p) { if (capture_target_opt->rank == promotion_rank) add_promotions(start_pos, *capture_target_opt, move_type::promotion); else pseudo_legal_moves.push_back({move_type::capture, start_pos, *capture_target_opt}); } int ep_check_rank = (current_p == player::white) ? 4 : 3; int ep_target_rank = (current_p == player::white) ? 5 : 2; int ep_idx = (ep_target_rank == 5) ? (8 + capture_target_opt->file) : capture_target_opt->file; if (r == ep_check_rank && capture_target_opt->rank == ep_target_rank && target_piece.type == piece_type::none && ep_idx >= 0 && ep_idx < 16 && board.en_passant_valid[ep_idx]) pseudo_legal_moves.push_back({move_type::en_passant, start_pos, *capture_target_opt}); } }
            } else if (p.type == piece_type::knight) { 
                 const std::vector<board_offset> knight_offsets={{1,2},{1,-2},{-1,2},{-1,-2},{2,1},{2,-1},{-2,1},{-2,-1}}; for(const auto&o:knight_offsets){auto t=apply_offset(start_pos,o);if(t){piece tp=board.pieces[t->rank][t->file];if(tp.type==piece_type::none)pseudo_legal_moves.push_back({move_type::normal_move,start_pos,*t});else if(tp.piece_player==opponent_p)pseudo_legal_moves.push_back({move_type::capture,start_pos,*t});}}
            } else if (p.type == piece_type::bishop) { generate_sliding_moves(start_pos, {{1,1},{1,-1},{-1,1},{-1,-1}});
            } else if (p.type == piece_type::rook) { generate_sliding_moves(start_pos, {{1,0},{-1,0},{0,1},{0,-1}});
            } else if (p.type == piece_type::queen) { generate_sliding_moves(start_pos, {{1,1},{1,-1},{-1,1},{-1,-1},{1,0},{-1,0},{0,1},{0,-1}});
            } else if (p.type == piece_type::king) {
                 const std::vector<board_offset>king_offsets={{1,1},{1,-1},{-1,1},{-1,-1},{1,0},{-1,0},{0,1},{0,-1}}; for(const auto&o:king_offsets){auto t=apply_offset(start_pos,o);if(t){piece tp=board.pieces[t->rank][t->file];if(tp.type==piece_type::none)pseudo_legal_moves.push_back({move_type::normal_move,start_pos,*t});else if(tp.piece_player==opponent_p)pseudo_legal_moves.push_back({move_type::capture,start_pos,*t});}} if(!board.in_check[static_cast<int>(current_p)]){int co=(current_p==player::white)?0:2; int cr=(current_p==player::white)?0:7; uint8_t ucr=static_cast<uint8_t>(cr); if(board.can_castle[co]&&board.pieces[cr][5].type==piece_type::none&&board.pieces[cr][6].type==piece_type::none&&!chess::ai::is_square_attacked(board,{ucr,4},opponent_p)&&!chess::ai::is_square_attacked(board,{ucr,5},opponent_p)&&!chess::ai::is_square_attacked(board,{ucr,6},opponent_p))pseudo_legal_moves.push_back({move_type::castle,start_pos,{ucr,6}}); if(board.can_castle[co+1]&&board.pieces[cr][1].type==piece_type::none&&board.pieces[cr][2].type==piece_type::none&&board.pieces[cr][3].type==piece_type::none&&!chess::ai::is_square_attacked(board,{ucr,4},opponent_p)&&!chess::ai::is_square_attacked(board,{ucr,3},opponent_p)&&!chess::ai::is_square_attacked(board,{ucr,2},opponent_p))pseudo_legal_moves.push_back({move_type::castle,start_pos,{ucr,2}}); }
            }
            // std::cerr << "DEBUG C++ Rules: Finished piece type " << std::endl;
        } // End file loop
        // std::cerr << "DEBUG C++ Rules: Finished piece loops..." << std::endl;

        // Add Special Moves
        size_t moves_before_special = pseudo_legal_moves.size();
        pseudo_legal_moves.push_back({move_type::resign, {}, {}});
        if (board.can_claim_draw) pseudo_legal_moves.push_back({move_type::claim_draw, {}, {}});
        // std::cerr << "DEBUG C++ Rules: Added " << std::endl;

        // Filter for Legality
        std::vector<chess_move> legal_moves;
        legal_moves.reserve(pseudo_legal_moves.size());
        for (const auto& move : pseudo_legal_moves) {
            if (move.type == move_type::resign || move.type == move_type::claim_draw) { legal_moves.push_back(move); continue; }
            try {
                board_state next_state = apply_move_impl(board, move); // Use helper for simulation
                 if (!is_player_in_check(next_state, current_p)) { legal_moves.push_back(move); }
            } catch (...) { /* Ignore errors during legality check */ }
        }
        // std::cerr << "DEBUG C++ Rules: Filtered down to " << legal_moves.size() << " legal moves." << std::endl;
        return legal_moves;
    }

    // --- board_state::initial_board_state Definition ---
    board_state board_state::initial_board_state() noexcept {
        board_state initial_state = {}; player p_w = player::white; player p_b = player::black; piece_type R = piece_type::rook; piece_type N = piece_type::knight; piece_type B = piece_type::bishop; piece_type Q = piece_type::queen; piece_type K = piece_type::king; piece_type P = piece_type::pawn; initial_state.pieces[0][0] = {R, p_w}; initial_state.pieces[0][7] = {R, p_w}; initial_state.pieces[0][1] = {N, p_w}; initial_state.pieces[0][6] = {N, p_w}; initial_state.pieces[0][2] = {B, p_w}; initial_state.pieces[0][5] = {B, p_w}; initial_state.pieces[0][3] = {Q, p_w}; initial_state.pieces[0][4] = {K, p_w}; for (int f = 0; f < 8; ++f) initial_state.pieces[1][f] = {P, p_w}; initial_state.pieces[7][0] = {R, p_b}; initial_state.pieces[7][7] = {R, p_b}; initial_state.pieces[7][1] = {N, p_b}; initial_state.pieces[7][6] = {N, p_b}; initial_state.pieces[7][2] = {B, p_b}; initial_state.pieces[7][5] = {B, p_b}; initial_state.pieces[7][3] = {Q, p_b}; initial_state.pieces[7][4] = {K, p_b}; for (int f = 0; f < 8; ++f) initial_state.pieces[6][f] = {P, p_b}; initial_state.can_castle[0] = true; initial_state.can_castle[1] = true; initial_state.can_castle[2] = true; initial_state.can_castle[3] = true; initial_state.current_player = p_w; initial_state.status = game_status::normal; initial_state.turns_since_last_capture_or_pawn = 0; initial_state.can_claim_draw = false; initial_state.in_check[static_cast<int>(p_w)] = false; initial_state.in_check[static_cast<int>(p_b)] = false; for(auto& flag : initial_state.en_passant_valid) { flag = false; } return initial_state;
    }

     // --- Operator Definitions ---
     bool board_position::operator==(const board_position& other) const { return rank == other.rank && file == other.file; }
     bool board_position::operator<(const board_position& other) const { if (rank != other.rank) return rank < other.rank; return file < other.file; }
     bool piece::operator==(const piece& other) const { return type == other.type && piece_player == other.piece_player; }

} // namespace chess