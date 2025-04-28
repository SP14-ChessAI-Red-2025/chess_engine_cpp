// src/chess_ai.cpp
#include "chess_cpp/chess_ai.hpp" // Include the corresponding header first
#include "chess_cpp/chess_rules.hpp"
#include "chess_cpp/nnue_evaluator.hpp"
#include "chess_cpp/threefold_repetition.hpp"

// Standard library includes
#include <vector>
#include <memory>
#include <functional>
#include <limits>
#include <algorithm> // For std::sort, std::max, std::min, std::find_if
#include <optional>
#include <string>
#include <stdexcept>
#include <iostream> // For logging (std::cout, std::cerr)
#include <future>   // Potentially for parallel search later
#include <unordered_map>
#include <mutex>
#include <map>
#include <cmath>    // For std::abs
#include <thread>   // Potentially for parallel search later
#include <atomic>
#include <chrono>   // For timing the search

namespace chess::ai {

    // --- Global Transposition Table & Cancellation Flag Definitions ---
    std::unordered_map<chess::board_state, transposition_table_entry, chess::board_state_hasher, chess::board_state_equal> transposition_table;
    std::mutex transposition_table_mutex;
    std::atomic<bool> search_cancelled(false);

    void cancel_ai_search() {
        search_cancelled.store(true);
        std::cout << "[INFO] AI search cancellation requested." << std::endl;
    }


    // --- Helper Function Definitions ---
    bool should_consider_move(const chess_move& move) {
        return move.type != move_type::resign && move.type != move_type::claim_draw;
    }

    int score_move(const board_state& state, const chess_move& move) {
        int score = 0;
        if (!in_bounds(move.start_position.rank, move.start_position.file)) return -10000;
        piece moved_piece = state.pieces[move.start_position.rank][move.start_position.file];
        if (!in_bounds(move.target_position.rank, move.target_position.file)) return -10000;
        piece target_piece = state.pieces[move.target_position.rank][move.target_position.file];

        if (move.type == move_type::capture || move.type == move_type::en_passant) {
             piece captured_piece = (move.type == move_type::en_passant) ?
                piece{piece_type::pawn, (moved_piece.piece_player == player::white ? player::black : player::white)}
                : target_piece;
             if (piece_values.count(captured_piece.type) && piece_values.count(moved_piece.type)) {
                  score += 10 * piece_values.at(captured_piece.type) - piece_values.at(moved_piece.type);
             }
        }
        if (move.type == move_type::promotion) {
            if (piece_values.count(move.promotion_target)) {
                 score += piece_values.at(move.promotion_target);
            }
        }
        return score;
    }

    bool is_square_attacked(const board_state& board, const board_position& square, player attacking_player) {
         if (!in_bounds(square.rank, square.file)) return false;
         int pawn_attack_rank_offset = (attacking_player == player::white) ? -1 : 1;
         int pawn_attack_file_offsets[] = {-1, 1};
         for (int file_offset : pawn_attack_file_offsets) {
             auto start_pos_opt = chess::apply_offset(square, {static_cast<int8_t>(pawn_attack_rank_offset), static_cast<int8_t>(file_offset)});
             if (start_pos_opt) {
                 const auto& p = board.pieces[start_pos_opt->rank][start_pos_opt->file];
                 if (p.type == piece_type::pawn && p.piece_player == attacking_player) return true;
             }
         }
         const board_offset knight_offsets[] = { {1, 2}, {1, -2}, {-1, 2}, {-1, -2}, {2, 1}, {2, -1}, {-2, 1}, {-2, -1} };
         for (const auto& offset : knight_offsets) {
             auto start_pos_opt = chess::apply_offset(square, offset);
             if (start_pos_opt) {
                 const auto& p = board.pieces[start_pos_opt->rank][start_pos_opt->file];
                 if (p.type == piece_type::knight && p.piece_player == attacking_player) return true;
             }
         }
         const board_offset sliding_offsets[] = { {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, 1}, {1, -1}, {-1, -1} };
         for (const auto& offset : sliding_offsets) {
             for (int i = 1; i < 8; ++i) {
                 auto start_pos_opt = chess::apply_offset(square, {static_cast<int8_t>(offset.rank_offset * i), static_cast<int8_t>(offset.file_offset * i)});
                 if (!start_pos_opt) break;
                 const auto& p = board.pieces[start_pos_opt->rank][start_pos_opt->file];
                 if (p.type != piece_type::none) {
                     if (p.piece_player == attacking_player) {
                         bool is_rook_dir = (offset.rank_offset == 0 || offset.file_offset == 0);
                         bool is_bishop_dir = (std::abs(offset.rank_offset) == 1 && std::abs(offset.file_offset) == 1);
                         if ((p.type == piece_type::rook && is_rook_dir) || (p.type == piece_type::bishop && is_bishop_dir) || (p.type == piece_type::queen)) return true;
                     }
                     break;
                 }
             }
         }
         const board_offset king_offsets[] = { {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, 1}, {1, -1}, {-1, -1} };
          for (const auto& offset : king_offsets) {
              auto start_pos_opt = chess::apply_offset(square, offset);
               if (start_pos_opt) {
                   const auto& p = board.pieces[start_pos_opt->rank][start_pos_opt->file];
                  if (p.type == piece_type::king && p.piece_player == attacking_player) return true;
              }
          }
         return false;
    }


    // --- Game Tree Class Member Function Definitions ---
    game_tree::game_tree(const board_state& state) : current_state(state) {}

    double game_tree::deepen(std::size_t depth, NNUEEvaluator& evaluator, double alpha, double beta, previous_board_states& history) {
        if (search_cancelled.load(std::memory_order_relaxed)) throw std::runtime_error("Search cancelled");

        // TT Lookup (Simplified)
        transposition_table_entry tt_entry; bool tt_hit = false; size_t tt_depth = 0;
        { std::lock_guard<std::mutex> lock(transposition_table_mutex);
          auto it = transposition_table.find(this->current_state);
          if (it != transposition_table.end()) { tt_entry = it->second; tt_hit = true; tt_depth = it->second.depth; }
        }
        if (tt_hit && tt_depth >= depth) return tt_entry.score; // Simplified TT return

        // Terminal Node Check
        if (this->current_state.status != game_status::normal) {
             return (this->current_state.status == game_status::checkmate) ? (-200000.0 - static_cast<double>(depth)) : 0.0;
        }
        if (depth == 0) {
            try {
                double eval_score = static_cast<double>(evaluator.evaluate(this->current_state));
                return (this->current_state.current_player == player::white) ? eval_score : -eval_score;
            } catch (...) { return 0.0; } // Catch potential evaluation errors
        }

        // Generate, Score, Sort Moves
        auto valid_moves = chess::get_valid_moves(this->current_state);
        std::vector<std::pair<int, chess_move>> scored_moves;
        scored_moves.reserve(valid_moves.size());
        for (const auto& m : valid_moves) {
            if (should_consider_move(m)) scored_moves.emplace_back(score_move(this->current_state, m), m);
        }
        if (scored_moves.empty()) {
             bool in_check = this->current_state.in_check[static_cast<int>(this->current_state.current_player)];
             return in_check ? (-200000.0 - static_cast<double>(depth)) : 0.0; // Checkmate or Stalemate
        }
        std::sort(scored_moves.begin(), scored_moves.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

        // Alpha-Beta Loop
        double best_val = -std::numeric_limits<double>::infinity();
        bool raised_alpha = false;
        for(const auto& scored_move_pair : scored_moves) {
            const auto& current_move = scored_move_pair.second;
            previous_board_states branch_history = history;
            board_state next_state = this->current_state;
            chess::apply_move(next_state, current_move, branch_history); // Apply to copy

            double child_score = -game_tree(next_state).deepen(depth - 1, evaluator, -beta, -alpha, branch_history);

            if (child_score > best_val) {
                 best_val = child_score;
                 if (best_val > alpha) { alpha = best_val; raised_alpha = true; }
            }
            if (alpha >= beta) break; // Prune
            if (search_cancelled.load(std::memory_order_relaxed)) throw std::runtime_error("Search cancelled");
        }

        // TT Store (Simplified)
        transposition_table_entry::node_type tt_node_type = transposition_table_entry::node_type::exact; // Default
        if (best_val <= alpha && !raised_alpha) tt_node_type = transposition_table_entry::node_type::upper_bound;
        else if (best_val >= beta) tt_node_type = transposition_table_entry::node_type::lower_bound;
        { std::lock_guard<std::mutex> lock(transposition_table_mutex);
          auto it = transposition_table.find(this->current_state);
          if (it == transposition_table.end() || depth >= it->second.depth) {
               transposition_table[this->current_state] = {best_val, depth, tt_node_type};
          }
        }
        return best_val;
    }

    // --- Chess AI State Class Member Function Definitions ---
    chess_ai_state::chess_ai_state(const std::string& model_path) try : evaluator_(model_path) {
        std::cout << "[INFO] AI State initialized with NNUE model: " << model_path << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "[ERROR] Runtime error initializing NNUE Evaluator: " << e.what() << std::endl;
        throw std::runtime_error("Failed to initialize AI state: NNUE evaluator failed.");
    } catch (...) {
         std::cerr << "[ERROR] Unknown error during AI state initialization." << std::endl;
         throw std::runtime_error("Failed to initialize AI state: Unknown error.");
    }

    void chess_ai_state::make_move(board_state& board, std::int32_t difficulty, previous_board_states& history) {
        search_cancelled.store(false);
        { std::lock_guard<std::mutex> lock(transposition_table_mutex); transposition_table.clear(); }
        std::cout << "[INFO] Transposition Table cleared for new search." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        if (board.status != game_status::normal) return;

        std::size_t search_depth = (difficulty <= 1) ? 2 : (difficulty <= 2) ? 4 : 6;
        std::cout << "[INFO] AI (" << (board.current_player == player::white ? "White" : "Black") << ") searching at depth: " << search_depth << " (Difficulty: " << difficulty << ")" << std::endl;
        if (search_depth < 1) return;

        double best_score = -std::numeric_limits<double>::infinity();
        chess_move best_move{};
        bool best_move_found = false;

        auto valid_moves = chess::get_valid_moves(board);
        std::vector<std::pair<int, chess_move>> scored_root_moves;
        scored_root_moves.reserve(valid_moves.size());
        for (const auto& m : valid_moves) {
            if (should_consider_move(m)) scored_root_moves.emplace_back(score_move(board, m), m);
        }

        if (scored_root_moves.empty()) {
             std::cerr << "[WARNING] AI found no gameplay moves at root." << std::endl;
             auto fallback_move = std::find_if(valid_moves.begin(), valid_moves.end(), [](const auto& m){ return m.type == move_type::resign || m.type == move_type::claim_draw; });
             if (fallback_move != valid_moves.end()) chess::apply_move(board, *fallback_move, history);
             return;
        }

        std::sort(scored_root_moves.begin(), scored_root_moves.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::cout << "[INFO] Searching " << scored_root_moves.size() << " root moves..." << std::endl;

        double alpha = -std::numeric_limits<double>::infinity();
        double beta = std::numeric_limits<double>::infinity();

        for (const auto& scored_move_pair : scored_root_moves) {
            const auto& current_move = scored_move_pair.second;

            // --- Corrected Printing & Flushing ---
            std::cout << "  Considering move: ("
                      << static_cast<int>(current_move.start_position.rank) << "," << static_cast<int>(current_move.start_position.file) << ")->"
                      << "(" << static_cast<int>(current_move.target_position.rank) << "," << static_cast<int>(current_move.target_position.file) << ")"
                      << " ..." << std::flush; // Flush BEFORE deepen
            // ---

            previous_board_states branch_history = history;
            try {
                board_state next_state_for_search = board;
                chess::apply_move(next_state_for_search, current_move, branch_history); // Apply to copy

                double score = -game_tree(next_state_for_search).deepen(search_depth - 1, evaluator_, -beta, -alpha, branch_history);

                // --- Corrected Printing & Flushing ---
                std::cout << " Score: " << score << std::endl; // Print score AFTER deepen and use endl to flush
                // ---

                if (search_cancelled.load(std::memory_order_relaxed)) throw std::runtime_error("Search cancelled");

                if (score > best_score) {
                    best_score = score; best_move = current_move; best_move_found = true;
                    if (best_score > alpha) alpha = best_score;
                }
                if (alpha >= beta) { std::cout << "      (Beta Cutoff)" << std::endl; break; }

            } catch (const std::runtime_error& e) {
                 std::cout << " Error: " << e.what() << std::endl;
                 if (std::string(e.what()) == "Search cancelled") goto apply_best_move;
            } catch (...) { std::cout << " Unknown Error" << std::endl; }
        }

    apply_best_move:
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[INFO] Search finished in " << duration.count() << " ms." << std::endl;

        if (best_move_found) {
            chess::apply_move(board, best_move, history); // Apply best found move
        } else {
             std::cerr << "[WARNING] AI could not find a best move (search failed or cancelled early?). Playing fallback." << std::endl;
             if (!scored_root_moves.empty()) {
                  const auto& fallback_move = scored_root_moves.front().second;
                  std::cerr << "[WARNING] Applying highest scored gameplay move as fallback." << std::endl;
                  chess::apply_move(board, fallback_move, history);
             } else { // Should already have been handled, but belt-and-suspenders
                  auto fallback_non_gameplay = std::find_if(valid_moves.begin(), valid_moves.end(), [](const auto& m){ return m.type == move_type::resign || m.type == move_type::claim_draw; });
                  if (fallback_non_gameplay != valid_moves.end()) {
                       std::cerr << "[WARNING] Applying resign/draw as fallback." << std::endl;
                       chess::apply_move(board, *fallback_non_gameplay, history);
                  } else {
                      std::cerr << "[ERROR] No valid moves found for AI, even as fallback!" << std::endl;
                      throw std::logic_error("AI failed to find any applicable move.");
                  }
             }
        }
    } // end of make_move function

} // namespace chess::ai