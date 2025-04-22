#include <chess_cpp/chess_ai.hpp>
#include <chess_cpp/chess_rules.hpp>
#include <chess_cpp/nnue_evaluator.hpp>
#include <vector>
#include <memory>
#include <functional> // For std::hash
#include <limits>
#include <algorithm>
#include <optional>
#include <string>
#include <stdexcept>
#include <iostream>
#include <future>
#include <unordered_map>
#include <mutex>
#include <map>
#include <cmath> // For std::isnan and std::isinf

namespace chess::ai {

    bool should_consider_move(const chess_move& move) {
        return move.type != move_type::resign && move.type != move_type::claim_draw;
    }

    bool are_states_equal(const board_state& s1, const board_state& s2) {
        if (s1.current_player != s2.current_player ||
            s1.status != s2.status ||
            s1.turns_since_last_capture_or_pawn != s2.turns_since_last_capture_or_pawn) {
             return false;
        }
        for(int r=0; r<8; ++r) {
            for(int f=0; f<8; ++f) {
                if (!(s1.pieces[r][f] == s2.pieces[r][f])) return false;
            }
        }
        for(int i=0; i<4; ++i) {
            if (s1.can_castle[i] != s2.can_castle[i]) return false;
        }
         for(int i=0; i<16; ++i) {
            if (s1.en_passant_valid[i] != s2.en_passant_valid[i]) return false;
        }
        return true;
    }

    // --- Move Ordering ---
    const std::map<piece_type, int> piece_values = {
        {piece_type::pawn, 100},
        {piece_type::knight, 300},
        {piece_type::bishop, 310},
        {piece_type::rook, 500},
        {piece_type::queen, 900},
        {piece_type::king, 20000}
    };

    int get_piece_value(piece_type type) {
        auto it = piece_values.find(type);
        return (it != piece_values.end()) ? it->second : 0;
    }

    // Simple move scoring heuristic for ordering
    int score_move(const board_state& board, const chess_move& move) {
        int score = 0;

        // Promotion moves
        if (move.type == move_type::promotion) {
            score = 20000 + get_piece_value(move.promotion_target);
            return score;
        }

        // Capture moves (MVV-LVA: Most Valuable Victim - Least Valuable Attacker)
        if (move.type == move_type::capture) {
            piece captured_piece = board.pieces[move.target_position.rank][move.target_position.file];
            piece capturing_piece = board.pieces[move.start_position.rank][move.start_position.file];
            int victim_value = get_piece_value(captured_piece.type);
            int aggressor_value = get_piece_value(capturing_piece.type);
            score = 10000 + (victim_value * 10) - aggressor_value; // MVV-LVA
            return score;
        }

        // En passant
        if (move.type == move_type::en_passant) {
            int victim_value = get_piece_value(piece_type::pawn);
            int aggressor_value = get_piece_value(piece_type::pawn);
            score = 10000 + (victim_value * 10) - aggressor_value;
            return score;
        }

        // Castling
        if (move.type == move_type::castle) {
            score = 50; // Small bonus for castling
            return score;
        }

        // Positional bonuses
        piece moving_piece = board.pieces[move.start_position.rank][move.start_position.file];
        if (moving_piece.type == piece_type::pawn) {
            // Bonus for advancing pawns
            score += 10 * (move.target_position.rank - move.start_position.rank);
        } else if (moving_piece.type == piece_type::knight || moving_piece.type == piece_type::bishop) {
            // Bonus for developing minor pieces
            score += 30;
        } else if (moving_piece.type == piece_type::rook || moving_piece.type == piece_type::queen) {
            // Bonus for controlling open files or attacking
            score += 20;
        }

        // Center control bonus
        const std::vector<std::pair<int, int>> center_squares = {{3, 3}, {3, 4}, {4, 3}, {4, 4}};
        for (const auto& center : center_squares) {
            if (move.target_position.rank == center.first && move.target_position.file == center.second) {
                score += 50; // Bonus for controlling the center
            }
        }

        // Penalty for leaving the king exposed
        if (moving_piece.type == piece_type::king) {
            // Check if the king moves to an unsafe square
            player opponent_color = (board.current_player == player::white) ? player::black : player::white;
            if (is_square_attacked(board, move.target_position, opponent_color)) {
                score -= 100; // Penalize unsafe king moves
            }
        }

        if (move.type == move_type::check) {
            score += 50; // Bonus for delivering a check
        }
        if (threatens_high_value_piece(board, move)) {
            score += 100; // Bonus for threatening a high-value piece
        }

        // Other moves get a base score of 0 initially
        return score;
    }

    bool is_square_attacked(const board_state& board, const board_position& square, player opponent_color) {
        for (const auto& move : chess::get_valid_moves(board)) {
            // Check if the move targets the square and is made by the opponent
            const auto& attacking_piece = board.pieces[move.start_position.rank][move.start_position.file];
            if (move.target_position == square && attacking_piece.piece_player == opponent_color) {
                return true;
            }
        }
        return false;
    }

    bool threatens_high_value_piece(const board_state& board, const chess_move& move) {
        const piece& target_piece = board.pieces[move.target_position.rank][move.target_position.file];
        return target_piece.type != piece_type::none && piece_values.at(target_piece.type) >= 5; // Example threshold
    }

    game_tree::game_tree(const board_state& state)
        : current_state(state), score(std::numeric_limits<double>::quiet_NaN()), move(), children() {}
    static std::unordered_map<std::size_t, double> transposition_table;
    static std::mutex transposition_table_mutex;

    struct board_state_hasher {
        std::size_t operator()(const board_state& state) const {
            // Example hash function combining board state properties
            std::size_t hash = 0;
            for (int r = 0; r < 8; ++r) {
                for (int f = 0; f < 8; ++f) {
                    hash ^= std::hash<int>()(static_cast<int>(state.pieces[r][f].type)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
            }
            hash ^= std::hash<int>()(static_cast<int>(state.current_player));
            return hash;
        }
    };

    // Alpha-Beta deepen function with Move Ordering
    double game_tree::deepen(std::size_t depth, NNUEEvaluator& evaluator, double alpha, double beta) {
        bool is_terminal = (current_state.status != game_status::normal);

        // Hash the current board state
        std::size_t board_hash = board_state_hasher()(current_state);

        {
            std::lock_guard<std::mutex> lock(transposition_table_mutex);
            if (transposition_table.find(board_hash) != transposition_table.end()) {
                return transposition_table[board_hash];
            }
        }

        if (depth == 0 || is_terminal) {
            try {
                score = evaluator.evaluate(current_state);
                if (std::isnan(score)) {
                    if (current_state.status == game_status::checkmate) {
                        score = (current_state.current_player == player::white) ? -200000.0 : 200000.0;
                    } else if (current_state.status == game_status::draw || current_state.status == game_status::resigned) {
                        score = 0.0;
                    } else {
                        score = -1000.0; // Fallback for unexpected NaN
                    }
                }
            } catch (const std::runtime_error& e) {
                std::cerr << "Evaluation error at depth " << depth << ": " << e.what() << std::endl;
                score = (current_state.status == game_status::checkmate) ?
                        ((current_state.current_player == player::white) ? -200000.0 : 200000.0) : 0.0;
            }

            // Cache the result in the transposition table
            transposition_table[board_hash] = score;
            return score;
        }

        auto valid_moves = chess::get_valid_moves(current_state);
        std::vector<std::pair<int, chess_move>> scored_moves;
        scored_moves.reserve(valid_moves.size());

        for (const auto& m : valid_moves) {
            if (should_consider_move(m)) {
                scored_moves.emplace_back(score_move(current_state, m), m);
            }
        }

        // Sort moves: highest score first
        std::sort(scored_moves.rbegin(), scored_moves.rend(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        bool is_maximizing_player = (current_state.current_player == player::white);

        if (is_maximizing_player) {
            score = -std::numeric_limits<double>::infinity();
            for(const auto& scored_move_pair : scored_moves) {
                const auto& current_move = scored_move_pair.second;
                board_state next_state = chess::apply_move(current_state, current_move);
                auto child = std::make_unique<game_tree>(next_state); // Create child node

                double child_score = child->deepen(depth - 1, evaluator, alpha, beta);
                score = std::max(score, child_score); // Update best score for maximizer
                alpha = std::max(alpha, score);       // Update alpha

                if (beta <= alpha) {
                    break; // Beta cutoff
                }
            }
        } else { // Minimizing player
            score = std::numeric_limits<double>::infinity();
            for(const auto& scored_move_pair : scored_moves) {
                const auto& current_move = scored_move_pair.second;
                board_state next_state = chess::apply_move(current_state, current_move);
                auto child = std::make_unique<game_tree>(next_state);

                double child_score = child->deepen(depth - 1, evaluator, alpha, beta);
                score = std::min(score, child_score); // Update best score for minimizer
                beta = std::min(beta, score);         // Update beta

                if (beta <= alpha) {
                    break; // Alpha cutoff
                }
            }
        }

        // Handle cases where no moves were explored or score remained infinite/NaN
        if (std::isinf(score) || std::isnan(score)) {
             is_terminal = (current_state.status != game_status::normal); // Re-check terminal state
             if(is_terminal) {
                  score = (current_state.status == game_status::checkmate) ?
                          ((current_state.current_player == player::white) ? -200000.0 : 200000.0) : 0.0;
             } else {
                 // Should not happen if valid_moves existed, but fallback to 0
                 score = 0.0;
             }
        }
        // Cache the result in the transposition table
        transposition_table[board_hash] = score;
        return score;
    }

     void game_tree::add_child(const chess_move& m, const board_state& next_state) {
         children.emplace_back(std::make_unique<game_tree>(next_state));
         children.back()->move = m;
     }


    chess_ai_state::chess_ai_state(const std::string& model_path)
        try : root(nullptr), evaluator_(model_path) {
            // Constructor body remains empty, initialization done in initializer list
        } catch (const std::runtime_error& e) {
            // Handle potential exceptions from NNUEEvaluator constructor
            std::cerr << "Error initializing NNUE Evaluator: " << e.what() << std::endl;
            throw std::runtime_error("Failed to initialize AI state: NNUE evaluator failed.");
        } catch (...) {
             std::cerr << "Unknown error during AI state initialization." << std::endl;
             throw std::runtime_error("Failed to initialize AI state: Unknown error.");
        }

    void chess_ai_state::make_move(board_state& board, std::int32_t difficulty) {
        auto current_root_state = board;

        if (current_root_state.status != game_status::normal) {
            std::cerr << "[INFO] Game is over. No moves to make." << std::endl;
            return;
        }

        std::size_t search_depth;
        if (difficulty <= 1) search_depth = 2; // Lower depth for lower difficulty
        else if (difficulty <= 2) search_depth = 3; // Medium depth
        else search_depth = 4; // Higher depth for higher difficulty

        std::cout << "[INFO] AI using search depth: " << search_depth << " (Difficulty: " << difficulty << ")" << std::endl;

        if (search_depth < 1) {
             std::cerr << "[ERROR] Search depth must be at least 1." << std::endl;
             return; // Or throw an exception
        }

        double best_score;
        chess_move best_move{};
        bool best_move_found = false;
        bool is_maximizing_player = (current_root_state.current_player == player::white);

        auto valid_moves = chess::get_valid_moves(current_root_state);

        std::vector<std::pair<int, chess_move>> scored_root_moves;
        scored_root_moves.reserve(valid_moves.size());
        for (const auto& current_move : valid_moves) {
            if (should_consider_move(current_move)) {
                 scored_root_moves.emplace_back(score_move(current_root_state, current_move), current_move);
            }
        }
        std::sort(scored_root_moves.rbegin(), scored_root_moves.rend(), [](const auto& a, const auto& b) {
            return a.first < b.first; // Sort descending by score
        });

        std::vector<std::future<double>> futures;
        std::vector<chess_move> future_moves;

        std::cout << "[INFO] Launching parallel search tasks for " << scored_root_moves.size() << " considered moves..." << std::endl;

        // Use a thread-safe transposition table
        std::mutex transposition_table_mutex;

        for (const auto& scored_move_pair : scored_root_moves) {
            const auto& current_move = scored_move_pair.second;
            board_state next_state = chess::apply_move(current_root_state, current_move);

            // Capture evaluator_ by reference and use thread-safe transposition table
            futures.push_back(std::async(std::launch::async,
                [&evaluator = this->evaluator_, next_state, search_depth, &transposition_table_mutex]() -> double {
                    auto move_subtree_root = std::make_unique<game_tree>(next_state);

                    // Thread-safe access to transposition table
                    std::size_t hash = board_state_hasher()(next_state);
                    {
                        std::lock_guard<std::mutex> lock(transposition_table_mutex);
                        if (transposition_table.find(hash) != transposition_table.end()) {
                            return transposition_table[hash];
                        }
                    }

                    // Perform the search
                    double score = move_subtree_root->deepen(search_depth - 1, evaluator,
                                                             -std::numeric_limits<double>::infinity(),
                                                              std::numeric_limits<double>::infinity());

                    // Store the result in the transposition table
                    {
                        std::lock_guard<std::mutex> lock(transposition_table_mutex);
                        transposition_table[hash] = score;
                    }

                    return score;
                }
            ));
            future_moves.push_back(current_move);
        }

        // Collect results and find the best move based on player
        if (is_maximizing_player) {
            best_score = -std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < futures.size(); ++i) {
                try {
                    double child_score = futures[i].get();
                    if (child_score > best_score) {
                        best_score = child_score;
                        best_move = future_moves[i];
                        best_move_found = true;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[ERROR] Exception in search task for move " << i << ": " << e.what() << std::endl;
                }
            }
        } else { // Minimizing player
            best_score = std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < futures.size(); ++i) {
                 try {
                    double child_score = futures[i].get();
                    if (child_score < best_score) {
                        best_score = child_score;
                        best_move = future_moves[i];
                        best_move_found = true;
                    }
                 } catch (const std::exception& e) {
                     std::cerr << "[ERROR] Exception in search task for move " << i << ": " << e.what() << std::endl;
                 }
            }
        }
        std::cout << "[INFO] Finished collecting results. Best score found: " << best_score << std::endl;

        if (best_move_found) {
            board = chess::apply_move(board, best_move);
        } else {
             std::cerr << "[WARNING] AI could not find a best move. Playing fallback." << std::endl;
             auto first_considered = std::find_if(valid_moves.begin(), valid_moves.end(), should_consider_move);
             if (first_considered != valid_moves.end()) {
                  board = chess::apply_move(board, *first_considered);
             } else if (!valid_moves.empty() && (valid_moves[0].type == move_type::resign || valid_moves[0].type == move_type::claim_draw)) {
                 board = chess::apply_move(board, valid_moves[0]);
             }
             // If no moves at all, game should be over.
        }
    }

} // namespace chess::ai