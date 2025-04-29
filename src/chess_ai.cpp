#include "chess_ai.hpp"

#include <ranges>
#include <utility>
#include <limits>
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <cassert>
#include <iostream>

#include <thread>
#include <mutex>
#include <condition_variable>

#define THREADING

namespace chess::ai {

struct game_tree;

using score_t = double;

struct chess_ai_state::thread_pool_t {
    struct worker {
        std::thread thread;
        std::mutex mutex;
        std::condition_variable cond;
        std::condition_variable cond2;

        game_tree* data_in = nullptr;
        score_t data_out = 0;
        bool has_result = false;

        bool should_exit = false;

        static score_t do_work(game_tree* game_tree);

        worker() {
            thread = std::thread{[&mutex = mutex, &should_exit = should_exit, this](game_tree*& data_in, score_t& data_out, std::condition_variable& cond, std::condition_variable& cond2) {
                while(true) {
                    // std::cout << "Worker iter" << std::endl;
                    std::unique_lock lock{mutex};
                    // std::cout << "Worker: got lock" << std::endl;

                    cond.wait(lock, [&data_in, &should_exit = should_exit] { return data_in != nullptr || should_exit; });

                    // std::cout << "Worker: done waiting" << std::endl;

                    if(should_exit) {
                        std::cout << std::format("Exiting worker thread") << std::endl;
                        break;
                    }

                    std::cout << std::format("Worker: got data") << std::endl;

                    data_out = do_work(data_in);
                    has_result = true;

                    std::cout << std::format("Worker: wrote {}", data_out) << std::endl;

                    // Reset data_in
                    data_in = nullptr;

                    cond2.notify_one();

                    std::cout << std::format("Worker: notify_one()") << std::endl;
                }
            }, std::ref(data_in), std::ref(data_out), std::ref(cond), std::ref(cond2)};
        }

        ~worker() {
            if(thread.joinable()) {
                should_exit = true;
                cond.notify_one();

                std::cout << "Joining" << std::endl;
                thread.join();
            }
        }
    };

    worker workers[4] = {};
};

chess_ai_state::chess_ai_state(const char* model_path)
#ifdef NNUE_ENABLED
    : nnue_evaluator{model_path}
#endif
{
    this->thread_pool = std::make_unique<thread_pool_t>();
}

chess_ai_state::~chess_ai_state() = default;

// Whether the AI should consider the move
// The AI currently ignores resignations and claiming draws
bool should_consider_move(chess_move move) {
    return move.type != move_type::resign && move.type != move_type::claim_draw;
}

// A positive result is good for player, negative is bad for player
score_t rank_board_old(const chess_ai_state& ai_state, const board_state& board, player player) {
    score_t board_value = 0;

    for(const auto& rank : board.pieces) {
        for(const auto& piece : rank) {
            score_t piece_value = 0;

            using enum piece_type;

            switch(piece.type) {
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

            if(piece.piece_player == player) {
                board_value += piece_value;
            } else {
                board_value -= piece_value;
            }
        }
    }

    return board_value;
}

score_t rank_board(chess_ai_state& ai_state, const board_state& board, player player) {
#ifdef NNUE_ENABLED
    return ai_state.nnue_evaluator.evaluate(board) * (player == player::white ? 1 : -1);
#else
    return rank_board_old(ai_state, board, player);
#endif
}

struct game_tree {
    chess_ai_state* ai_state;

    board_state current_state;

    enum player player; // The maximizing player

    std::optional<chess_move> move = {}; // The move that resulted in the current_state

    std::int32_t start_depth = 0;

    std::vector<game_tree> get_children() {
        std::vector<game_tree> children = {};

        for(auto& move : get_valid_moves(current_state) | std::views::filter(should_consider_move)) {
            auto board = apply_move(current_state, move);

            children.emplace_back(ai_state, board, player, move, start_depth);
        }

        return children;
    }

    // Rank the current state according to the minimax algorithm
    score_t minimax(std::size_t depth, bool maximizing, score_t alpha, score_t beta) {
        if(depth == 0) {
            return rank_board(*ai_state, current_state, player);
        }

        auto children = get_children();

        if(maximizing) {
            score_t max_score = std::numeric_limits<decltype(max_score)>::min();

            for(auto& child : children) {
                auto score = child.minimax(depth - 1, false, alpha, beta);

                max_score = std::max(score, max_score);

                if(score >= beta) {
                    break;
                }

                alpha = std::max(score, alpha);
            }

            return max_score;
        } else {
            score_t min_score = std::numeric_limits<decltype(min_score)>::max();

            for(auto child : children) {
                auto score = child.minimax(depth - 1, true, alpha, beta);

                min_score = std::min(score, min_score);

                if(score <= alpha) {
                    break;
                }

                beta = std::min(score, beta);
            }

            return min_score;
        }
    }

    board_state get_best_move(std::size_t depth, chess_ai_state& ai_state) {
        assert(depth != 0);

        auto children = get_children();

        std::vector<std::pair<game_tree*, score_t>> children_with_scores;
        children_with_scores.reserve(children.size());

#ifndef THREADING
        score_t alpha = std::numeric_limits<decltype(alpha)>::min();
        score_t beta = std::numeric_limits<decltype(beta)>::max();

        std::ranges::transform(children, std::back_inserter(children_with_scores), [=](game_tree& child) {
            std::cout << std::format("calling minimax(_, {}, {}, {})", false, alpha, beta) << std::endl;

            auto val = child.minimax(depth - 1, false, alpha, beta);

            std::cout << "Got value: " << val << std::endl;

            return std::make_pair(&child, val);
        });
#else

        std::size_t child_idx = 0;

        while(true) {
            if(child_idx >= children.size()) {
                goto done;
            }

            for(std::size_t i = 0; i < std::size(ai_state.thread_pool->workers); i++) {
                std::size_t child_idx_tmp = child_idx + i;

                if(child_idx_tmp >= children.size()) {
                    break;
                }

                auto& child = children[child_idx_tmp];

                auto& worker = ai_state.thread_pool->workers[i];

                {
                    std::unique_lock lock{worker.mutex};

                    worker.data_in = &child;

                    worker.cond.notify_one();

                    std::cout << std::format("Main thread: sent data to worker thread {}", i) << std::endl;
                }
            }

            for(std::size_t i = 0; i < std::size(ai_state.thread_pool->workers); i++) {
                std::size_t child_idx_tmp = child_idx + i;

                if(child_idx_tmp >= children.size()) {
                    break;
                }

                {
                    auto& child = children[child_idx_tmp];

                    auto& worker = ai_state.thread_pool->workers[i];

                    std::unique_lock lock{worker.mutex};

                    // std::cout << "Main thread: got lock again" << std::endl;

                    worker.cond2.wait(lock, [&has_result = worker.has_result] { return has_result; });

                    std::cout << std::format("Main thread: got value {}", worker.data_out) << std::endl;

                    worker.has_result = false;

                    children_with_scores.emplace_back(&child, worker.data_out);
                }
            }

            child_idx += std::size(ai_state.thread_pool->workers);
        }
        done:
#endif

        auto projection = &std::pair<game_tree*, score_t>::second;

        return std::ranges::max_element(children_with_scores, {}, projection)->first->current_state;
    }
};

score_t chess_ai_state::thread_pool_t::worker::do_work(game_tree* game_tree) {
    score_t alpha = -std::numeric_limits<decltype(alpha)>::max();
    score_t beta = std::numeric_limits<decltype(beta)>::max();

    std::cout << std::format("calling minimax({}, {}, {}, {})", game_tree->start_depth - 1, false, alpha, beta) << std::endl;

    auto val = game_tree->minimax(game_tree->start_depth - 1, false, alpha, beta);

    std::cout << "Got value: " << val << std::endl;

    return val;
}

void chess_ai_state::make_move(board_state& board, std::int32_t difficulty) {
    if(board.status != game_status::normal) throw std::runtime_error{"Game is over"};
    if(difficulty == 0) throw std::runtime_error{"Difficulty must be at least 1"};

    game_tree tree{this, board, board.current_player, {}, difficulty};

    board = tree.get_best_move(difficulty, *this);
}

} // namespace chess::ai
