#pragma once

#include "chess_rules.hpp"
#include <chess_cpp/nnue_evaluator.hpp>

namespace chess::ai {

struct chess_ai_state {
#ifdef NNUE_ENABLED
    NNUEEvaluator nnue_evaluator;
#endif

    struct thread_pool_t;
    thread_pool_t* thread_pool;

    // model_path is unused if NNUE_ENABLE is not defined
    explicit chess_ai_state(const char* model_path);

    void make_move(board_state& board, std::int32_t difficulty);

    ~chess_ai_state();
};

}