#pragma once

#include "chess_rules.hpp"
#include <chess_cpp/nnue_evaluator.hpp>

namespace chess::ai {

struct chess_ai_state {
#ifdef NNUE_ENABLED
    NNUEEvaluator nnue_evaluator;

    explicit chess_ai_state(const char* model_path);
#endif

    void make_move(board_state& board, std::int32_t difficulty);
};

}