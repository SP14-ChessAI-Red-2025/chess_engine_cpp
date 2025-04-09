#pragma once

#include "chess_rules.hpp"

namespace chess::ai {

struct chess_ai_state {
    void make_move(board_state& board, std::int32_t difficulty);
};

}