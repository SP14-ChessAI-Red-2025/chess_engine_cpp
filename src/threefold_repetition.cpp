#include "chess_rules.hpp"

#include <unordered_set>
#include <cstddef>

namespace chess {
struct board_state_hasher {
    std::size_t operator()(const board_state& board) const {
        // TODO: Implement
        return 0;
    }
};

struct board_state_equality {
    bool operator()(const board_state& board1, const board_state& board2) const {
        for(std::uint8_t rank = 0; rank < 8; rank++) {
            for(std::uint8_t file = 0; file < 8; file++) {
                if(board1.pieces[rank][file] != board2.pieces[rank][file]) return false;
            }
        }

        return true;
    }
};

struct previous_board_states {
    std::unordered_multiset<board_state, board_state_hasher, board_state_equality> encountered_board_states;

    bool draw_allowed = false;

    void add_board_state(const board_state& board_state) {
        encountered_board_states.insert(board_state);

        if(encountered_board_states.count(board_state) >= 3) draw_allowed = true;
    }
};

}