import sys
from ctypes import *

class BoardPosition(Structure):
    _fields_ = [("rank", c_uint8), ("file", c_uint8)]

class ChessMove(Structure):
    _fields_ = [("start_position", BoardPosition), ("target_position", BoardPosition)]

class Piece(Structure):
    _fields_ = [("piece_type", c_int), ("player", c_int)]

class BoardState(Structure):
    _fields_ = [("pieces", (Piece * 8) * 8), ("has_castled", c_bool * 2), ("in_check", c_bool * 2), ("current_player", c_int)]


def main():
    if len(sys.argv) < 2:
        print("Must specify library path on the command line")

    library_path = sys.argv[1]

    lib = CDLL(library_path)

    get_initial_board_state = lib.get_initial_board_state
    get_initial_board_state.restype = BoardState

    get_valid_moves = lib.get_valid_moves

    get_valid_moves.argtypes = [BoardState, POINTER(c_size_t)]
    get_valid_moves.restype = POINTER(ChessMove)


    free_moves = lib.free_moves

    free_moves.argtypes = [POINTER(ChessMove)]
    free_moves.restype = None


    size = c_size_t()

    board_state = get_initial_board_state()

    valid_moves = get_valid_moves(board_state, byref(size))

    valid_moves_array = (ChessMove * size.value).from_address(addressof(valid_moves.contents))

    def print_move(cm):
        coords = [cm.start_position.rank, cm.start_position.file, cm.target_position.rank, cm.target_position.file]
        print(f"({coords[0]}, {coords[1]}) -> ({coords[2]}, {coords[3]})")

    for move in valid_moves_array:
        print_move(move)

    free_moves(valid_moves)


if __name__ == "__main__":
    main()
