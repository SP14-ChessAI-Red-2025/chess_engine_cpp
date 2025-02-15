import sys
from ctypes import *

class BoardPosition(Structure):
    _fields_ = [("rank", c_uint8), ("file", c_uint8)]

    def __str__(self) -> str:
        file_str = chr(ord("a") + self.file);

        return f"{file_str}{self.rank+1}"

class ChessMove(Structure):
    _fields_ = [("type", c_int), ("start_position", BoardPosition), ("target_position", BoardPosition), ("promotion_target", c_int)]

    def __str__(self) -> str:
        # TODO: customize output for castling and promotion
        type_str_arr = ["Move", "Capture", "En passant", "Castle", "Promotion"]

        return f"{type_str_arr[self.type]}: {self.start_position} -> {self.target_position}"

class Piece(Structure):
    _fields_ = [("piece_type", c_int), ("piece_player", c_int)]

class BoardState(Structure):
    _fields_ = [("pieces", (Piece * 8) * 8), ("can_castle", c_bool * 2), ("in_check", c_bool * 2), ("en_passant_valid", c_bool * 16), ("turns_since_last_capture_or_pawn", c_int), ("current_player", c_int), ("status", c_int)]

class ChessAIState(Structure):
    pass

def piece_type_to_str(piece_type: int) -> str:
    return

def move_to_str(move: ChessMove, board_state: BoardState) -> str:
    piece_type = board_state.pieces[move.start_position.rank][move.start_position.file].piece_type
    piece_str = ["none", "pawn", "knight", "bishop", "rook", "queen", "king"][piece_type]

    return f"{piece_str}: {move}"


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


    apply_move = lib.apply_move

    apply_move.argtypes = [POINTER(BoardState), ChessMove]
    apply_move.restype = None

    init_ai_state = lib.init_ai_state
    init_ai_state.restype = POINTER(ChessAIState)

    free_ai_state = lib.free_ai_state
    free_ai_state.argtypes = [c_void_p]
    free_ai_state.restype = None


    size = c_size_t()

    board_state = get_initial_board_state()

    valid_moves = get_valid_moves(board_state, byref(size))

    valid_moves_array = (ChessMove * size.value).from_address(addressof(valid_moves.contents))

    ai_state = init_ai_state()


    for move in valid_moves_array:
        print(move_to_str(move, board_state))

    free_moves(valid_moves)

    free_ai_state(ai_state)


if __name__ == "__main__":
    main()
