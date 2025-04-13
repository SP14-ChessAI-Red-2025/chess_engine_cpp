from ctypes import *

from collections.abc import Callable

from typing import Any, Self

class BoardPosition(Structure):
    _fields_ = [("rank", c_uint8), ("file", c_uint8)]

    def __str__(self) -> str:
        file_str = chr(ord("a") + self.file)

        return f"{file_str}{self.rank+1}"

class ChessMove(Structure):
    _fields_ = [("type", c_int), ("start_position", BoardPosition), ("target_position", BoardPosition), ("promotion_target", c_int)]

    def __str__(self) -> str:
        # TODO: customize output for castling, promotion, resignation, and claiming a draw
        type_str_arr = ["Move", "Capture", "En passant", "Castle", "Promotion", "Claim draw", "Resign"]

        return f"{type_str_arr[self.type]}: {self.start_position} -> {self.target_position}"

class Piece(Structure):
    _fields_ = [("piece_type", c_int), ("piece_player", c_int)]

class BoardStateCType(Structure):
    _fields_ = [("pieces", (Piece * 8) * 8), ("can_castle", c_bool * 4), ("in_check", c_bool * 2), ("en_passant_valid", c_bool * 16), ("turns_since_last_capture_or_pawn", c_int), ("current_player", c_int), ("status", c_int), ("can_claim_draw", c_bool)]

class BoardState:
    board_state_impl: BoardStateCType

    def __init__(self, board_state: BoardStateCType):
        self.board_state_impl = board_state

    @property
    def pieces(self) -> Any: # I don't know how to use ctypes arrays with type hints
        return self.board_state_impl.pieces

    @property
    def status(self) -> int:
        return self.board_state_impl.status

    @property
    def current_player(self) -> int:
        return self.board_state_impl.current_player


class ChessEngine:
    # I don't know the proper way to use type hints with ctypes functions, so I use Any here

    __get_initial_board_state: Callable[[], BoardStateCType]

    __get_valid_moves: Callable[[BoardStateCType, Any], Any]
    __free_moves: Callable[[Any], None]

    __apply_move: Callable[[Any, ChessMove], None]
    __ai_move: Callable[[c_void_p, Any, int], None]

    __free_ai_state: Callable[[c_void_p], None]

    board_state: BoardState
    __ai_state: c_void_p

    def __init__(self, library_path: str) -> None:
        lib = CDLL(library_path)

        self.__get_initial_board_state = lib.get_initial_board_state
        self.__get_initial_board_state.restype = BoardStateCType

        self.__get_valid_moves = lib.get_valid_moves
        self.__get_valid_moves.argtypes = [BoardStateCType, POINTER(c_size_t)]
        self.__get_valid_moves.restype = POINTER(ChessMove)


        self.__free_moves = lib.free_moves
        self.__free_moves.argtypes = [POINTER(ChessMove)]
        self.__free_moves.restype = None


        self.__apply_move = lib.apply_move

        self.__apply_move.argtypes = [POINTER(BoardStateCType), ChessMove]
        self.__apply_move.restype = None

        init_ai_state = lib.init_ai_state
        init_ai_state.restype = c_void_p

        self.__free_ai_state = lib.free_ai_state
        self.__free_ai_state.argtypes = [c_void_p]
        self.__free_ai_state.restype = None

        self.__ai_move = lib.ai_move
        self.__ai_move.argtypes = [c_void_p, POINTER(BoardStateCType), c_int32]
        self.__ai_move.restype = None

        self.__ai_state = init_ai_state()

        self.board_state = BoardState(self.__get_initial_board_state())

    def get_valid_moves(self) -> list[ChessMove]:
        size = c_size_t()

        valid_moves_ptr = self.__get_valid_moves(self.board_state.board_state_impl, byref(size))

        if size.value == 0:
            self.__free_moves(valid_moves_ptr)

            raise Exception("Error getting list of valid moves")
        else:
            valid_move_array = (ChessMove * size.value).from_address(addressof(valid_moves_ptr.contents))

            valid_moves = []

            for move in valid_move_array:
                valid_moves.append(ChessMove())

                memmove(byref(valid_moves[-1]), byref(move), sizeof(move))

            self.__free_moves(valid_moves_ptr)

            return valid_moves

    def apply_move(self, move: ChessMove) -> None:
        self.__apply_move(byref(self.board_state.board_state_impl), move)

    def ai_move(self, difficulty: int) -> None:
        self.__ai_move(self.__ai_state, self.board_state.board_state_impl, difficulty)

    def move_to_str(self, move: ChessMove) -> str:
        piece_type = self.board_state.pieces[move.start_position.rank][move.start_position.file].piece_type
        piece_str = ["none", "pawn", "knight", "bishop", "rook", "queen", "king"][piece_type]

        return f"{piece_str}: {move}"

    def free_memory(self) -> None:
        self.__free_ai_state(self.__ai_state)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.free_memory()