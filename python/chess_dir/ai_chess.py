from ctypes import *
from collections.abc import Callable
from typing import Any, Self
import os

# To run, library may be different.
# python ai_example.py ../build/src/chess_cpp_pybind.cpython-312-x86_64-linux-gnu.so ../model/nnue_model.onnx

# --- CTypes Structures (Same as chess_dir.py) ---

class BoardPosition(Structure):
    _fields_ = [("rank", c_uint8), ("file", c_uint8)]

    def __str__(self) -> str:
        file_str = chr(ord("a") + self.file)
        return f"{file_str}{self.rank+1}"

class ChessMove(Structure):
    _fields_ = [("type", c_int), ("start_position", BoardPosition), ("target_position", BoardPosition), ("promotion_target", c_int)]

    def __str__(self) -> str:
        type_str_arr = ["Move", "Capture", "En passant", "Castle", "Promotion", "Claim draw", "Resign"]
        type_index = self.type
        if 0 <= type_index < len(type_str_arr):
            type_str = type_str_arr[type_index]
        else:
            type_str = "UnknownMoveType"

        if self.type == 3: # Castle
             # Determine kingside/queenside based on rook start position
             side = "Kingside" if self.start_position.file == 7 else "Queenside"
             return f"{type_str} ({side})"
        elif self.type == 4: # Promotion
             promo_piece_type = self.promotion_target
             promo_piece_str = ["?", "P", "N", "B", "R", "Q", "K"][promo_piece_type] # Map int to piece char
             return f"{type_str}: {self.start_position} -> {self.target_position}={promo_piece_str}"
        elif self.type == 5: # Draw
            return f"{type_str}"
        elif self.type == 6: # Resign
            return f"{type_str}"
        else: # Normal moves, captures, en passant
            return f"{type_str}: {self.start_position} -> {self.target_position}"


class Piece(Structure):
    _fields_ = [("piece_type", c_int), ("piece_player", c_int)]

class BoardStateCType(Structure):
    _fields_ = [("pieces", (Piece * 8) * 8),
                ("can_castle", c_bool * 4),
                ("in_check", c_bool * 2),
                ("en_passant_valid", c_bool * 16),
                ("turns_since_last_capture_or_pawn", c_int),
                ("current_player", c_int),
                ("status", c_int),
                ("can_claim_draw", c_bool)]

class BoardState:
    """Wrapper around the C board state structure."""
    board_state_impl: BoardStateCType

    def __init__(self, board_state: BoardStateCType):
        self.board_state_impl = board_state

    @property
    def pieces(self) -> Any: # Ctypes arrays are tricky with type hints
        """Access the 8x8 board pieces. pieces[rank][file]."""
        return self.board_state_impl.pieces

    @property
    def status(self) -> int:
        """Get the current game status (0: normal, 1: draw, 2: checkmate, 3: resigned)."""
        return self.board_state_impl.status

    @property
    def current_player(self) -> int:
        """Get the current player (0: white, 1: black)."""
        return self.board_state_impl.current_player

    @property
    def can_claim_draw(self) -> bool:
         """Check if the current player can claim a draw (e.g., 50-move rule)."""
         return self.board_state_impl.can_claim_draw

    @property
    def in_check(self) -> tuple[bool, bool]:
         """Check if white (index 0) or black (index 1) is in check."""
         return (self.board_state_impl.in_check[0], self.board_state_impl.in_check[1])

    @property
    def can_castle(self) -> tuple[bool, bool, bool, bool]:
         """Check castling rights [White Kingside, White Queenside, Black Kingside, Black Queenside]."""
         return (self.board_state_impl.can_castle[0], self.board_state_impl.can_castle[1],
                 self.board_state_impl.can_castle[2], self.board_state_impl.can_castle[3])

    @property
    def turns_since_last_capture_or_pawn(self) -> int:
         """Get the number of half-moves since the last capture or pawn move."""
         return self.board_state_impl.turns_since_last_capture_or_pawn


class ChessEngine:
    """
    Python wrapper for the C++ chess_dir engine library.

    Handles loading the library, managing the AI state (including the NNUE model),
    and interacting with the core chess_dir logic functions.

    Use as a context manager (`with ChessEngine(...) as engine:`) to ensure
    proper memory management of the C++ AI state.
    """
    # Type hints for C function pointers
    __get_initial_board_state: Callable[[], BoardStateCType]
    __get_valid_moves: Callable[[BoardStateCType, Any], Any]
    __free_moves: Callable[[Any], None]
    __apply_move: Callable[[Any, ChessMove], None]
    __ai_move: Callable[[c_void_p, Any, c_int32], None]
    __init_ai_state: Callable[[c_char_p], c_void_p]
    __free_ai_state: Callable[[c_void_p], None]

    board_state: BoardState
    __ai_state: c_void_p | None # Can be None if initialization fails

    def __init__(self, library_path: str, model_path: str) -> None:
        """
        Initializes the Chess Engine.

        Args:
            library_path: Path to the compiled C++ shared library
                          (e.g., 'chess_cpp_pybind.so' or 'chess_cpp_pybind.dll').
            model_path: Path to the ONNX NNUE model file (e.g., 'nnue_model.onnx').

        Raises:
            FileNotFoundError: If the library or model file does not exist.
            OSError: If the library cannot be loaded.
            RuntimeError: If the C++ AI state fails to initialize (e.g., invalid model).
        """
        if not os.path.exists(library_path):
            raise FileNotFoundError(f"Shared library not found at: {library_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found at: {model_path}")

        try:
            lib = CDLL(library_path)
        except OSError as e:
            raise OSError(f"Failed to load shared library from {library_path}: {e}")

        # --- Load C++ Functions ---
        try:
            # Board state functions
            self.__get_initial_board_state = lib.get_initial_board_state
            self.__get_initial_board_state.restype = BoardStateCType

            # Move functions
            self.__get_valid_moves = lib.get_valid_moves
            self.__get_valid_moves.argtypes = [BoardStateCType, POINTER(c_size_t)]
            self.__get_valid_moves.restype = POINTER(ChessMove)

            self.__free_moves = lib.free_moves
            self.__free_moves.argtypes = [POINTER(ChessMove)]
            self.__free_moves.restype = None

            self.__apply_move = lib.apply_move
            self.__apply_move.argtypes = [POINTER(BoardStateCType), ChessMove]
            self.__apply_move.restype = None

            # AI State functions
            self.__init_ai_state = lib.init_ai_state
            self.__init_ai_state.argtypes = [c_char_p] # Expects a C string (bytes) for model path
            self.__init_ai_state.restype = c_void_p

            self.__free_ai_state = lib.free_ai_state
            self.__free_ai_state.argtypes = [c_void_p]
            self.__free_ai_state.restype = None

            # AI move function
            self.__ai_move = lib.ai_move
            self.__ai_move.argtypes = [c_void_p, POINTER(BoardStateCType), c_int32]
            self.__ai_move.restype = None

        except AttributeError as e:
            raise AttributeError(f"Failed to find a required function in the library {library_path}. "
                                 f"Ensure the library is compiled correctly and matches the API. Missing function: {e}")

        # --- Initialize AI State ---
        model_path_bytes = model_path.encode('utf-8') # Encode path for C API
        self.__ai_state = self.__init_ai_state(model_path_bytes)
        if not self.__ai_state:
             raise RuntimeError(f"Failed to initialize C++ AI state. "
                                f"Check library compatibility and if the model file '{model_path}' is valid/loadable by the C++ backend.")

        # --- Initialize Board State ---
        self.board_state = BoardState(self.__get_initial_board_state())

    def get_valid_moves(self) -> list[ChessMove]:
        """Returns a list of all valid moves for the current board state."""
        size = c_size_t()
        valid_moves_ptr = self.__get_valid_moves(self.board_state.board_state_impl, byref(size))

        if not valid_moves_ptr:
             if size.value == 0:
                 return []
             else:
                 raise RuntimeError("C++ get_valid_moves returned a null pointer unexpectedly.")

        valid_moves = []
        try:
            if size.value > 1000:
                 raise MemoryError(f"Reported number of valid moves ({size.value}) is excessively large.")

            for i in range(size.value):
                 move_c = valid_moves_ptr[i]
                 move_py = ChessMove()
                 memmove(byref(move_py), byref(move_c), sizeof(ChessMove))
                 valid_moves.append(move_py)

        finally:
            self.__free_moves(valid_moves_ptr)

        return valid_moves

    def apply_move(self, move: ChessMove) -> None:
        """Applies the given move to the internal board state."""
        if not isinstance(move, ChessMove):
            raise TypeError("apply_move requires a ChessMove object")
        self.__apply_move(byref(self.board_state.board_state_impl), move)

    def ai_move(self, difficulty: int) -> None:
        """
        Requests the C++ AI to make a move based on the current board state.
        The difficulty parameter will influence the AI's search depth and time.
        """
        if self.__ai_state is None:
             raise RuntimeError("AI state is not initialized. Cannot perform AI move.")
        self.__ai_move(self.__ai_state, byref(self.board_state.board_state_impl), c_int32(difficulty))

    def move_to_str(self, move: ChessMove) -> str:
        """Provides a basic string representation of a move."""
        try:
            start_piece = self.board_state.pieces[move.start_position.rank][move.start_position.file]
            piece_type = start_piece.piece_type
            piece_str = ["?", "P", "N", "B", "R", "Q", "K"][piece_type]
        except (IndexError, KeyError):
            piece_str = "?"

        return f"{piece_str}: {move}"

    def free_memory(self) -> None:
        """
        Explicitly frees the C++ AI state memory.
        Called automatically when using the context manager protocol.
        """
        if self.__ai_state is not None:
            self.__free_ai_state(self.__ai_state)
            self.__ai_state = None

    def __enter__(self) -> Self:
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the runtime context related to this object, ensuring cleanup."""
        self.free_memory()
