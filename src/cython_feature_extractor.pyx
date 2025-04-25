# src/chess_engine_cython.pyx
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from libc.stdint cimport uint8_t, uint32_t, int64_t, int8_t
import sys
import os
import traceback
import cython # To use cython.bint

# --- Python Enums ---
class Player: WHITE = 0; BLACK = 1
class PieceType: NONE=0; PAWN=1; KNIGHT=2; BISHOP=3; ROOK=4; QUEEN=5; KING=6
class MoveType: NORMAL_MOVE=0; CAPTURE=1; CASTLE=2; EN_PASSANT=3; PROMOTION=4; RESIGN=5; CLAIM_DRAW=6
class GameStatus: NORMAL=0; DRAW=1; CHECKMATE=2; RESIGNED=3; DRAW_BY_REPETITION=4

# --- C Definitions ---
ctypedef chess.board_position CBoardPosition
ctypedef chess.piece CPiece
ctypedef chess.chess_move CChessMove
ctypedef chess.board_state CBoardState
ctypedef chess.piece_type CPieceType
ctypedef chess.move_type CMoveType
ctypedef chess.player CPlayer
ctypedef chess.game_status CGameStatus

# --- C Function Declarations (from python_api.hpp) ---
cdef extern from "chess_cpp/python_api.hpp":
    void* engine_create(const char* model_path) nogil
    void engine_destroy(void* engine_handle_opaque) nogil
    CBoardState* engine_get_board_state(void* engine_handle_opaque) nogil
    size_t engine_get_valid_moves(void* engine_handle_opaque, CChessMove* out_moves_buffer, size_t buffer_capacity) nogil
    cython.bint engine_apply_move(void* engine_handle_opaque, const CChessMove* move, CBoardState* out_board_state) nogil
    cython.bint engine_ai_move(void* engine_handle_opaque, int difficulty, void* callback) nogil
    cython.bint engine_move_to_str(void* engine_handle_opaque, const CChessMove* move, char* buffer, size_t buffer_size) nogil
    # <<< DECLARE CANCELLATION & RESET FUNCTIONS >>>
    void engine_cancel_search(void* engine_handle_opaque) nogil
    void engine_reset(void* engine_handle_opaque) nogil


# --- Python Wrapper Classes ---

cdef class BoardPosition:
    cdef CBoardPosition c_pos
    def __cinit__(self, int rank, int file):
        if not (0 <= rank < 8 and 0 <= file < 8): raise ValueError("Rank/file out of bounds")
        self.c_pos.rank = <uint8_t>rank; self.c_pos.file = <uint8_t>file
    property rank: def __get__(self): return self.c_pos.rank
    property file: def __get__(self): return self.c_pos.file
    def __repr__(self): return f"BoardPosition(rank={self.rank}, file={self.file})"
    def __eq__(self, other):
        if not isinstance(other, BoardPosition): return NotImplemented
        return self.rank == other.rank and self.file == other.file
    def __hash__(self): return hash((self.rank, self.file))

cdef class Piece:
    cdef int _type; cdef int _player
    def __init__(self, int ptype, int player): self._type = ptype; self._player = player
    property type: def __get__(self): return self._type
    property player: def __get__(self): return self._player
    def __repr__(self): return f"Piece(type={self.type}, player={self.player})"

cdef class ChessMove:
    cdef CChessMove c_move
    cdef BoardPosition _start_pos_py; cdef BoardPosition _target_pos_py
    cdef int _type_py; cdef int _promotion_py
    def __cinit__(self, int move_type, BoardPosition start, BoardPosition target, int promotion=PieceType.NONE):
        if start is None: raise TypeError("ChessMove start is None")
        if target is None: raise TypeError("ChessMove target is None")
        self._type_py = move_type; self._start_pos_py = start; self._target_pos_py = target; self._promotion_py = promotion
        self.c_move.type = <CMoveType>move_type; self.c_move.start_position = start.c_pos
        self.c_move.target_position = target.c_pos; self.c_move.promotion_target = <CPieceType>promotion
    property type: def __get__(self): return self._type_py
    property start: def __get__(self): return self._start_pos_py
    property target: def __get__(self): return self._target_pos_py
    property promotion: def __get__(self): return self._promotion_py
    def __repr__(self):
        promo_str = f", promotion={self.promotion}" if self.type == MoveType.PROMOTION else ""
        return f"ChessMove(type={self.type}, start={self.start!r}, target={self.target!r}{promo_str})"


cdef class ChessEngine:
    cdef void* c_engine_handle
    cdef cython.bint _handle_valid

    def __cinit__(self, str library_path, str model_path):
        self.c_engine_handle = NULL; self._handle_valid = False
        print(f"Cython ChessEngine: Initializing...")
        cdef bytes model_path_bytes = model_path.encode('utf-8')
        cdef const char* c_model_path = model_path_bytes
        try:
            self.c_engine_handle = engine_create(c_model_path)
            if self.c_engine_handle == NULL: raise MemoryError("engine_create returned NULL")
            self._handle_valid = True
            print(f"Cython ChessEngine: Init OK (handle: {<Py_ssize_t>self.c_engine_handle}).")
        except Exception as e:
             print(f"Cython ChessEngine: Error during __cinit__/engine_create: {e}", file=sys.stderr)
             traceback.print_exc(); self.c_engine_handle = NULL; self._handle_valid = False; raise

    def __dealloc__(self):
         if self._handle_valid and self.c_engine_handle != NULL:
            print(f"Cython ChessEngine: Destroying (handle: {<Py_ssize_t>self.c_engine_handle}).")
            with nogil: engine_destroy(self.c_engine_handle)
            self.c_engine_handle = NULL; self._handle_valid = False

    def _check_handle(self):
        if not self._handle_valid or self.c_engine_handle == NULL:
            raise RuntimeError("Chess engine handle is invalid.")

    @property
    def board_state(self):
        """Returns the current board state as a Python dictionary."""
        self._check_handle()
        cdef CBoardState* c_state_ptr
        cdef CBoardState c_state_copy
        cdef int r, f, i
        cdef CPiece c_piece
        cdef list py_pieces, row_list
        cdef dict py_state
        c_state_ptr = engine_get_board_state(self.c_engine_handle)
        if c_state_ptr == NULL: raise RuntimeError("engine_get_board_state returned NULL.")
        memcpy(&c_state_copy, c_state_ptr, sizeof(CBoardState))
        py_state = {}
        py_pieces = []
        for r in range(8):
            row_list = []
            for f in range(8):
                c_piece = c_state_copy.pieces[r][f]
                row_list.append({'type': <int>c_piece.type, 'player': <int>c_piece.piece_player})
            py_pieces.append(row_list)
        py_state['pieces'] = py_pieces
        py_state['current_player'] = <int>c_state_copy.current_player
        py_state['can_castle'] = [bool(c_state_copy.can_castle[i]) for i in range(4)]
        py_state['en_passant_valid'] = [bool(c_state_copy.en_passant_valid[i]) for i in range(16)]
        py_state['turns_since_last_capture_or_pawn'] = c_state_copy.turns_since_last_capture_or_pawn
        py_state['status'] = <int>c_state_copy.status
        py_state['can_claim_draw'] = bool(c_state_copy.can_claim_draw)
        py_state['in_check'] = [bool(c_state_copy.in_check[i]) for i in range(2)]
        return py_state

    def get_valid_moves(self):
        """Returns a list of valid ChessMove objects."""
        self._check_handle()
        cdef size_t buffer_capacity = 256
        cdef CChessMove* c_moves_buffer = <CChessMove*>malloc(sizeof(CChessMove) * buffer_capacity)
        if c_moves_buffer == NULL: raise MemoryError("Failed to allocate move buffer.")
        cdef size_t num_moves_found = 0
        cdef size_t i
        py_moves = []
        try:
            num_moves_found = engine_get_valid_moves(self.c_engine_handle, c_moves_buffer, buffer_capacity)
            if num_moves_found > buffer_capacity:
                 print(f"Warning: Num moves ({num_moves_found}) > buffer ({buffer_capacity}).", file=sys.stderr)
                 num_moves_found = buffer_capacity
            for i in range(num_moves_found):
                 c_move = c_moves_buffer[i]
                 start_pos = BoardPosition(c_move.start_position.rank, c_move.start_position.file)
                 target_pos = BoardPosition(c_move.target_position.rank, c_move.target_position.file)
                 py_move = ChessMove(<int>c_move.type, start_pos, target_pos, <int>c_move.promotion_target)
                 py_moves.append(py_move)
        finally:
            free(c_moves_buffer)
        return py_moves

    def apply_move(self, ChessMove move):
        """Applies the given move and returns the new board state dictionary."""
        self._check_handle()
        if not isinstance(move, ChessMove): raise TypeError("Requires ChessMove object.")
        cdef CBoardState c_new_state
        cdef cython.bint success = 0
        cdef int r, f, i; cdef CPiece c_piece; cdef list py_pieces, row_list; cdef dict py_state
        try:
             with nogil: success = engine_apply_move(self.c_engine_handle, &move.c_move, &c_new_state)
             if not success: raise ValueError("engine_apply_move returned false.")
             # Convert C state to Python dict
             py_state = {}; py_pieces = []
             for r in range(8):
                 row_list = [];
                 for f in range(8): c_piece = c_new_state.pieces[r][f]; row_list.append({'type': <int>c_piece.type, 'player': <int>c_piece.piece_player})
                 py_pieces.append(row_list)
             py_state['pieces'] = py_pieces; py_state['current_player'] = <int>c_new_state.current_player
             py_state['can_castle'] = [bool(c_new_state.can_castle[i]) for i in range(4)]
             py_state['en_passant_valid'] = [bool(c_new_state.en_passant_valid[i]) for i in range(16)]
             py_state['turns_since_last_capture_or_pawn'] = c_new_state.turns_since_last_capture_or_pawn
             py_state['status'] = <int>c_new_state.status; py_state['can_claim_draw'] = bool(c_new_state.can_claim_draw)
             py_state['in_check'] = [bool(c_new_state.in_check[i]) for i in range(2)]
             return py_state
        except Exception as e: print(f"Cython apply_move error: {e}", file=sys.stderr); traceback.print_exc(); raise

    def ai_move(self, int difficulty):
        """Triggers the AI calculation (blocking) and returns the new board state dict."""
        self._check_handle()
        # <<< MOVED CDEF DECLARATIONS TO THE TOP >>>
        cdef cython.bint success = 0
        cdef CBoardState c_state_after_ai
        cdef CBoardState* c_state_ptr # Now declared at the top
        cdef int r, f, i
        cdef CPiece c_piece
        cdef list py_pieces, row_list
        cdef dict py_state
        # <<< END MOVED DECLARATIONS >>>
        try:
             with nogil: success = engine_ai_move(self.c_engine_handle, difficulty, NULL)
             if not success: print("Warning: engine_ai_move returned false.", file=sys.stderr)
             # Fetch the updated state AFTER ai_move completed
             with nogil: c_state_ptr = engine_get_board_state(self.c_engine_handle)
             if c_state_ptr == NULL: raise RuntimeError("engine_get_board_state returned NULL after AI move.")
             memcpy(&c_state_after_ai, c_state_ptr, sizeof(CBoardState))
             # Convert updated C state to Python dict
             py_state = {}; py_pieces = []
             for r in range(8):
                 row_list = [];
                 for f in range(8): c_piece = c_state_after_ai.pieces[r][f]; row_list.append({'type': <int>c_piece.type, 'player': <int>c_piece.piece_player})
                 py_pieces.append(row_list)
             py_state['pieces'] = py_pieces; py_state['current_player'] = <int>c_state_after_ai.current_player
             py_state['can_castle'] = [bool(c_state_after_ai.can_castle[i]) for i in range(4)]
             py_state['en_passant_valid'] = [bool(c_state_after_ai.en_passant_valid[i]) for i in range(16)]
             py_state['turns_since_last_capture_or_pawn'] = c_state_after_ai.turns_since_last_capture_or_pawn
             py_state['status'] = <int>c_state_after_ai.status; py_state['can_claim_draw'] = bool(c_state_after_ai.can_claim_draw)
             py_state['in_check'] = [bool(c_state_after_ai.in_check[i]) for i in range(2)]
             return py_state
        except Exception as e: print(f"Cython ai_move error: {e}", file=sys.stderr); traceback.print_exc(); raise

    def move_to_str(self, ChessMove move):
        # ... (implementation as before) ...
        self._check_handle();
        if not isinstance(move, ChessMove): raise TypeError("Requires ChessMove object.")
        cdef char buffer[16]; cdef cython.bint success = 0
        memset(buffer, 0, sizeof(buffer))
        try:
             success = engine_move_to_str(self.c_engine_handle, &move.c_move, buffer, sizeof(buffer))
             if not success: print("Warning: engine_move_to_str failed...", file=sys.stderr); return buffer.decode('utf-8', errors='ignore')[:sizeof(buffer)-1]
        except Exception as e: print(f"Cython move_to_str error: {e}", file=sys.stderr); traceback.print_exc(); return "Error"
        return buffer.decode('utf-8', errors='replace')

    # <<< NEW METHOD TO EXPOSE CANCELLATION >>>
    def stop_search(self):
        """Signals the C++ engine to stop the current AI search."""
        self._check_handle()
        print("Cython ChessEngine: Requesting AI search cancellation...")
        with nogil:
            engine_cancel_search(self.c_engine_handle)
        print("Cython ChessEngine: Cancel request sent via C API.")

    # <<< NEW METHOD TO EXPOSE RESET >>>
    def reset(self):
        """Resets the C++ engine state to the initial position."""
        self._check_handle()
        print("Cython ChessEngine: Requesting internal engine reset...")
        with nogil:
            engine_reset(self.c_engine_handle)
        print("Cython ChessEngine: Internal reset request sent via C API.")

    # Context manager methods
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_value, tb):
        print("Cython ChessEngine: Exiting context.")
        pass # __dealloc__ handles cleanup
