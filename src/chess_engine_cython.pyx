# src/chess_engine_cython.pyx
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from libc.stddef cimport size_t, ptrdiff_t # Use ptrdiff_t for addresses
from libc.stdint cimport uint8_t, uint32_t, int64_t, int8_t
import sys
import os
import traceback
import cython # To use cython.bints


# --- Python Enums ---
class Player: WHITE = 0; BLACK = 1
class PieceType: NONE=0; PAWN=1; KNIGHT=2; BISHOP=3; ROOK=4; QUEEN=5; KING=6
class MoveType: NORMAL_MOVE=0; CAPTURE=1; CASTLE=2; EN_PASSANT=3; PROMOTION=4; RESIGN=5; CLAIM_DRAW=6
class GameStatus: NORMAL=0; DRAW=1; CHECKMATE=2; RESIGNED=3; DRAW_BY_REPETITION=4

# --- C Definitions ---
cdef extern from "chess_cpp/chess_rules.hpp" namespace "chess":
    ctypedef struct board_position: pass
    ctypedef struct piece: pass
    ctypedef struct chess_move: pass
    ctypedef struct board_state: pass
    ctypedef int piece_type
    ctypedef int move_type
    ctypedef int player
    ctypedef int game_status

ctypedef board_position CBoardPosition
ctypedef piece CPiece
ctypedef chess_move CChessMove
ctypedef board_state CBoardState
ctypedef piece_type CPieceType
ctypedef move_type CMoveType
ctypedef player CPlayer
ctypedef game_status CGameStatus

# --- C Function Declarations ---
cdef extern from "chess_cpp/python_api.hpp":
    void* engine_create(const char* model_path) nogil
    void engine_destroy(void* engine_handle_opaque) nogil
    CBoardState* engine_get_board_state(void* engine_handle_opaque) nogil
    size_t engine_get_valid_moves(void* engine_handle_opaque, CChessMove* out_moves_buffer, size_t buffer_capacity) nogil
    CBoardState* engine_apply_move(void* engine_handle_opaque, const CChessMove* move) nogil
    CBoardState* engine_ai_move(void* engine_handle_opaque, int difficulty, void* callback) nogil
    cython.bint engine_move_to_str(void* engine_handle_opaque, const CChessMove* move, char* buffer, size_t buffer_size) nogil
    void engine_cancel_search(void* engine_handle_opaque) nogil
    double engine_evaluate_board(void* engine_handle_opaque) nogil 
    void engine_reset_engine(void* engine_handle_opaque) nogil
pass

# --- Python Wrapper Classes ---
cdef class BoardPosition:
    cdef CBoardPosition c_pos

    def __cinit__(self, int rank, int file):
        if not (0 <= rank < 8 and 0 <= file < 8): 
            raise ValueError("Rank/file out of bounds")
        self.c_pos.rank = <uint8_t>rank
        self.c_pos.file = <uint8_t>file

    property rank:
        def __get__(self): # Correctly indented under property
            return self.c_pos.rank

    property file:
        def __get__(self): # Correctly indented under property
            return self.c_pos.file

    def __repr__(self): # Correctly indented at class level
        return f"BoardPosition(rank={self.rank}, file={self.file})"

    def __eq__(self, other): # Correctly indented at class level
        if not isinstance(other, BoardPosition):
            return NotImplemented
        return self.rank == other.rank and self.file == other.file

    def __hash__(self): # Correctly indented at class level
        return hash((self.rank, self.file))


cdef class Piece:
    cdef int _type
    cdef int _player

    def __init__(self, int ptype, int player):
        self._type = ptype
        self._player = player

    property type:
        def __get__(self): # Correctly indented under property
            return self._type

    property player:
        def __get__(self): # Correctly indented under property
            return self._player

    def __repr__(self): # Correctly indented at class level
        return f"Piece(type={self.type}, player={self.player})"


cdef class ChessMove:
    cdef CChessMove c_move
    cdef BoardPosition _start_pos_py
    cdef BoardPosition _target_pos_py
    cdef int _type_py
    cdef int _promotion_py

    def __cinit__(self, int move_type, BoardPosition start, BoardPosition target, int promotion=PieceType.NONE): # Correctly indented
        if start is None: 
            TypeError("ChessMove start is None")
        if target is None:
            raise TypeError("ChessMove target is None")
        self._type_py = move_type
        self._start_pos_py = start
        self._target_pos_py = target
        self._promotion_py = promotion
        self.c_move.type = <CMoveType>move_type
        self.c_move.start_position = start.c_pos
        self.c_move.target_position = target.c_pos
        self.c_move.promotion_target = <CPieceType>promotion

    property type:
        def __get__(self):
             return self._type_py

    property start:
        def __get__(self):
             return self._start_pos_py

    property target:
        def __get__(self):
             return self._target_pos_py

    property promotion:
        def __get__(self):
             return self._promotion_py

    def __repr__(self): # Correctly indented at class level
        promo_str = f", promotion={self.promotion}" if self.type == MoveType.PROMOTION else ""
        return f"ChessMove(type={self.type}, start={self.start!r}, target={self.target!r}{promo_str})"


# --- ChessEngine Class ---
cdef class ChessEngine:
    cdef void* c_engine_handle
    cdef cython.bint _handle_valid
    cdef CChessMove* c_moves_buffer
    cdef size_t c_moves_buffer_capacity

    def __cinit__(self, str library_path, str model_path):
        self.c_engine_handle = NULL
        self._handle_valid = False
        self.c_moves_buffer = NULL
        self.c_moves_buffer_capacity = 256
        print(f"Cython ChessEngine: Initializing...")
        cdef bytes model_path_bytes = model_path.encode('utf-8')
        cdef const char* c_model_path = model_path_bytes
        try:
            self.c_engine_handle = engine_create(c_model_path)
            if self.c_engine_handle == NULL:
                raise MemoryError("engine_create returned NULL")
            self._handle_valid = True
            print(f"Cython ChessEngine: Init OK (handle: {<ptrdiff_t>self.c_engine_handle}).")
            self.c_moves_buffer = <CChessMove*>malloc(sizeof(CChessMove) * self.c_moves_buffer_capacity)
            if self.c_moves_buffer == NULL:
                print("[ERROR CYTHON] Failed to allocate internal move buffer!", file=sys.stderr)
                if self._handle_valid: 
                    with nogil: engine_destroy(self.c_engine_handle)
                raise MemoryError("Failed to allocate internal move buffer")
            print(f"Cython ChessEngine: Move buffer allocated at {<ptrdiff_t>self.c_moves_buffer}.")
        except Exception as e:
            print(f"Cython ChessEngine: Error during __cinit__: {e}", file=sys.stderr)
            if self.c_moves_buffer != NULL: 
                free(self.c_moves_buffer)
                self.c_moves_buffer = NULL
            if self._handle_valid and self.c_engine_handle != NULL: 
                with nogil: engine_destroy(self.c_engine_handle)
            self.c_engine_handle = NULL
            self._handle_valid = False 

    def __dealloc__(self):
        if self._handle_valid and self.c_engine_handle != NULL:
            with nogil: engine_destroy(self.c_engine_handle)
            self.c_engine_handle = NULL
            self._handle_valid = False
        if self.c_moves_buffer != NULL:
            free(self.c_moves_buffer)
            self.c_moves_buffer = NULL

    def _check_handle(self):
        if not self._handle_valid or self.c_engine_handle == NULL:
            raise RuntimeError("Chess engine handle invalid.")
        if self.c_moves_buffer == NULL:
            raise RuntimeError("Chess engine internal move buffer invalid (NULL).")

    @property
    def board_state_address(self) -> ptrdiff_t:
        cdef CBoardState* c_state_ptr
        self._check_handle()
        with nogil: c_state_ptr = engine_get_board_state(self.c_engine_handle)
        return <ptrdiff_t>c_state_ptr if c_state_ptr != NULL else 0

    def reset(self):
        """Resets the C++ engine state to the initial position."""
        self._check_handle()
        print("Cython ChessEngine: Requesting internal engine reset...")
        with nogil: engine_reset_engine(self.c_engine_handle)
        print("Cython ChessEngine: Internal reset request sent via C API.")

    def get_valid_moves_address_count(self) -> tuple[ptrdiff_t, int]:
        self._check_handle()
        cdef size_t num_moves_found = 0
        with nogil: num_moves_found = engine_get_valid_moves(self.c_engine_handle, self.c_moves_buffer, self.c_moves_buffer_capacity)
        if num_moves_found > self.c_moves_buffer_capacity:
            num_moves_found = self.c_moves_buffer_capacity
        return <ptrdiff_t>self.c_moves_buffer, <int>num_moves_found

    def apply_move(self, ptrdiff_t move_address) -> ptrdiff_t:
        self._check_handle()
        if move_address == 0: 
            raise ValueError("Cannot apply NULL move address")
        cdef const CChessMove* move_ptr = <const CChessMove*>move_address
        cdef CBoardState* new_state_ptr = NULL
        with nogil: new_state_ptr = engine_apply_move(self.c_engine_handle, move_ptr)
        if new_state_ptr == NULL: 
            print("[ERROR CYTHON apply_move] engine_apply_move returned NULL.", file=sys.stderr)
            return 0
        return <ptrdiff_t>new_state_ptr

    def evaluate_board(self) -> float:
        """
        Evaluate the current board state using the NNUE evaluator.
        Returns a float representing the evaluation score.
        """
        self._check_handle()  # Ensure the engine handle is valid
        cdef double evaluation = 0.0
        try:
            with nogil:
                evaluation = engine_evaluate_board(self.c_engine_handle)
            return evaluation
        except Exception as e:
            print(f"[ERROR CYTHON evaluate_board] {e}", file=sys.stderr)
            return 0.0  # Return a default value on error

    def ai_move(self, int difficulty) -> ptrdiff_t:
        self._check_handle()
        cdef CBoardState* new_state_ptr = NULL
        try:
            with nogil: new_state_ptr = engine_ai_move(self.c_engine_handle, difficulty, NULL)
            if new_state_ptr == NULL:
                print("[WARNING CYTHON] engine_ai_move returned NULL.", file=sys.stderr)
                return 0
            return <ptrdiff_t>new_state_ptr
        except Exception as e: 
            print(f"Cython ai_move error: {e}", file=sys.stderr)
            traceback.print_exc()
            return 0

    def move_to_str(self, ptrdiff_t move_address):
        self._check_handle()
        if move_address == 0:
            return "ERR! (Null Address)"
        cdef const CChessMove* move_ptr = <const CChessMove*>move_address
        cdef char buffer[16]
        cdef cython.bint success = 0
        memset(buffer, 0, sizeof(buffer))
        try:
            with nogil: success = engine_move_to_str(self.c_engine_handle, move_ptr, buffer, sizeof(buffer))
            if not success: return buffer.decode('utf-8', errors='ignore')[:sizeof(buffer)-1]
            return buffer.decode('utf-8', errors='replace')
        except Exception as e: 
            print(f"Cython move_to_str error: {e}", file=sys.stderr)
            return "ERR!"

    def stop_search(self):
        if not self._handle_valid or self.c_engine_handle == NULL: return
        with nogil: engine_cancel_search(self.c_engine_handle)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_value, tb): pass