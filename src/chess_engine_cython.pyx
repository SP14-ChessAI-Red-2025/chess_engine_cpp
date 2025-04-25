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
import cython # To use cython.bints


# --- Python Enums ---
class Player: WHITE = 0; BLACK = 1
class PieceType: NONE=0; PAWN=1; KNIGHT=2; BISHOP=3; ROOK=4; QUEEN=5; KING=6
class MoveType: NORMAL_MOVE=0; CAPTURE=1; CASTLE=2; EN_PASSANT=3; PROMOTION=4; RESIGN=5; CLAIM_DRAW=6
class GameStatus: NORMAL=0; DRAW=1; CHECKMATE=2; RESIGNED=3; DRAW_BY_REPETITION=4

# --- C Definitions ---
# Use ctypedef to alias the C++ structs/enums for easier use in Cython
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

# --- C Function Declarations (from python_api.hpp) ---
cdef extern from "chess_cpp/python_api.hpp":
    # Declare the C functions we will call
    void* engine_create(const char* model_path) nogil
    void engine_destroy(void* engine_handle_opaque) nogil
    CBoardState* engine_get_board_state(void* engine_handle_opaque) nogil
    size_t engine_get_valid_moves(void* engine_handle_opaque, CChessMove* out_moves_buffer, size_t buffer_capacity) nogil
    cython.bint engine_apply_move(void* engine_handle_opaque, const CChessMove* move, CBoardState* out_board_state) nogil
    cython.bint engine_ai_move(void* engine_handle_opaque, int difficulty, void* callback) nogil # Callback is NULL
    cython.bint engine_move_to_str(void* engine_handle_opaque, const CChessMove* move, char* buffer, size_t buffer_size) nogil
    void engine_cancel_search(void* engine_handle_opaque) nogil

# --- Python Wrapper Classes ---

cdef class BoardPosition:
    cdef CBoardPosition c_pos
    def __cinit__(self, int rank, int file):
        if not (0 <= rank < 8 and 0 <= file < 8): raise ValueError("Rank/file out of bounds")
        self.c_pos.rank = <uint8_t>rank
        self.c_pos.file = <uint8_t>file

    property rank:
        def __get__(self): return self.c_pos.rank
    property file:
        def __get__(self): return self.c_pos.file

    def __repr__(self): return f"BoardPosition(rank={self.rank}, file={self.file})"
    def __eq__(self, other):
        if not isinstance(other, BoardPosition): return NotImplemented
        return self.rank == other.rank and self.file == other.file
    def __hash__(self): return hash((self.rank, self.file))

cdef class Piece:
    cdef int _type
    cdef int _player
    def __init__(self, int ptype, int player):
        self._type = ptype; self._player = player

    property type:
        def __get__(self): return self._type
    property player:
        def __get__(self): return self._player

    def __repr__(self): return f"Piece(type={self.type}, player={self.player})"

cdef class ChessMove:
    cdef CChessMove c_move
    cdef BoardPosition _start_pos_py
    cdef BoardPosition _target_pos_py
    cdef int _type_py
    cdef int _promotion_py

    def __cinit__(self, int move_type, BoardPosition start, BoardPosition target, int promotion=PieceType.NONE):
        if start is None: raise TypeError("ChessMove start is None")
        if target is None: raise TypeError("ChessMove target is None")
        self._type_py = move_type
        self._start_pos_py = start
        self._target_pos_py = target
        self._promotion_py = promotion
        self.c_move.type = <CMoveType>move_type
        self.c_move.start_position = start.c_pos
        self.c_move.target_position = target.c_pos
        self.c_move.promotion_target = <CPieceType>promotion

    property type:
         def __get__(self): return self._type_py
    property start:
         def __get__(self): return self._start_pos_py
    property target:
         def __get__(self): return self._target_pos_py
    property promotion:
         def __get__(self): return self._promotion_py

    def __repr__(self):
        promo_str = f", promotion={self.promotion}" if self.type == MoveType.PROMOTION else ""
        return f"ChessMove(type={self.type}, start={self.start!r}, target={self.target!r}{promo_str})"


cdef class ChessEngine:
    cdef void* c_engine_handle
    cdef cython.bint _handle_valid
    cdef CChessMove* c_moves_buffer
    cdef size_t c_moves_buffer_capacity

    def __cinit__(self, str library_path, str model_path):
        cdef bytes model_path_bytes = model_path.encode('utf-8')
        cdef const char* c_model_path = model_path_bytes
        self.c_engine_handle = NULL
        self._handle_valid = False
        print(f"Cython ChessEngine: Initializing...")
        
        self.c_moves_buffer = NULL
        self.c_moves_buffer_capacity = 256 # Default capacity, can be adjusted
        try:
            self.c_engine_handle = engine_create(c_model_path)
            if self.c_engine_handle == NULL:
                raise MemoryError("engine_create returned NULL")
            self._handle_valid = True
            print(f"Cython ChessEngine: Init OK (handle: {<Py_ssize_t>self.c_engine_handle}).")
        except Exception as e:
            print(f"Cython ChessEngine: Error during __cinit__/engine_create: {e}", file=sys.stderr)
            if self._handle_valid and self.c_engine_handle != NULL:
                with nogil: engine_destroy(self.c_engine_handle)
            raise # Re-raise error

    def __dealloc__(self):
        if self._handle_valid and self.c_engine_handle != NULL:
            print(f"Cython ChessEngine: Destroying (handle: {<Py_ssize_t>self.c_engine_handle}).")
            with nogil: engine_destroy(self.c_engine_handle)
            self.c_engine_handle = NULL
            self._handle_valid = False

    def _check_handle(self):
        if not self._handle_valid or self.c_engine_handle == NULL:
            raise RuntimeError("Chess engine handle is invalid.")

    @property
    def board_state_address(self): # Rename property for clarity
        """Returns the memory address of the current C board_state struct."""
        cdef CBoardState* c_state_ptr
        self._check_handle()
        print("[DEBUG CYTHON] Entering board_state_address getter", file=sys.stderr)
        c_state_ptr = engine_get_board_state(self.c_engine_handle)
        if c_state_ptr == NULL:
            # Maybe return 0 or raise specific error? Returning 0 for now.
            print("[ERROR CYTHON] engine_get_board_state returned NULL", file=sys.stderr)
            return 0
        # Cast the pointer address to a Python integer (Py_ssize_t is suitable)
        address = <Py_ssize_t>c_state_ptr
        print(f"[DEBUG CYTHON] Returning address: {address}", file=sys.stderr)
        return address

    def get_valid_moves_address_count(self):
        """Calls C API to fill internal buffer, returns buffer address and move count."""
        self._check_handle()
        cdef size_t num_moves_found = 0
        print("[DEBUG CYTHON] Entering get_valid_moves_address_count", file=sys.stderr) # Debug

        # Call C function to fill the engine's internal buffer
        num_moves_found = engine_get_valid_moves(
            self.c_engine_handle,
            self.c_moves_buffer, # Use the buffer owned by the object
            self.c_moves_buffer_capacity
        )

        # Optional: Add logic here to reallocate buffer if num_moves_found > capacity
        # For now, we assume capacity is sufficient or C func handles overflow gracefully.
        if num_moves_found > self.c_moves_buffer_capacity:
            print(f"[WARNING CYTHON] Moves found ({num_moves_found}) exceeds buffer capacity ({self.c_moves_buffer_capacity}). Truncating.", file=sys.stderr)
            num_moves_found = self.c_moves_buffer_capacity

        # Return the address of the buffer and the count
        address = <ptrdiff_t>self.c_moves_buffer
        count = <int>num_moves_found # Cast size_t to int for Python
        print(f"[DEBUG CYTHON] Returning move buffer address={address}, count={count}", file=sys.stderr) # Debug
        return address, count

    def apply_move(self, ChessMove move): # This still takes a Cython ChessMove object? We might need to change this too.
         # TODO: This method might need refactoring if the Python side now
         # works primarily with ctypes moves or dictionaries representing moves.
         # For now, assuming it can still somehow get a CChessMove struct.
         self._check_handle()
         if not isinstance(move, ChessMove): raise TypeError("Requires ChessMove object.")
         cdef CBoardState c_new_state # Need to define CBoardState here
         cdef cython.bint success = 0
         # engine_apply_move expects const CChessMove* - getting it from 'move' (Cython obj)
         with nogil: success = engine_apply_move(self.c_engine_handle, &move.c_move, &c_new_state)
         if not success: raise ValueError("engine_apply_move returned false.")
         # Return new state - use pointer version?
         return <ptrdiff_t>&c_new_state # TEMPORARY - returning address of stack var is BAD! Needs fixing.

    def ai_move(self, int difficulty):
        """Triggers the AI calculation (blocking) and returns the new board state dict."""
        self._check_handle()
        cdef cython.bint success = 0
        cdef CBoardState c_state_after_ai
        cdef CBoardState* c_state_ptr
        try:
             with nogil: success = engine_ai_move(self.c_engine_handle, difficulty, NULL)
             if not success: print("Warning: engine_ai_move returned false.", file=sys.stderr)
             # Fetch the updated state AFTER ai_move completed
             with nogil: c_state_ptr = engine_get_board_state(self.c_engine_handle)
             if c_state_ptr == NULL: raise RuntimeError("engine_get_board_state returned NULL after AI move.")
             memcpy(&c_state_after_ai, c_state_ptr, sizeof(CBoardState))
             # Convert updated C state to Python dict
             py_state = {}
             py_pieces = []
             for r in range(8):
                 row_list = []
                 for f in range(8):
                     c_piece = c_state_after_ai.pieces[r][f]
                     row_list.append({'type': <int>c_piece.type, 'player': <int>c_piece.piece_player})
                 py_pieces.append(row_list)
             py_state['pieces'] = py_pieces
             py_state['current_player'] = <int>c_state_after_ai.current_player
             py_state['can_castle'] = [bool(c_state_after_ai.can_castle[i]) for i in range(4)]
             py_state['en_passant_valid'] = [bool(c_state_after_ai.en_passant_valid[i]) for i in range(16)]
             py_state['turns_since_last_capture_or_pawn'] = c_state_after_ai.turns_since_last_capture_or_pawn
             py_state['status'] = <int>c_state_after_ai.status
             py_state['can_claim_draw'] = bool(c_state_after_ai.can_claim_draw)
             py_state['in_check'] = [bool(c_state_after_ai.in_check[i]) for i in range(2)]
             return py_state
        except Exception as e:
             print(f"Cython ai_move error: {e}", file=sys.stderr)
             traceback.print_exc()
             raise

    def move_to_str(self, move_address, move_type, start_rank, start_file, target_rank, target_file, promotion): # Example: pass data instead of object
         """Converts move data (obtained via ctypes) to string."""
         # TODO: Refactor this to accept raw data or a ctypes object/address
         # For now, it's incompatible with the new approach.
         pass # Placeholder

    # <<< NEW METHOD TO EXPOSE CANCELLATION >>>
    def stop_search(self):
        """Signals the C++ engine to stop the current AI search."""
        self._check_handle() # Ensure handle is valid
        print("Cython ChessEngine: Requesting AI search cancellation...")
        # Call the C function - release the GIL as it might involve C++ atomics/mutexes
        with nogil:
            engine_cancel_search(self.c_engine_handle)
        print("Cython ChessEngine: Cancel request sent via C API.")
    # <<< END NEW METHOD >>>

    # Context manager methods
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_value, tb):
        print("Cython ChessEngine: Exiting context.")
        pass # __dealloc__ handles cleanup
