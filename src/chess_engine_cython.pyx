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
    # Engine function now fills OUR buffer
    size_t engine_get_valid_moves(void* engine_handle_opaque, CChessMove* out_moves_buffer, size_t buffer_capacity) nogil
    # Apply move now takes only the move pointer, modifies internal state
    cython.bint engine_apply_move(void* engine_handle_opaque, const CChessMove* move, CBoardState* out_ignored_state) nogil # Last arg ignored
    cython.bint engine_ai_move(void* engine_handle_opaque, int difficulty, void* callback) nogil # Callback is NULL
    cython.bint engine_move_to_str(void* engine_handle_opaque, const CChessMove* move, char* buffer, size_t buffer_size) nogil
    void engine_cancel_search(void* engine_handle_opaque) nogil

# --- Python Wrapper Classes (Unchanged, but not directly used by apply_move/move_to_str anymore) ---

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


# --- ChessEngine Class ---
cdef class ChessEngine:
    cdef void* c_engine_handle
    cdef cython.bint _handle_valid
    # --- Buffer members defined ---
    cdef CChessMove* c_moves_buffer           # Pointer to allocated C move buffer
    cdef size_t c_moves_buffer_capacity       # Capacity of the allocated buffer

    def __cinit__(self, str library_path, str model_path):
        # Default initialization
        self.c_engine_handle = NULL
        self._handle_valid = False
        self.c_moves_buffer = NULL # Initialize buffer pointer to NULL
        self.c_moves_buffer_capacity = 256 # Example capacity, adjust if needed

        print(f"Cython ChessEngine: Initializing...")
        cdef bytes model_path_bytes = model_path.encode('utf-8')
        cdef const char* c_model_path = model_path_bytes
        try:
            # --- Initialize C++ Engine Handle ---
            self.c_engine_handle = engine_create(c_model_path)
            if self.c_engine_handle == NULL:
                raise MemoryError("engine_create returned NULL")
            self._handle_valid = True
            # Use ptrdiff_t for printing pointer address robustly
            print(f"Cython ChessEngine: Init OK (handle: {<ptrdiff_t>self.c_engine_handle}).")

            # --- Allocate internal C buffer for moves ---
            print(f"Cython ChessEngine: Allocating move buffer (capacity={self.c_moves_buffer_capacity})...")
            self.c_moves_buffer = <CChessMove*>malloc(sizeof(CChessMove) * self.c_moves_buffer_capacity)
            if self.c_moves_buffer == NULL:
                print("[ERROR CYTHON] Failed to allocate internal move buffer!", file=sys.stderr)
                if self._handle_valid and self.c_engine_handle != NULL:
                    with nogil: engine_destroy(self.c_engine_handle)
                self._handle_valid = False
                self.c_engine_handle = NULL
                raise MemoryError("Failed to allocate internal move buffer in ChessEngine __cinit__")
            # Use ptrdiff_t for printing pointer address robustly
            print(f"Cython ChessEngine: Move buffer allocated successfully at address {<ptrdiff_t>self.c_moves_buffer}.")

        except Exception as e:
            print(f"Cython ChessEngine: Error during __cinit__: {e}", file=sys.stderr)
            # Cleanup partially created resources
            if self.c_moves_buffer != NULL: # Free buffer if allocated before another error
                free(self.c_moves_buffer)
                self.c_moves_buffer = NULL
            if self._handle_valid and self.c_engine_handle != NULL: # Destroy handle if created before error
                with nogil: engine_destroy(self.c_engine_handle)
            self.c_engine_handle = NULL
            self._handle_valid = False
            raise # Re-raise the exception


    def __dealloc__(self):
        # --- Ensure BOTH handle and buffer are freed ---
        if self._handle_valid and self.c_engine_handle != NULL:
            # Use ptrdiff_t for printing pointer address robustly
            print(f"Cython ChessEngine: Destroying handle: {<ptrdiff_t>self.c_engine_handle}.")
            with nogil: engine_destroy(self.c_engine_handle)
            self.c_engine_handle = NULL
            self._handle_valid = False
        # Free buffer if it was allocated
        if self.c_moves_buffer != NULL:
            # Use ptrdiff_t for printing pointer address robustly
            print(f"Cython ChessEngine: Freeing internal move buffer at address {<ptrdiff_t>self.c_moves_buffer}.")
            free(self.c_moves_buffer)
            self.c_moves_buffer = NULL


    def _check_handle(self):
        if not self._handle_valid or self.c_engine_handle == NULL:
            raise RuntimeError("Chess engine handle is invalid.")
        # Check if buffer pointer is NULL
        if self.c_moves_buffer == NULL:
            raise RuntimeError("Chess engine internal move buffer is invalid (NULL).")

    @property
    def board_state_address(self):
        """Returns the memory address of the current C board_state struct."""
        cdef CBoardState* c_state_ptr
        self._check_handle() # Only checks handle validity now
        # print("[DEBUG CYTHON] Entering board_state_address getter", file=sys.stderr) # Optional
        with nogil: # Ensure C call is nogil if appropriate
             c_state_ptr = engine_get_board_state(self.c_engine_handle)
        if c_state_ptr == NULL:
            print("[ERROR CYTHON] engine_get_board_state returned NULL", file=sys.stderr)
            return 0
        # Cast pointer address to ptrdiff_t for returning as Python int
        address = <ptrdiff_t>c_state_ptr
        # print(f"[DEBUG CYTHON] Returning board state address: {address}", file=sys.stderr) # Optional
        return address

    def get_valid_moves_address_count(self):
        """Calls C API to fill internal buffer, returns buffer address and move count."""
        self._check_handle() # Checks handle AND that buffer is not NULL
        cdef size_t num_moves_found = 0
        # print("[DEBUG CYTHON] Entering get_valid_moves_address_count", file=sys.stderr) # Optional

        # Call C function to fill the engine's internal buffer
        with nogil: # Ensure C call is nogil if appropriate
             num_moves_found = engine_get_valid_moves(
                 self.c_engine_handle,
                 self.c_moves_buffer, # Use the allocated buffer
                 self.c_moves_buffer_capacity
             )

        if num_moves_found > self.c_moves_buffer_capacity:
            # This case indicates the C function *found* more moves than could fit.
            # The buffer will contain only the first `c_moves_buffer_capacity` moves.
            print(f"[WARNING CYTHON] Moves found ({num_moves_found}) exceeds buffer capacity ({self.c_moves_buffer_capacity}). Buffer truncated by C API.", file=sys.stderr)
            # The number of moves *actually in the buffer* is the capacity.
            num_moves_found = self.c_moves_buffer_capacity

        # Get the address of the buffer
        address = <ptrdiff_t>self.c_moves_buffer # Address of our Cython-managed buffer
        count = <int>num_moves_found

        # This should now print a non-zero address
        # print(f"[DEBUG CYTHON] Returning move buffer address={address}, count={count}", file=sys.stderr)
        return address, count


    def apply_move(self, ptrdiff_t move_address):
        """Applies move using address of a CChessMove struct from the internal buffer."""
        self._check_handle()
        if move_address == 0: raise ValueError("Cannot apply NULL move address")

        # Cast the integer address back to a C pointer
        cdef const CChessMove* move_ptr = <const CChessMove*>move_address
        cdef cython.bint success = 0
        # Dummy state to satisfy C API signature, content will be ignored
        cdef CBoardState ignored_state

        # print(f"[DEBUG CYTHON apply_move] Applying move at address {move_address}")
        # Call C API - it modifies the engine's internal state directly
        with nogil:
            success = engine_apply_move(self.c_engine_handle, move_ptr, &ignored_state)

        if not success:
            # Consider if C API provides more error info (e.g., invalid move address)
            raise ValueError("engine_apply_move returned false (invalid move or internal error).")
        # print(f"[DEBUG CYTHON apply_move] engine_apply_move successful.")

        # Return True/False to indicate success/failure to the Python caller
        return bool(success)

    def ai_move(self, int difficulty):
        """Triggers the AI calculation (blocking). Returns success/fail."""
        self._check_handle()
        cdef cython.bint success = 0
        # print(f"[DEBUG CYTHON ai_move] Calling engine_ai_move difficulty={difficulty}")
        try:
            # This function modifies the internal state directly
            with nogil: success = engine_ai_move(self.c_engine_handle, difficulty, NULL)
            if not success:
                 print("[WARNING CYTHON] engine_ai_move returned false.", file=sys.stderr)
            # else:
            #      print("[DEBUG CYTHON ai_move] engine_ai_move successful.")
            return bool(success) # Return Python bool
        except Exception as e:
            print(f"Cython ai_move error: {e}", file=sys.stderr)
            traceback.print_exc()
            return False

    def move_to_str(self, ptrdiff_t move_address):
        """Converts move data at the specified address to string."""
        self._check_handle()
        if move_address == 0: return "ERR! (Null Address)"

        # Cast the integer address back to a C pointer
        cdef const CChessMove* move_ptr = <const CChessMove*>move_address
        cdef char buffer[16] # Small buffer for SAN/UCI
        cdef cython.bint success = 0
        memset(buffer, 0, sizeof(buffer)) # Clear buffer

        try:
            with nogil: # Ensure C call is nogil if appropriate
                success = engine_move_to_str(self.c_engine_handle, move_ptr, buffer, sizeof(buffer))

            if not success:
                print("[Warning CYTHON move_to_str] engine_move_to_str failed or buffer too small.", file=sys.stderr)
                # Attempt to decode what might be in the buffer anyway
                return buffer.decode('utf-8', errors='ignore')[:sizeof(buffer)-1]

            # Decode successfully populated buffer
            return buffer.decode('utf-8', errors='replace')

        except Exception as e:
            print(f"Cython move_to_str error: {e}", file=sys.stderr)
            return "ERR!"

    def stop_search(self):
        """Signals the C++ engine to stop the current AI search."""
        self._check_handle()
        # print("Cython ChessEngine: Requesting AI search cancellation...")
        with nogil: engine_cancel_search(self.c_engine_handle)
        # print("Cython ChessEngine: Cancel request sent via C API.")

    # Context manager methods (Unchanged)
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_value, tb):
        # __dealloc__ handles cleanup, don't need anything here usually
        # print("Cython ChessEngine: Exiting context.")
        pass