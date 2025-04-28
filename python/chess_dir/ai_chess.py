# python/chess_dir/ai_chess.py
import time
import sys
from ctypes import (
    CDLL, Structure, POINTER, c_void_p, c_size_t, c_int8, c_uint8, c_int32,
    c_double, c_char_p, byref, create_string_buffer, c_bool, cast, sizeof,
    create_unicode_buffer, addressof, string_at, c_uint8, c_int8, c_bool, c_int32
)
import os
import platform # To potentially adjust library name
import traceback # For detailed error printing

# --- Define C structures matching C++ structs ---
# Ensure these exactly match the layout in C++ headers

class BoardPosition(Structure):
    _fields_ = [("rank", c_uint8),
                ("file", c_uint8)]
    def __repr__(self):
        return f"Pos(r={self.rank}, f={self.file})"


class Piece(Structure):
    _fields_ = [("type", c_int8), 
                ("piece_player", c_int8)]
    def __repr__(self):
        return f"Piece(t={self.type}, p={self.piece_player})"

class ChessMove(Structure):
    _fields_ = [("type", c_int8),
                ("start_position", BoardPosition),
                ("target_position", BoardPosition),
                ("promotion_target", c_int8)]
    def __repr__(self):
         return (f"Move(t={self.type}, start={self.start_position}, "
                 f"target={self.target_position}, promo={self.promotion_target})")


class BoardState(Structure):
    _fields_ = [("pieces", (Piece * 8) * 8),
                ("can_castle", c_bool * 4),
                ("in_check", c_bool * 2),
                ("en_passant_valid", c_bool * 16),
                ("turns_since_last_capture_or_pawn", c_int32),
                ("current_player", c_int8),
                ("status", c_int8),
                ("can_claim_draw", c_bool)]

# --- Enums (for reference and potential use in Python logic) ---
class PieceType:
    NONE = 0
    PAWN = 1
    KNIGHT = 2
    BISHOP = 3
    ROOK = 4
    QUEEN = 5
    KING = 6

class Player:
    WHITE = 0
    BLACK = 1

class MoveType:
    NORMAL = 0
    CAPTURE = 1
    EN_PASSANT = 2
    CASTLE = 3
    PROMOTION = 4
    CLAIM_DRAW = 5
    RESIGN = 6

class GameStatus:
    NORMAL = 0
    DRAW = 1
    CHECKMATE = 2
    RESIGNED = 3
    DRAW_BY_REPETITION = 4 # Added specific draw type

# --- ChessEngine Class ---
class ChessEngine:
    """ Python wrapper for the C++ chess engine shared library using EngineHandle. """
    def __init__(self, library_path, model_path):
        """ Initializes the ChessEngine using the EngineHandle C API. """
        self._lib = None
        self._engine_handle = None # Opaque pointer to C++ EngineHandle
        self.board_state = None    # Python ctypes BoardState structure (local copy)

        print(f"Attempting to load library: {library_path}")
        print(f"Attempting to load model: {model_path}")

        # --- Validate Paths ---
        if not os.path.exists(library_path):
            base, _ = os.path.splitext(library_path)
            system = platform.system()
            ext = ".so" # Default Linux
            if system == "Windows": ext = ".dll"
            elif system == "Darwin": ext = ".dylib"
            library_path = base + ext
            if not os.path.exists(library_path):
                raise FileNotFoundError(f"Shared library not found: {library_path}")

        if not os.path.exists(model_path):
             raise FileNotFoundError(f"NNUE model file not found: {model_path}")

        try:
            # --- Load Library ---
            self._lib = CDLL(library_path)
            print(f"Library loaded successfully from: {library_path}")

            # --- Define argtypes and restype for C functions (NEW API) ---
            self._lib.engine_create.argtypes = [c_char_p]
            self._lib.engine_create.restype = c_void_p
            self._lib.engine_destroy.argtypes = [c_void_p]
            self._lib.engine_destroy.restype = None

            self._lib.engine_get_board_state.argtypes = [c_void_p]
            self._lib.engine_get_board_state.restype = POINTER(BoardState)

            self._lib.engine_get_valid_moves.argtypes = [c_void_p, POINTER(ChessMove), c_size_t]
            self._lib.engine_get_valid_moves.restype = c_size_t # Returns number of moves found

            # === engine_apply_move signature ===
            self._lib.engine_apply_move.argtypes = [c_void_p, POINTER(ChessMove)]
            self._lib.engine_apply_move.restype = c_bool

            self._lib.engine_ai_move.argtypes = [c_void_p, c_int32, c_void_p] # Handle, difficulty, callback (ignored)
            self._lib.engine_ai_move.restype = c_bool # Returns success/failure

            self._lib.engine_cancel_search.argtypes = [c_void_p]
            self._lib.engine_cancel_search.restype = None

            # Move to string helper
            self._lib.engine_move_to_str.argtypes = [c_void_p, POINTER(ChessMove), c_char_p, c_size_t]
            self._lib.engine_move_to_str.restype = c_bool

            print("C function signatures defined (EngineHandle API).")

            # --- Initialize C++ Engine Handle ---
            model_path_bytes = model_path.encode('utf-8')
            self._engine_handle = self._lib.engine_create(model_path_bytes)
            if not self._engine_handle:
                raise RuntimeError("Failed to initialize C++ EngineHandle (engine_create returned null).")
            print(f"Engine handle initialized (pointer: {self._engine_handle}).")

            # --- Get Initial Board State ---
            self._update_local_board_state() # Fetch initial state into self.board_state
            if self.board_state is None:
                 raise RuntimeError("Failed to get initial board state from engine handle.")
            print("Initial board state obtained.")

        except Exception as e:
             print(f"Error during ChessEngine initialization: {e}")
             self.close() # Ensure cleanup on error
             raise # Re-raise the exception

    def _update_local_board_state(self):
        """ Fetches the current board state from C++ and updates the local copy. """
        if not self._lib or not self._engine_handle:
            print("Warning: Cannot update board state, engine not initialized.")
            self.board_state = None
            return

        # print("[PYTHON DEBUG _update_local] Calling engine_get_board_state...") # ADDED
        board_ptr = self._lib.engine_get_board_state(self._engine_handle)
        if board_ptr:
            # ADDED Debug prints:
            try:
                c_player_val = board_ptr.contents.current_player
                # print(f"[PYTHON DEBUG _update_local] C state pointer valid. Player value from C struct: {c_player_val}")
            except Exception as e:
                print(f"[PYTHON DEBUG _update_local] Error accessing C pointer contents: {e}")
                c_player_val = -99 # Indicate error

            # Original line:
            self.board_state = board_ptr.contents

            # ADDED Debug print:
            if self.board_state:
                print(f"[PYTHON DEBUG _update_local] self.board_state updated. Player value in Python copy: {self.board_state.current_player}")
        else:
            self.board_state = None # Mark as invalid

    def get_valid_moves(self) -> list[ChessMove]:
        """ Gets a list of valid moves for the current board state using the EngineHandle API. """
        if not self._lib: raise RuntimeError("Library not loaded.")
        if not self._engine_handle: raise RuntimeError("Engine handle not initialized.")
        if self.board_state is None: raise RuntimeError("Board state not available.") # Check local copy

        # print(f"[PYTHON DEBUG] sizeof(ChessMove) = {sizeof(ChessMove)}")

        # --- First Call: Get the number of moves needed ---
        # print("[PYTHON DEBUG] Calling engine_get_valid_moves (1st time) to get size...")
        num_moves_needed = self._lib.engine_get_valid_moves(self._engine_handle, None, 0)

        # print(f"[PYTHON DEBUG] engine_get_valid_moves (1st time) reported {num_moves_needed} moves needed.")

        if num_moves_needed == 0:
            # print("[PYTHON DEBUG] No valid moves available.")
            return [] # No moves to get

        # --- Allocate Buffer in Python ---
        MoveArrayType = ChessMove * num_moves_needed
        moves_buffer = MoveArrayType()
        # print(f"[PYTHON DEBUG] Allocated buffer of size {sizeof(moves_buffer)} bytes for {num_moves_needed} moves.")

        # --- Second Call: Fill the buffer ---
        # print("[PYTHON DEBUG] Calling engine_get_valid_moves (2nd time) to fill buffer...")
        num_moves_filled = self._lib.engine_get_valid_moves(self._engine_handle, moves_buffer, num_moves_needed)

        if num_moves_filled != num_moves_needed:
            # This indicates an error or inconsistency in the C++ side
            print(f"[PYTHON WARNING/ERROR] Mismatch in move count: Needed={num_moves_needed}, Filled={num_moves_filled}. Using filled count.")
            # Adjust loop range if counts mismatch, though ideally they shouldn't
            count_to_use = min(num_moves_needed, num_moves_filled)
            if num_moves_filled > num_moves_needed: # Should not happen
                print("[PYTHON ERROR] C++ reported filling more moves than buffer size! Limiting read.")
                count_to_use = num_moves_needed
        else:
            count_to_use = num_moves_filled

        # --- Convert ctypes buffer to Python list ---
        moves = []
        if count_to_use > 0: # Only proceed if there are moves to convert
            try:
                # Iterate through the buffer we allocated and filled, up to the actual filled count
                moves = [moves_buffer[i] for i in range(count_to_use)]
                # print(f"[PYTHON DEBUG] Successfully converted buffer to list of {len(moves)} moves.")
            except Exception as e:
                print(f"Error converting Python buffer to list: {e}")
                print(traceback.format_exc())
                moves = []

        return moves

    # ===  apply_move method ===
    def apply_move(self, move: ChessMove) -> BoardState | None: # Return BoardState again
        if not self._lib: raise RuntimeError("Library not loaded.")
        if not self._engine_handle: raise RuntimeError("Engine handle not initialized.")

        result_board_state = BoardState() # Buffer to receive C++ result

        print(f"[PYTHON apply_move] Calling C++ engine_apply_move for move {self.move_to_str(move)}")
        success = self._lib.engine_apply_move(
            self._engine_handle,
            byref(move),
            byref(result_board_state) # Pass output buffer
        )
        print(f"[PYTHON apply_move] C++ engine_apply_move returned: {success}")

        if not success:
            print(f"[PYTHON WARNING] engine_apply_move returned false.")
            return None

        # --- SUCCESS: Read explicitly from buffer BEFORE assigning ---
        try:
            player_in_buffer = result_board_state.current_player # Read directly from buffer
            print(f"[PYTHON apply_move] Player read DIRECTLY from result buffer: {player_in_buffer}") # <<< ADD THIS LOG
        except Exception as e_read:
            print(f"[PYTHON apply_move] ERROR reading player directly from buffer: {e_read}")
            player_in_buffer = -99 # Indicate error

        # Now assign the whole struct
        self.board_state = result_board_state
        # Log the value AFTER assignment to self.board_state
        print(f"[PYTHON apply_move] Updated self.board_state from output param. Player NOW IN self.board_state: {self.board_state.current_player if self.board_state else 'None'}")

        # Check if the value changed during assignment
        if self.board_state and self.board_state.current_player != player_in_buffer:
             print("[PYTHON apply_move] *** MISMATCH between buffer read and self.board_state read! ***")


        return self.board_state

    def ai_move(self, difficulty: int = 2):
        """ Asks the C++ AI to calculate and apply its best move using the EngineHandle API. """
        if not self._lib: raise RuntimeError("Library not loaded.")
        if not self._engine_handle: raise RuntimeError("Engine handle not initialized.")

        # Pass handle, difficulty. Callback is null (0) for now.
        success = self._lib.engine_ai_move(self._engine_handle, c_int32(difficulty), 0)
        if not success:
            # Log error? C++ side should print errors.
            print(f"[PYTHON WARNING] engine_ai_move returned false.")
            # Consider if the local board state should be marked invalid or re-fetched

        # --- IMPORTANT: Update local board state copy after C++ modifies it ---
        self._update_local_board_state()


    def evaluate_board(self) -> float:
        """Gets the static evaluation of the current board state."""
        if not self._lib:
            raise RuntimeError("Library not loaded.")
        if not self._engine_handle:
            raise RuntimeError("Engine handle not initialized.")

        # Check if the evaluate function exists in the loaded library
        if hasattr(self._lib, 'engine_evaluate_board'):
            self._lib.engine_evaluate_board.argtypes = [c_void_p]
            self._lib.engine_evaluate_board.restype = c_double
            return self._lib.engine_evaluate_board(self._engine_handle)
        else:
            raise AttributeError("engine_evaluate_board function not found in C API.")

    def cancel_search(self):
        """ Signals the C++ engine to stop the current search using the EngineHandle API. """
        if not self._lib: raise RuntimeError("Library not loaded.")
        if not self._engine_handle: raise RuntimeError("Engine handle not initialized.")
        print("Requesting search cancellation...")
        self._lib.engine_cancel_search(self._engine_handle)


    # --- Context Manager Methods ---
    def __enter__(self):
        """Allows using the engine with a 'with' statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensures resources are cleaned up when exiting a 'with' block."""
        self.close()

    def close(self):
        """ Frees the C++ EngineHandle. """
        print("Closing ChessEngine...")
        if hasattr(self, '_lib') and self._lib and hasattr(self, '_engine_handle') and self._engine_handle:
            if hasattr(self._lib, 'engine_destroy'):
                print(f"Destroying engine handle (pointer: {self._engine_handle})...")
                try:
                    self._lib.engine_destroy(self._engine_handle)
                except Exception as e:
                    print(f"  Warning: Error calling engine_destroy: {e}")
            else:
                 print("  Warning: engine_destroy function not found in library.")
            self._engine_handle = None # Mark as destroyed

        print("ChessEngine closed.")

    # --- Helper Methods for Display/Debugging ---
    def _square_to_str(self, pos: BoardPosition) -> str:
        """Converts a BoardPosition to algebraic notation (e.g., 'e4')."""
        if not (0 <= pos.rank < 8 and 0 <= pos.file < 8):
             return "??"
        return chr(ord('a') + pos.file) + str(pos.rank + 1)

    def move_to_str(self, move: ChessMove) -> str:
        """Converts a ChessMove to a string representation using the C API."""
        if not self._lib: return "ERR! (No Lib)"
        if not self._engine_handle: return "ERR! (No Handle)"
        if not hasattr(self._lib, 'engine_move_to_str'): return "ERR! (No Func)"

        buffer_size = 10 # Max SAN length is usually less (e.g., O-O-O# or e8=Q#)
        move_str_buffer = create_string_buffer(buffer_size)

        try:
            # Pass move by reference (pointer)
            success = self._lib.engine_move_to_str(self._engine_handle, byref(move), move_str_buffer, buffer_size)
            if success:
                return move_str_buffer.value.decode('utf-8')
            else:
                # C function failed or buffer was too small
                partial_str = move_str_buffer.value.decode('utf-8', errors='ignore')
                print(f"Warning: engine_move_to_str failed or buffer too small. Partial: '{partial_str}'")
                # Fallback to basic coords if SAN fails
                start_sq = self._square_to_str(move.start_position)
                target_sq = self._square_to_str(move.target_position)
                return f"{start_sq}{target_sq}?" # Indicate failed SAN conversion
        except Exception as e:
            print(f"Error calling engine_move_to_str: {e}")
            return "ERR!"


    def print_board(self):
        """Prints a simple text representation of the current board state."""
        if self.board_state is None: # Check the local copy
            print("Board state not initialized.")
            return

        state = self.board_state
        piece_map = {
            # White pieces
            (PieceType.PAWN, Player.WHITE): "P", (PieceType.KNIGHT, Player.WHITE): "N",
            (PieceType.BISHOP, Player.WHITE): "B", (PieceType.ROOK, Player.WHITE): "R",
            (PieceType.QUEEN, Player.WHITE): "Q", (PieceType.KING, Player.WHITE): "K",
            # Black pieces
            (PieceType.PAWN, Player.BLACK): "p", (PieceType.KNIGHT, Player.BLACK): "n",
            (PieceType.BISHOP, Player.BLACK): "b", (PieceType.ROOK, Player.BLACK): "r",
            (PieceType.QUEEN, Player.BLACK): "q", (PieceType.KING, Player.BLACK): "k",
        }
        print("\n  a b c d e f g h")
        print(" +-----------------+")
        for r in range(7, -1, -1): # Print from rank 8 down to 1
            print(f"{r+1}|", end="")
            for c in range(8): # Files a to h
                piece = state.pieces[r][c]
                char = piece_map.get((piece.type, piece.piece_player), ".") # Get char or '.' if empty
                print(f" {char}", end="")
            print(f" |{r+1}")
        print(" +-----------------+")
        print("  a b c d e f g h")

        # Print game state info
        player_str = "White" if state.current_player == Player.WHITE else "Black"
        status_map = {
            GameStatus.NORMAL: "Normal", GameStatus.DRAW: "Draw",
            GameStatus.CHECKMATE: "Checkmate", GameStatus.RESIGNED: "Resigned",
            GameStatus.DRAW_BY_REPETITION: "Draw by Repetition/50-Move/etc." # Combine draw types for simplicity
        }
        status_str = status_map.get(state.status, f"Unknown ({state.status})")
        print(f"Turn: {player_str}")
        print(f"Status: {status_str}")
        print(f"White in Check: {bool(state.in_check[Player.WHITE])}")
        print(f"Black in Check: {bool(state.in_check[Player.BLACK])}")
        print(f"Can Claim Draw: {bool(state.can_claim_draw)}")
        print(f"50-Move Counter: {state.turns_since_last_capture_or_pawn // 2}") # Show full moves

        # Optional: Print Castling Rights
        wc = state.can_castle
        castle_str = f"Castling: W:(K{'✓' if wc[0] else '✗'} Q{'✓' if wc[1] else '✗'}) B:(K{'✓' if wc[2] else '✗'} Q{'✓' if wc[3] else '✗'})"
        print(castle_str)

        # Optional: Print En Passant Targets
        ep_targets = []
        for i in range(16):
            if state.en_passant_valid[i]:
                rank = 2 if i < 8 else 5 # Rank where the capturing pawn *lands*
                file = i % 8
                # Need to construct the target square string (e.g., e3 or e6)
                ep_sq_str = chr(ord('a') + file) + str(rank + 1)
                ep_targets.append(ep_sq_str)
        if ep_targets:
            print(f"En Passant Target(s): {', '.join(ep_targets)}")


# --- Helper functions (outside class) --- (Keep target_for_move as it was)
def target_for_move(move: ChessMove) -> tuple[int, int] | None:
    """
    Calculates the target (rank, file) tuple for a move, handling castling.
    Useful for highlighting squares in a UI.
    """
    if move.type == MoveType.CASTLE:
        # Determine king's target square based on side
        king_start_file = 4 # King always starts on file 4 (e)
        king_target_file = 6 if move.target_position.file > king_start_file else 2 # Target file g or c
        return (move.start_position.rank, king_target_file) # King's final square
    elif move.type == MoveType.CLAIM_DRAW or move.type == MoveType.RESIGN:
        # These moves don't have a target square on the board
        raise ValueError("Cannot get target square for Resign or Claim Draw moves.")
    else:
        # Normal moves use the move's target position
        return (move.target_position.rank, move.target_position.file)


# Example usage (if run directly)
if __name__ == "__main__":
    # --- Configuration ---
    # Adjust these paths as needed
    DEFAULT_LIB_PATH = "../build/src/libchess_cpp.so" # Example for Linux build
    DEFAULT_MODEL_PATH = "../models/model-small.onnx" # Example model path

    lib_path = os.getenv("CHESS_LIB_PATH", DEFAULT_LIB_PATH)
    model_path = os.getenv("CHESS_MODEL_PATH", DEFAULT_MODEL_PATH)

    print("--- Chess Engine Python Wrapper Test ---")
    print(f"Using Library: {os.path.abspath(lib_path)}")
    print(f"Using Model: {os.path.abspath(model_path)}")

    try:
        with ChessEngine(lib_path, model_path) as engine:
            engine.print_board()
            # print(f"Initial Evaluation: {engine.evaluate_board():.2f}") # Evaluate might not exist

            # Get and print valid moves for the initial position
            valid_moves = engine.get_valid_moves()
            print(f"\nValid moves ({len(valid_moves)}):")
            move_strs = [engine.move_to_str(m) for m in valid_moves]
            print(", ".join(move_strs[:20]) + ("..." if len(move_strs) > 20 else "")) # Print first 20

            # Example: Make the first valid move (e.g., e2e4)
            if valid_moves:
                # Find e2e4 using the new move_to_str
                move_to_make = next((m for m in valid_moves if engine.move_to_str(m).startswith('e2e4')), None)
                if move_to_make is None and valid_moves: # If e2e4 not found, take the first valid one
                     move_to_make = valid_moves[0]

                if move_to_make:
                    print(f"\nApplying move: {engine.move_to_str(move_to_make)}")
                    engine.apply_move(move_to_make)
                    engine.print_board()
                    # print(f"Evaluation after move: {engine.evaluate_board():.2f}")

                    # Example: Let AI make a move
                    print("\nAI is thinking...")
                    engine.ai_move(difficulty=2) # SET DIFFICULTY HERE
                    print("AI has moved.")
                    engine.print_board()
                    # print(f"Evaluation after AI move: {engine.evaluate_board():.2f}")
                else:
                    print("\nCould not find a valid move to apply.")


            else:
                print("\nNo valid moves from initial position? (Error)")

    except FileNotFoundError as e:
        print(f"\nError: Required file not found.")
        print(e)
        print("Please ensure the library and model paths are correct.")
        print("You might need to build the C++ project first (e.g., in ../build).")
    except (RuntimeError, OSError, AttributeError) as e:
        print(f"\nAn error occurred: {e}")
        print("Check C++ build, library paths, and function signatures.")
        print(traceback.format_exc()) # Print traceback for these errors too
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print(traceback.format_exc()) # Print traceback

    print("\n--- Test Finished ---")

