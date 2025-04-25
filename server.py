# server.py (Synchronous AI Mode - Using ctypes for state/move conversion)
import os
import sys
import traceback
import threading # Keep for Lock
import time
from flask import Flask, jsonify, request
from flask_cors import CORS
# --- ADD ctypes ---
from ctypes import (
    Structure, POINTER, c_void_p, c_size_t, c_int8, c_uint8, c_int32,
    c_bool, cast, sizeof, pointer # Added 'pointer'
)
# --- END ctypes ---

# --- Path Setup ---
project_root = os.path.dirname(os.path.abspath(__file__))
python_dir_path = os.path.join(project_root, 'python')
if python_dir_path not in sys.path:
    sys.path.append(python_dir_path)
src_dir_path = os.path.join(project_root, 'src')
if src_dir_path not in sys.path:
     sys.path.append(src_dir_path)

try:
    # Import Engine class and Enums
    from chess_dir.ai_chess import ChessEngine, Player, PieceType, GameStatus, MoveType
    # We define structures via ctypes now, so don't import Cython wrappers for them
except ImportError as e:
    print(f"Error importing ChessEngine/Enums: {e}")
    print(f"Attempted import from: {os.path.join(python_dir_path, 'chess_dir')}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

# --- Configuration --- (Ensure these paths are correct relative to server.py)
LIBRARY_PATH = "build/src/libchess_cpp.so"
MODEL_PATH = "model/trained_nnue.onnx" # Make sure this is your model file

# --- ADD ctypes Structures (VERIFY THESE MATCH C++ EXACTLY) ---
class CtypesBoardPosition(Structure):
    _pack_ = 1
    _fields_ = [("rank", c_uint8), ("file", c_uint8)]

class CtypesPiece(Structure):
    _pack_ = 1
    _fields_ = [("type", c_int8), ("piece_player", c_int8)]

# ADD CtypesChessMove definition
class CtypesChessMove(Structure):
    _pack_ = 1
    _fields_ = [("type", c_int8),             # Corresponds to move_type enum
                ("start_position", CtypesBoardPosition),
                ("target_position", CtypesBoardPosition),
                ("promotion_target", c_int8)] # Corresponds to piece_type enum

class CtypesBoardState(Structure):
    _pack_ = 1
    _fields_ = [
        ("pieces", (CtypesPiece * 8) * 8),            # 64 * 2 = 128 bytes
        ("can_castle", c_bool * 4),                   # 4 * 1 = 4 bytes
        ("in_check", c_bool * 2),                     # 2 * 1 = 2 bytes
        ("en_passant_valid", c_bool * 16),            # 16 * 1 = 16 bytes
        ("turns_since_last_capture_or_pawn", c_int32),# 1 * 4 = 4 bytes
        ("current_player", c_int8),                   # 1 * 1 = 1 byte
        ("status", c_int8),                           # 1 * 1 = 1 byte
        ("can_claim_draw", c_bool),                   # 1 * 1 = 1 byte
        # Verify if padding exists/is needed based on C++ struct definition and alignment
    ]
# --- END ctypes Structures ---


# --- Flask App Setup ---
app = Flask(__name__)
# It's generally better to get allowed origins from environment variables in production
# Defaulting to your github.io page for now.
frontend_url = os.environ.get("FRONTEND_URL", "https://sp14-chessai-red-2025.github.io")
cors_origins = [frontend_url]
# Allow credentials if needed in the future, but usually not for simple GET/POST
CORS(app, resources={r"/api/*": {"origins": cors_origins}})
app.logger.info(f"CORS enabled for origin: {frontend_url}")


# --- Global Chess Engine Instance & Lock ---
engine_lock = threading.Lock() # Keep lock for sequential access


# --- Engine Initialization ---
engine = None # Initialize as None
try:
    # Check paths carefully
    abs_lib_path = os.path.abspath(os.path.join(project_root, LIBRARY_PATH))
    abs_model_path = os.path.abspath(os.path.join(project_root, MODEL_PATH))
    app.logger.info(f"Checking Library Path: {abs_lib_path}")
    app.logger.info(f"Checking Model Path: {abs_model_path}")

    if not os.path.exists(abs_lib_path): raise FileNotFoundError(f"Library not found: {abs_lib_path}")
    if not os.path.exists(abs_model_path): raise FileNotFoundError(f"Model not found: {abs_model_path}")

    app.logger.info(f"Loading Chess Engine with Lib: {abs_lib_path}, Model: {abs_model_path}")
    # Use absolute paths for engine creation
    engine = ChessEngine(library_path=abs_lib_path, model_path=abs_model_path)
    app.logger.info("Chess Engine loaded successfully.")
except Exception as init_error:
    app.logger.error(f"FATAL: Failed to initialize ChessEngine: {init_error}", exc_info=True)
    # Keep engine as None

# --- Helper Functions ---
def moves_to_list_ctypes(move_buffer_address, num_moves):
    """Converts a C buffer of ChessMove structs (via address) to a list of Python dicts."""
    if not engine or move_buffer_address == 0 or num_moves <= 0:
        return []

    move_list = []
    try:
        # Cast the address to a pointer to the first ChessMove in the buffer
        move_ptr = cast(move_buffer_address, POINTER(CtypesChessMove))

        for i in range(num_moves):
            # Access the current move struct using pointer indexing
            current_move = move_ptr[i]

            # Convert ctypes data to dictionary
            # TODO: Implement SAN generation if needed. Requires engine access.
            # Might need a separate Cython call: engine.get_san_for_move_data(...)
            move_dict = {
                "type": current_move.type,
                "start": {"rank": current_move.start_position.rank, "file": current_move.start_position.file},
                "target": {"rank": current_move.target_position.rank, "file": current_move.target_position.file},
                "promotion": current_move.promotion_target,
                "san": "N/A" # Placeholder for SAN string
            }
            move_list.append(move_dict)
        app.logger.info(f"Successfully converted {len(move_list)} moves from ctypes buffer.")

    except Exception as e:
        app.logger.error(f"Error processing ctypes move buffer: {e}", exc_info=True)
        return [] # Return empty list on error

    return move_list

# --- API Endpoints ---

@app.route('/api/state', methods=['GET'])
def get_state():
    """Returns the current board state by reading from C pointer via ctypes."""
    if not engine:
        app.logger.error("/api/state: Engine not initialized")
        return jsonify({"error": "Chess engine not initialized on server"}), 500

    try:
        address = 0
        with engine_lock:
             # Call the Cython property to get the address
             address = engine.board_state_address # Get address as integer

        if address == 0:
             app.logger.error("/api/state: engine.board_state_address returned 0 (NULL ptr)")
             return jsonify({"error": "Engine state pointer is unavailable"}), 500
        app.logger.info(f"/api/state: Received address: {address}") # Log address

        # --- Convert address to ctypes pointer ---
        try:
            state_ptr = cast(address, POINTER(CtypesBoardState))
            if not state_ptr:
                 app.logger.error("/api/state: ctypes.cast resulted in a NULL pointer")
                 return jsonify({"error": "Failed to cast engine state pointer"}), 500
            # Access the data via pointer.contents
            c_state = state_ptr.contents
            # app.logger.info("/api/state: Successfully accessed pointer.contents") # Optional log
        except Exception as cast_error:
             app.logger.error(f"Error casting or accessing ctypes pointer in /api/state: {cast_error}", exc_info=True)
             return jsonify({"error": "Failed to interpret engine state pointer"}), 500
        # --- END Conversion ---

        # --- Build dictionary in Python using ctypes data ---
        py_state = {}
        py_pieces = []
        try:
            # app.logger.info("/api/state: Starting dict conversion...") # Optional log
            for r in range(8):
                row_list = []
                for f in range(8):
                    c_piece = c_state.pieces[r][f]
                    row_list.append({'type': c_piece.type, 'player': c_piece.piece_player})
                py_pieces.append(row_list)

            py_state['pieces'] = py_pieces
            py_state['current_player'] = c_state.current_player
            py_state['can_castle'] = list(c_state.can_castle)
            py_state['in_check'] = list(c_state.in_check)
            py_state['en_passant_valid'] = list(c_state.en_passant_valid)
            py_state['turns_since_last_capture_or_pawn'] = c_state.turns_since_last_capture_or_pawn
            py_state['status'] = c_state.status
            py_state['can_claim_draw'] = c_state.can_claim_draw
            # app.logger.info("/api/state: Finished dict conversion.") # Optional log

        except Exception as conversion_error:
            app.logger.error(f"Error converting ctypes state to dict in /api/state: {conversion_error}", exc_info=True)
            return jsonify({"error": "Failed to convert engine state data"}), 500
        # --- END Building Dictionary ---

        return jsonify(py_state)

    except Exception as e:
        app.logger.error(f"Error DURING /api/state processing: {type(e).__name__}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to get board state due to server error: {type(e).__name__}"}), 500


@app.route('/api/moves', methods=['GET'])
def get_valid_moves_api():
    """Returns a list of valid moves for the current player using ctypes."""
    if not engine:
        app.logger.error("/api/moves: Engine not initialized")
        return jsonify({"error": "Chess engine not initialized"}), 500

    try:
        address = 0
        count = 0
        with engine_lock:
            # Call the new Cython method
            address, count = engine.get_valid_moves_address_count()

        app.logger.info(f"/api/moves: Got address={address}, count={count}")

        # Use the new helper function to convert the buffer data
        valid_moves_list = moves_to_list_ctypes(address, count)

        return jsonify(valid_moves_list)

    except Exception as e:
        app.logger.error(f"Error in /api/moves: {e}", exc_info=True)
        return jsonify({"error": "Failed to get valid moves"}), 500


@app.route('/api/apply_move', methods=['POST'])
def apply_move_api():
    """Applies a player move and returns the new state."""
    # --- Placeholder - Needs Implementation ---
    # This endpoint needs to be refactored to work with the ctypes approach.
    # See previous discussion for options on how to handle this:
    # 1. New Cython function taking raw move data.
    # 2. Construct ctypes.ChessMove here and pass its address to Cython.
    # 3. Find the move address in the buffer from get_valid_moves_address_count and pass that.
    if not engine:
        app.logger.error("/api/apply_move: Engine not initialized")
        return jsonify({"error": "Chess engine not initialized"}), 500

    app.logger.error("FATAL: /api/apply_move endpoint is not yet fully implemented for the ctypes approach!")
    return jsonify({"error": "Apply move functionality requires update"}), 501 # 501 Not Implemented
    # --- End Placeholder ---


@app.route('/api/ai_move', methods=['POST'])
def trigger_ai_move():
    """Triggers the AI to make a move SYNCHRONOUSLY and returns the new state."""
    if not engine:
        app.logger.error("/api/ai_move: Engine not initialized")
        return jsonify({"error": "Chess engine not initialized"}), 500

    try:
        new_state_dict = None # Define outside the lock
        with engine_lock:
            difficulty = 5 # Default difficulty
            if request.is_json and isinstance(request.json, dict):
                 difficulty = request.json.get('difficulty', 5)
            elif request.data: # Handle cases where JSON might not be parsed correctly but data exists
                 app.logger.warning("/api/ai_move called without parseable JSON body, using default difficulty.")
            else:
                 app.logger.warning("/api/ai_move called without body, using default difficulty.")


            app.logger.info(f"Calculating AI move (difficulty={difficulty}, blocking)...")
            start_time = time.time()

            # Assuming engine.ai_move updates the internal state correctly
            # and doesn't rely on Python/Cython objects for state transfer now.
            # It might need modification if it previously returned a state object.
            ai_move_success = engine.ai_move(difficulty=difficulty) # Check if ai_move returns success?
            if not ai_move_success:
                 app.logger.error("/api/ai_move: engine.ai_move returned false/failure.")
                 # Even if it failed, try getting the state, might be unchanged or error state
                 # Fall through to get the current state address

            end_time = time.time()
            app.logger.info(f"AI move calculation attempted in {end_time - start_time:.2f} seconds.")

            # Fetch the state *after* ai_move completed using the address method
            address = engine.board_state_address
            if address == 0:
                 app.logger.error("/api/ai_move: engine.board_state_address returned 0 after AI move!")
                 return jsonify({"error": "AI move ran but engine state pointer is unavailable"}), 500

            # Convert the address to state dict using ctypes (similar to /api/state)
            state_ptr = cast(address, POINTER(CtypesBoardState))
            if not state_ptr:
                 app.logger.error("/api/ai_move: ctypes.cast resulted in a NULL pointer after AI move")
                 return jsonify({"error": "Failed to cast engine state pointer after AI move"}), 500
            c_state = state_ptr.contents

            # Build dictionary (could refactor this into a helper)
            new_state_dict = {}
            py_pieces = []
            for r in range(8):
                row_list = []
                for f in range(8):
                    c_piece = c_state.pieces[r][f]
                    row_list.append({'type': c_piece.type, 'player': c_piece.piece_player})
                py_pieces.append(row_list)
            new_state_dict['pieces'] = py_pieces
            new_state_dict['current_player'] = c_state.current_player
            new_state_dict['can_castle'] = list(c_state.can_castle)
            new_state_dict['in_check'] = list(c_state.in_check)
            new_state_dict['en_passant_valid'] = list(c_state.en_passant_valid)
            new_state_dict['turns_since_last_capture_or_pawn'] = c_state.turns_since_last_capture_or_pawn
            new_state_dict['status'] = c_state.status
            new_state_dict['can_claim_draw'] = c_state.can_claim_draw

        # Return the newly fetched and converted state dictionary
        app.logger.info(f"Returning state after AI move. New current player: {new_state_dict.get('current_player', 'N/A')}")
        return jsonify(new_state_dict), 200

    except Exception as e:
        app.logger.error(f"Error in /api/ai_move (Sync): {e}", exc_info=True)
        return jsonify({"error": "Failed to calculate AI move due to server error"}), 500


@app.route('/api/reset', methods=['POST'])
def reset_game():
    """Destroys and re-initializes the chess engine."""
    global engine
    with engine_lock:
        app.logger.info("Received request to reset engine...")
        # Re-initialization logic (ensure paths are correct)
        try:
            # Creating a new instance should handle cleanup of the old one via __dealloc__
            abs_lib_path = os.path.abspath(os.path.join(project_root, LIBRARY_PATH))
            abs_model_path = os.path.abspath(os.path.join(project_root, MODEL_PATH))

            if not os.path.exists(abs_lib_path): raise FileNotFoundError(f"Library not found for reset: {abs_lib_path}")
            if not os.path.exists(abs_model_path): raise FileNotFoundError(f"Model not found for reset: {abs_model_path}")

            app.logger.info("Attempting to re-initialize engine...")
            engine = ChessEngine(library_path=abs_lib_path, model_path=abs_model_path)
            app.logger.info("Engine re-initialized successfully.")
            # Optionally fetch and return the initial state
            # address = engine.board_state_address
            # state_dict = ... convert address ...
            # return jsonify({"message": "Game reset successfully", "initial_state": state_dict}), 200
            return jsonify({"message": "Game reset successfully"}), 200
        except Exception as e:
            app.logger.error(f"Error during engine reset: {e}", exc_info=True)
            engine = None # Ensure engine is None if reset fails
            return jsonify({"error": f"Failed to reset engine: {e}"}), 500


# --- Run the Server ---
if __name__ == '__main__':
    if not engine:
        print("FATAL: Cannot start server: Chess engine failed to initialize during startup.", file=sys.stderr)
        sys.exit(1) # Exit if engine isn't ready
    else:
        port = int(os.environ.get('PORT', 5000)) # Use PORT env var if available (Render sets this)
        print(f"Starting Flask server (Synchronous AI Mode) on port {port}...")
        # Use Gunicorn in production via Procfile, but app.run for local testing
        # For Render, Gunicorn command in Procfile/Start Command is typical
        # Ensure threaded=False is suitable for your C++ engine's threading model
        app.run(debug=False, host='0.0.0.0', port=port, threaded=False)