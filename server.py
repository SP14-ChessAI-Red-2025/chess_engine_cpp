# server.py (Synchronous AI Mode - Using ctypes for state/move conversion)
import os
import sys
import traceback
import threading # Keep for Lock
import time
from flask import Flask, jsonify, request
from flask_cors import CORS
from ctypes import (
    Structure, POINTER, c_void_p, c_size_t, c_int8, c_uint8, c_int32,
    c_bool, cast, sizeof, pointer, addressof
)

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
        ("_padding_", c_uint8 * 3)]
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

            # --- Get Move String (Requires engine access, potential lock needed if done outside) ---
            # Calculate address of the specific move
            current_move_addr = move_buffer_address + i * sizeof(CtypesChessMove)
            san = engine.move_to_str(current_move_addr) if engine else "ERR!"
            # --- End Get Move String ---

            # Convert ctypes data to dictionary
            move_dict = {
                "type": current_move.type,
                "start": {"rank": current_move.start_position.rank, "file": current_move.start_position.file},
                "target": {"rank": current_move.target_position.rank, "file": current_move.target_position.file},
                "promotion": current_move.promotion_target,
                "san": san # Use the generated SAN string
            }
            move_list.append(move_dict)
        # app.logger.info(f"Successfully converted {len(move_list)} moves from ctypes buffer.")

    except Exception as e:
        app.logger.error(f"Error processing ctypes move buffer: {e}", exc_info=True)
        return [] # Return empty list on error

    return move_list

def state_address_to_dict(address):
    """Converts a C board_state address to a Python dictionary."""
    if address == 0:
        app.logger.error("state_address_to_dict received NULL address.")
        return None
    try:
        state_ptr = cast(address, POINTER(CtypesBoardState))
        if not state_ptr:
            app.logger.error("state_address_to_dict: ctypes.cast resulted in NULL pointer.")
            return None
        c_state = state_ptr.contents
        py_state = {}
        py_pieces = []
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
        return py_state
    except Exception as e:
        app.logger.error(f"Error converting C state address {address} to dict: {e}", exc_info=True)
        return None

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
             address = engine.board_state_address # Get address as integer

        if address == 0:
             app.logger.error("/api/state: engine.board_state_address returned 0 (NULL ptr)")
             return jsonify({"error": "Engine state pointer is unavailable"}), 500
        # app.logger.info(f"/api/state: Received address: {address}") # Log address

        py_state = state_address_to_dict(address)

        if py_state is None:
             return jsonify({"error": "Failed to read or convert engine state"}), 500

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
            # Call the new Cython method to get buffer address and count
            address, count = engine.get_valid_moves_address_count()
            # Use the helper function to convert the buffer data
            # Engine lock is needed because move_to_str is called inside helper
            valid_moves_list = moves_to_list_ctypes(address, count)

        # app.logger.info(f"/api/moves: Got address={address}, count={count}, list_len={len(valid_moves_list)}")

        return jsonify(valid_moves_list)

    except Exception as e:
        app.logger.error(f"Error in /api/moves: {e}", exc_info=True)
        return jsonify({"error": "Failed to get valid moves"}), 500


@app.route('/api/apply_move', methods=['POST'])
def apply_move_api():
    """
    Applies a player move received from the frontend.
    Finds the matching C move in the engine's buffer and passes its address
    to the Cython apply_move function. Returns the new board state.
    """
    if not engine:
        app.logger.error("/api/apply_move: Engine not initialized")
        return jsonify({"error": "Chess engine not initialized"}), 500

    # --- Get Move Data from JSON ---
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.json
    if not isinstance(data, dict):
         return jsonify({"error": "Invalid JSON format, expected an object"}), 400

    try:
        start_data = data['start'] # e.g., {'rank': 1, 'file': 4}
        target_data = data['target'] # e.g., {'rank': 3, 'file': 4}
        # Handle promotion - it might be missing or None for non-promotion moves
        promotion_data = data.get('promotion', PieceType.NONE)
        if promotion_data is None: # Treat None as no promotion
            promotion_data = PieceType.NONE

        start_rank = int(start_data['rank'])
        start_file = int(start_data['file'])
        target_rank = int(target_data['rank'])
        target_file = int(target_data['file'])
        promotion_piece = int(promotion_data)

        # Basic validation
        if not (0 <= start_rank < 8 and 0 <= start_file < 8 and
                0 <= target_rank < 8 and 0 <= target_file < 8 and
                0 <= promotion_piece <= 6):
            raise ValueError("Invalid rank/file/promotion data")

    except (KeyError, TypeError, ValueError) as e:
        app.logger.error(f"Invalid move data received in /api/apply_move: {e}. Data: {data}")
        return jsonify({"error": f"Invalid move data format: {e}"}), 400
    # --- End Get Move Data ---

    app.logger.info(f"Received move: {start_rank},{start_file} -> {target_rank},{target_file} (promo={promotion_piece})")

    try:
        move_address_to_apply = 0
        with engine_lock:
            # 1. Get the address and count of the current valid C moves
            moves_buffer_address, num_moves = engine.get_valid_moves_address_count()
            if moves_buffer_address == 0 or num_moves == 0:
                 app.logger.warning("No valid moves found in engine buffer for comparison.")
                 return jsonify({"error": "Move cannot be applied: No valid moves available"}), 400

            # 2. Iterate through the C buffer to find the matching move
            found_move = False
            move_ptr = cast(moves_buffer_address, POINTER(CtypesChessMove))
            for i in range(num_moves):
                c_move = move_ptr[i]
                # Compare start, target, and promotion
                if (c_move.start_position.rank == start_rank and
                    c_move.start_position.file == start_file and
                    c_move.target_position.rank == target_rank and
                    c_move.target_position.file == target_file and
                    c_move.promotion_target == promotion_piece):

                    # Calculate the address of this specific move struct
                    # Note: addressof(c_move) might get address of stack copy, calculate manually
                    move_address_to_apply = moves_buffer_address + i * sizeof(CtypesChessMove)
                    found_move = True
                    app.logger.info(f"Found matching C move at index {i}, address {move_address_to_apply}")
                    break # Stop searching once found

            if not found_move:
                app.logger.warning(f"Move {start_rank},{start_file}->{target_rank},{target_file}({promotion_piece}) not found in valid moves buffer.")
                return jsonify({"error": "Invalid move: Not found in current valid moves"}), 400

            # 3. Call the Cython engine's apply_move with the C move's address
            app.logger.info(f"Calling engine.apply_move with address: {move_address_to_apply}")
            apply_success = engine.apply_move(move_address_to_apply)

            if not apply_success:
                 app.logger.error(f"Cython engine.apply_move({move_address_to_apply}) returned false.")
                 return jsonify({"error": "Engine failed to apply the move"}), 500

            app.logger.info(f"Engine successfully applied move at address: {move_address_to_apply}")

            # 4. Fetch the new board state address
            new_state_address = engine.board_state_address
            if new_state_address == 0:
                app.logger.error("Failed to get new board state address after applying move!")
                return jsonify({"error": "Failed to retrieve board state after move"}), 500

            # 5. Convert the new C state to a Python dictionary
            new_state_dict = state_address_to_dict(new_state_address)
            if new_state_dict is None:
                 return jsonify({"error": "Failed to convert new board state"}), 500

        # 6. Return the new state as JSON
        app.logger.info(f"Returning new state. Player: {new_state_dict.get('current_player')}, Status: {new_state_dict.get('status')}")
        return jsonify(new_state_dict), 200

    except ValueError as e: # Catch specific value errors from Cython/C
         app.logger.error(f"Error applying move in engine: {e}", exc_info=True)
         return jsonify({"error": f"Engine error applying move: {e}"}), 400 # 400 might be better if move was invalid
    except RuntimeError as e: # Catch handle errors etc.
         app.logger.error(f"Runtime error during /api/apply_move: {e}", exc_info=True)
         return jsonify({"error": f"Server runtime error: {e}"}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error in /api/apply_move: {e}", exc_info=True)
        return jsonify({"error": "Unexpected server error during move application"}), 500


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

            # Call the Cython ai_move method, which modifies internal state
            ai_move_success = engine.ai_move(difficulty=difficulty)
            if not ai_move_success:
                 app.logger.error("/api/ai_move: engine.ai_move returned false/failure.")
                 # Fall through to get the (potentially unchanged) state

            end_time = time.time()
            app.logger.info(f"AI move calculation attempted in {end_time - start_time:.2f} seconds.")

            # Fetch the new state address *after* ai_move completed
            address = engine.board_state_address
            new_state_dict = state_address_to_dict(address)
            if new_state_dict is None:
                return jsonify({"error": "Failed to retrieve or convert board state after AI move"}), 500

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

            # Fetch and return the initial state after reset
            address = engine.board_state_address
            state_dict = state_address_to_dict(address)
            if state_dict is None:
                 return jsonify({"error": "Engine reset but failed to get initial state"}), 500

            return jsonify({"message": "Game reset successfully", "initial_state": state_dict}), 200
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