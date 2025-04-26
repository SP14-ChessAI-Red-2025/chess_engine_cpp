# server.py (Corrected AI Move Handling)
import os
import sys
import traceback
import threading
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
if python_dir_path not in sys.path: sys.path.append(python_dir_path)
src_dir_path = os.path.join(project_root, 'src')
if src_dir_path not in sys.path: sys.path.append(src_dir_path)

try:
    # Make sure ai_chess.py has the updated methods returning BoardState object
    from chess_dir.ai_chess import ChessEngine, Player, PieceType, GameStatus, MoveType
    # Import the ctypes structure definitions from ai_chess or define them here
    from chess_dir.ai_chess import CtypesBoardState, CtypesChessMove, CtypesPiece, CtypesBoardPosition # Assuming they are exposed or defined
except ImportError as e:
    print(f"Error importing ChessEngine/Enums/Structures: {e}"); sys.exit(1)
except AttributeError as e:
     print(f"Error importing specific ctypes structures (check ai_chess.py): {e}"); sys.exit(1)


# --- Configuration ---
LIBRARY_PATH = "build/src/libchess_cpp.so" # Path is less critical if linked via setup.py
MODEL_PATH = "model/trained_nnue.onnx"

# --- Flask App Setup ---
app = Flask(__name__)
# Ensure this matches your frontend URL (e.g., from GitHub Pages or Render frontend service)
frontend_url = os.environ.get("FRONTEND_URL", "https://sp14-chessai-red-2025.github.io") # Or your specific frontend URL
CORS(app, resources={r"/api/*": {"origins": [frontend_url]}})
app.logger.info(f"CORS enabled for origin: {frontend_url}")

# --- Global Engine Instance & Lock ---
engine_lock = threading.Lock()
engine = None

# --- Engine Initialization ---
try:
    abs_lib_path = os.path.abspath(os.path.join(project_root, LIBRARY_PATH)) # Informational
    abs_model_path = os.path.abspath(os.path.join(project_root, MODEL_PATH))
    # Validate model path existence as it's crucial
    if not os.path.exists(abs_model_path):
        raise FileNotFoundError(f"NNUE Model not found: {abs_model_path}")

    # Pass model path to ChessEngine constructor
    # Library path is handled by Cython linking if setup.py is used correctly
    engine = ChessEngine(library_path=abs_lib_path, model_path=abs_model_path)
    app.logger.info("Chess Engine loaded successfully.")
except FileNotFoundError as fnf_error:
     app.logger.error(f"FATAL: Required file not found: {fnf_error}", exc_info=True)
     sys.exit(1) # Exit if model is missing
except Exception as init_error:
    app.logger.error(f"FATAL: Failed to initialize ChessEngine: {init_error}", exc_info=True)
    sys.exit(1) # Exit if engine init fails


# --- Helper Functions ---
def moves_to_list_ctypes(move_buffer_address, num_moves):
    """Converts C move buffer address to a list of Python move dicts."""
    if not engine or move_buffer_address == 0 or num_moves <= 0: return []
    move_list = []
    try:
        # Cast the integer address to a pointer to the CtypesChessMove structure
        move_ptr = cast(move_buffer_address, POINTER(CtypesChessMove))
        for i in range(num_moves):
            current_move = move_ptr[i]
            # Calculate address of the current move in the buffer for move_to_str
            current_move_addr = move_buffer_address + i * sizeof(CtypesChessMove)
            san = "N/A"
            try:
                # Pass the integer address of the move to the Cython wrapper
                san = engine.move_to_str(current_move_addr)
            except Exception as san_err:
                 app.logger.warning(f"Error getting SAN for move index {i}: {san_err}")

            move_dict = {
                "type": current_move.type,
                "start": {"rank": current_move.start_position.rank, "file": current_move.start_position.file},
                "target": {"rank": current_move.target_position.rank, "file": current_move.target_position.file},
                "promotion": current_move.promotion_target,
                "san": san
            }
            move_list.append(move_dict)
    except Exception as e:
        app.logger.error(f"Error processing ctypes move buffer: {e}", exc_info=True)
        return [] # Return empty list on error
    return move_list

def state_address_to_dict(address):
    """Converts a C board_state pointer address (integer) to a Python dictionary."""
    if address == 0:
        app.logger.error("state_address_to_dict received NULL address.")
        return None
    try:
        # Cast the integer address to a pointer to the CtypesBoardState structure
        state_ptr = cast(address, POINTER(CtypesBoardState))
        if not state_ptr:
             app.logger.error("state_address_to_dict: cast resulted in NULL pointer.")
             return None

        c_state = state_ptr.contents # Dereference the pointer to get the structure
        # Now call the helper that works with the ctypes object
        return ctypes_state_obj_to_dict(c_state)

    except Exception as e:
        app.logger.error(f"Error converting C state address {address} to dict: {e}", exc_info=True)
        return None

# --- NEW HELPER FUNCTION ---
def ctypes_state_obj_to_dict(c_state: CtypesBoardState):
    """Converts a Python ctypes BoardState object to a dictionary."""
    if not c_state:
        app.logger.error("ctypes_state_obj_to_dict received None object.")
        return None
    try:
        py_state = {}
        py_pieces = []
        # Iterate using range (assuming 8x8 board)
        for r in range(8):
            row_list = []
            for f in range(8):
                piece_obj = c_state.pieces[r][f] # Access the nested structure
                row_list.append({'type': piece_obj.type, 'player': piece_obj.piece_player})
            py_pieces.append(row_list)

        py_state['pieces'] = py_pieces
        py_state['current_player'] = c_state.current_player
        py_state['can_castle'] = list(c_state.can_castle) # Convert C array to Python list
        py_state['in_check'] = list(c_state.in_check)     # Convert C array to Python list
        py_state['en_passant_valid'] = list(c_state.en_passant_valid) # Convert C array
        py_state['turns_since_last_capture_or_pawn'] = c_state.turns_since_last_capture_or_pawn
        py_state['status'] = c_state.status
        py_state['can_claim_draw'] = c_state.can_claim_draw
        app.logger.debug(f"ctypes_state_obj_to_dict extracted player: {py_state.get('current_player')}")
        return py_state
    except Exception as e:
        app.logger.error(f"Error converting Ctypes BoardState object to dict: {e}", exc_info=True)
        return None
# --- END NEW HELPER ---


# --- API Endpoints ---
@app.route('/api/state', methods=['GET'])
def get_state():
    """Gets the current board state dictionary."""
    if not engine:
        return jsonify({"error": "Engine not initialized"}), 500
    try:
        address = 0
        with engine_lock:
            # Get the address of the internal C++ state
            address = engine.board_state_address # Access Cython property
        py_state = state_address_to_dict(address) # Convert address to dict
        if py_state is None:
            return jsonify({"error": "Failed to read state from address"}), 500
        app.logger.info(f"GET /api/state returning state. Player: {py_state.get('current_player')}, Status: {py_state.get('status')}")
        return jsonify(py_state)
    except Exception as e:
        app.logger.error(f"/api/state error: {e}", exc_info=True)
        return jsonify({"error": "Server error getting state"}), 500


@app.route('/api/moves', methods=['GET'])
def get_valid_moves_api():
    """Gets the list of valid moves for the current state."""
    if not engine: return jsonify({"error": "Engine not initialized"}), 500
    try:
        address, count = 0, 0
        valid_moves_list = []
        with engine_lock:
            # Call the Cython method which handles C API calls
            address, count = engine.get_valid_moves_address_count()
            # Pass the address and count to the helper
            valid_moves_list = moves_to_list_ctypes(address, count)
        return jsonify(valid_moves_list)
    except AttributeError as ae:
        app.logger.error(f"/api/moves error: Cython method missing? {ae}", exc_info=True)
        return jsonify({"error": "Internal server error getting moves"}), 500
    except Exception as e:
        app.logger.error(f"/api/moves error: {e}", exc_info=True)
        return jsonify({"error": "Server error getting moves"}), 500


@app.route('/api/apply_move', methods=['POST'])
def apply_move_api(): # Conditional Workaround still present here
    """ Applies player move, gets new state POINTER from engine, returns corrected state dict. """
    if not engine: return jsonify({"error": "Chess engine not initialized"}), 500
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    data = request.json
    if not isinstance(data, dict): return jsonify({"error": "Invalid JSON format"}), 400

    previous_player = data.get('current_player', None)
    if previous_player is None:
         # Fallback: Fetch current state to determine player (adds overhead)
        try:
            with engine_lock:
                addr = engine.board_state_address
                state_dict_before = state_address_to_dict(addr)
                if state_dict_before:
                    previous_player = state_dict_before.get('current_player')
        except Exception: pass # Ignore error here
        if previous_player is None: app.logger.warning("Could not determine previous player for conditional check.")

    try: # Extract and validate move data from JSON
        start_rank = int(data['start']['rank']); start_file = int(data['start']['file'])
        target_rank = int(data['target']['rank']); target_file = int(data['target']['file'])
        promotion_data = data.get('promotion', PieceType.NONE)
        promotion_piece = int(promotion_data) if promotion_data is not None else PieceType.NONE
        if not (0 <= start_rank < 8 and 0 <= start_file < 8 and
                0 <= target_rank < 8 and 0 <= target_file < 8 and
                0 <= promotion_piece <= 6): raise ValueError("Invalid move data")
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid move data: {e}"}), 400

    try:
        new_state_dict = None
        with engine_lock:
            # 1. Get current valid moves buffer address and count
            moves_buffer_address, num_moves = engine.get_valid_moves_address_count()
            if moves_buffer_address == 0 or num_moves == 0:
                 return jsonify({"error": "No valid moves available"}), 400

            # 2. Find the matching C move ADDRESS
            move_address_to_apply = 0
            move_ptr = cast(moves_buffer_address, POINTER(CtypesChessMove))
            for i in range(num_moves):
                c_move = move_ptr[i]
                if (c_move.start_position.rank == start_rank and
                    c_move.start_position.file == start_file and
                    c_move.target_position.rank == target_rank and
                    c_move.target_position.file == target_file and
                    c_move.promotion_target == promotion_piece):
                    move_address_to_apply = moves_buffer_address + i * sizeof(CtypesChessMove)
                    break

            if move_address_to_apply == 0:
                return jsonify({"error": "Move address not found in valid moves"}), 400

            # 3. Call Cython engine.apply_move with the INTEGER ADDRESS
            # This Cython method handles calling the C API and returns the address of the *updated* internal state
            app.logger.debug(f"Calling Cython engine.apply_move with address: {move_address_to_apply}")
            new_state_address_int = engine.apply_move(move_address_to_apply) # Returns ptrdiff_t (integer address)

            if new_state_address_int == 0:
                app.logger.error("Cython engine.apply_move returned NULL address (0).")
                return jsonify({"error": "Engine failed to apply the move"}), 500
            app.logger.debug(f"Cython engine.apply_move returned address: {new_state_address_int}")

            # 4. Convert the C++ state at the returned address to dict
            app.logger.debug(f"Attempting state_address_to_dict with address: {new_state_address_int}")
            new_state_dict = state_address_to_dict(new_state_address_int) # Convert the address

            if new_state_dict is None:
                 app.logger.error(f"state_address_to_dict failed for address {new_state_address_int}")
                 return jsonify({"error": "Failed to convert new board state"}), 500
            app.logger.debug(f"state_address_to_dict returned player: {new_state_dict.get('current_player')}")

            # === CONDITIONAL WORKAROUND (Keep for apply_move for now, remove after testing AI) ===
            if 'current_player' in new_state_dict:
                player_read_from_state = new_state_dict['current_player']
                if previous_player is not None and player_read_from_state == previous_player:
                    expected_next_player = 1 - previous_player
                    app.logger.warning(f"CONDITIONAL WORKAROUND (ApplyMove): Player read ({player_read_from_state}) matches previous player ({previous_player}). Flipping to {expected_next_player}.")
                    new_state_dict['current_player'] = expected_next_player
                else:
                     app.logger.info(f"Conditional workaround (ApplyMove) NOT needed: Player read ({player_read_from_state}) is already different from previous player ({previous_player}).")
            # === END WORKAROUND ===

        # 5. Return the potentially modified dictionary
        current_p_final = new_state_dict.get('current_player', 'N/A')
        status_final = new_state_dict.get('status', 'N/A')
        app.logger.info(f"Move applied. Returning state. Player: {current_p_final}, Status: {status_final}")
        return jsonify(new_state_dict), 200

    except ValueError as e: # Catch specific errors
         app.logger.error(f"ValueError during apply_move: {e}", exc_info=True)
         return jsonify({"error": f"Engine error: {e}"}), 500
    except Exception as e:
        app.logger.error(f"Error in /api/apply_move: {e}", exc_info=True)
        return jsonify({"error": "Server error applying move"}), 500

# --- MODIFIED /api/ai_move Endpoint ---
@app.route('/api/ai_move', methods=['POST'])
def trigger_ai_move():
    """Triggers AI move, gets new state OBJECT from engine, returns corrected state dict."""
    if not engine: return jsonify({"error": "Chess engine not initialized"}), 500

    difficulty = 1
    if request.is_json and isinstance(request.json, dict):
        try:
            difficulty = int(request.json.get('difficulty', 1))
        except (ValueError, TypeError): difficulty = 1

    previous_player = None # AI player who is about to move
    try:
        new_state_dict = None
        with engine_lock:
             # --- Get current player BEFORE AI moves ---
            try:
                addr = engine.board_state_address
                state_dict_before = state_address_to_dict(addr) # Reading address here is fine
                if state_dict_before:
                    previous_player = state_dict_before.get('current_player')
                else:
                     app.logger.warning("Could not fetch state before AI move to determine previous player.")
            except Exception as e_fetch:
                 app.logger.warning(f"Error fetching state before AI move: {e_fetch}")
            # --- End Fetch ---
            if previous_player is None:
                 app.logger.error("FATAL: Could not determine which player's turn it is for AI move.")
                 return jsonify({"error": "Cannot determine AI player turn"}), 500

            app.logger.info(f"Calculating AI move for player {previous_player} (difficulty={difficulty})...")
            start_time = time.time()

            # === Call engine.ai_move wrapper, GET PYTHON BoardState OBJECT back ===
            # NOTE: Assumes ai_chess.py ai_move was updated to return BoardState object or None
            returned_state_obj = engine.ai_move(difficulty=difficulty)
            end_time = time.time()
            app.logger.info(f"AI move calculation attempt finished in {end_time - start_time:.2f} seconds.")

            # === Check if the Cython method returned a valid object ===
            if not returned_state_obj:
                 app.logger.error("engine.ai_move (Cython wrapper) returned None object.")
                 # Try to return current state if possible (using address is okay here)
                 current_address = engine.board_state_address
                 state_dict = state_address_to_dict(current_address)
                 if state_dict: return jsonify(state_dict), 200
                 else: return jsonify({"error": "AI move failed and subsequent state read failed"}), 500

            app.logger.debug(f"engine.ai_move returned Python BoardState object of type: {type(returned_state_obj)}")

            # === Convert the RETURNED CTYPES OBJECT to dict ===
            # REMOVED: Incorrect address casting/reading logic
            new_state_dict = ctypes_state_obj_to_dict(returned_state_obj) # <<< USE NEW HELPER

            if new_state_dict is None:
                app.logger.error(f"ctypes_state_obj_to_dict failed for returned object")
                return jsonify({"error": "Failed to convert board state after AI move"}), 500
            app.logger.debug(f"ctypes_state_obj_to_dict returned player: {new_state_dict.get('current_player')}")

            # === CONDITIONAL WORKAROUND (Apply to the new_state_dict) ===
            if 'current_player' in new_state_dict:
                player_read_from_state = new_state_dict['current_player']
                # Only flip if the player read matches the player *who just moved* (the AI)
                if player_read_from_state == previous_player:
                    expected_next_player = 1 - previous_player
                    app.logger.warning(f"CONDITIONAL WORKAROUND (AI): Player read ({player_read_from_state}) matches previous player ({previous_player}). Flipping to {expected_next_player}.")
                    new_state_dict['current_player'] = expected_next_player # Modify the dict
                else:
                     app.logger.info(f"Conditional workaround (AI) NOT needed: Player read ({player_read_from_state}) is already different from previous player ({previous_player}).")
            else:
                 app.logger.error("Cannot apply conditional workaround (AI): 'current_player' key missing.")
            # === END WORKAROUND ===

        # Return the potentially modified dictionary
        current_p_final = new_state_dict.get('current_player', 'N/A')
        status_final = new_state_dict.get('status', 'N/A')
        # This log should now show the player *after* the workaround
        app.logger.info(f"AI moved. Returning state. Player: {current_p_final}, Status: {status_final}")
        return jsonify(new_state_dict), 200

    except Exception as e:
        app.logger.error(f"Error in /api/ai_move: {e}", exc_info=True)
        return jsonify({"error": "Server error during AI move"}), 500
# --- END MODIFIED /api/ai_move ---

@app.route('/api/reset', methods=['POST'])
def reset_game():
    """Resets the chess engine to the initial state."""
    global engine
    with engine_lock:
        app.logger.info("Received request to reset engine...");
        try:
            # Re-validate model path before creating new engine
            abs_model_path = os.path.abspath(os.path.join(project_root, MODEL_PATH))
            if not os.path.exists(abs_model_path):
                raise FileNotFoundError(f"Model not found during reset: {abs_model_path}")

            # No need for library path if Cython handles linking
            abs_lib_path = os.path.abspath(os.path.join(project_root, LIBRARY_PATH)) # Keep for info if needed

            if engine:
                try:
                    # Assuming Cython wrapper has a close or __dealloc__
                    # No explicit close needed if using 'with' or relying on GC,
                    # but creating a new instance should replace the old one.
                    pass # Let the old engine be garbage collected if not using 'with'
                except Exception as close_err:
                    app.logger.warning(f"Error potentially closing old engine instance: {close_err}")

            # Create a new engine instance
            engine = ChessEngine(library_path=abs_lib_path, model_path=abs_model_path)
            app.logger.info("Engine re-initialized.")

            # Get initial state from the *new* engine instance
            address = engine.board_state_address # Get address from new engine
            state_dict = state_address_to_dict(address) # Convert address

            if state_dict is None:
                return jsonify({"error": "Engine reset but failed to read initial state"}), 500
            return jsonify({"message": "Game reset", "initial_state": state_dict}), 200
        except Exception as e:
            app.logger.error(f"Error during reset: {e}", exc_info=True)
            engine = None # Ensure engine is None if reset failed
            return jsonify({"error": f"Failed to reset engine: {e}"}), 500

# --- Run the Server ---
if __name__ == '__main__':
    if not engine:
        print("FATAL: Engine failed initialization during script setup.", file=sys.stderr)
        sys.exit(1)
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask server (Sync AI) on port {port}...")
    # Ensure debug=False for production/Render
    # threaded=False is often safer with C extensions that might not be thread-safe
    app.run(debug=False, host='0.0.0.0', port=port, threaded=False)