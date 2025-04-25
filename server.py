# server.py (Synchronous AI Mode)
import os
import sys
import traceback
import threading # Keep for Lock
import time
from flask import Flask, jsonify, request
from flask_cors import CORS

# --- Path Setup ---
project_root = os.path.dirname(os.path.abspath(__file__))
python_dir_path = os.path.join(project_root, 'python')
if python_dir_path not in sys.path:
    sys.path.append(python_dir_path)
src_dir_path = os.path.join(project_root, 'src')
if src_dir_path not in sys.path:
     sys.path.append(src_dir_path)

try:
    # Ensure you are importing from the correct relative path
    from chess_dir.ai_chess import ChessEngine, Player, PieceType, GameStatus, ChessMove, BoardPosition, MoveType
except ImportError as e:
    print(f"Error importing ChessEngine: {e}")
    print(f"Attempted import from: {os.path.join(python_dir_path, 'chess_dir')}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

# --- Configuration --- (Ensure these paths are correct relative to server.py)
LIBRARY_PATH = "build/src/libchess_cpp.so"
MODEL_PATH = "model/trained_nnue.onnx" # Make sure this is your model file

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)

# --- Global Chess Engine Instance & Lock ---
engine_lock = threading.Lock() # Keep lock for sequential access

# --- Engine Initialization ---
engine = None # Initialize as None
try:
    # Check paths carefully
    abs_lib_path = os.path.abspath(os.path.join(project_root, LIBRARY_PATH))
    abs_model_path = os.path.abspath(os.path.join(project_root, MODEL_PATH))
    print(f"Checking Library Path: {abs_lib_path}")
    print(f"Checking Model Path: {abs_model_path}")

    if not os.path.exists(abs_lib_path): raise FileNotFoundError(f"Library not found: {abs_lib_path}")
    if not os.path.exists(abs_model_path): raise FileNotFoundError(f"Model not found: {abs_model_path}")

    print(f"Loading Chess Engine with Lib: {abs_lib_path}, Model: {abs_model_path}")
    # Use absolute paths for engine creation
    engine = ChessEngine(library_path=abs_lib_path, model_path=abs_model_path)
    print("Chess Engine loaded successfully.")
except Exception as init_error:
    print(f"FATAL: Failed to initialize ChessEngine: {init_error}")
    print(traceback.format_exc())
    # Keep engine as None

# --- Helper Functions ---
def board_state_to_dict(state):
    if not state: return None
    pieces_list = []
    for r in range(8):
        row_list = []
        for f in range(8):
            piece = state.pieces[r][f]
            row_list.append({"type": piece.type, "player": piece.piece_player})
        pieces_list.append(row_list)
    return {
        "pieces": pieces_list,
        "can_castle": list(state.can_castle),
        "in_check": list(state.in_check),
        "turns_since_capture_or_pawn": state.turns_since_last_capture_or_pawn,
        "current_player": state.current_player,
        "status": state.status,
        "can_claim_draw": bool(state.can_claim_draw)
        # No ai_is_thinking needed
    }

def moves_to_list(moves):
    if not engine: return []
    move_list = []
    for move in moves: # 'move' is a chess_dir.ai_chess.ChessMove object
        try: san = engine.move_to_str(move)
        except Exception as e_san:
            print(f"Warning: move_to_str failed for move: {move} - Error: {e_san}")
            san = "err"

        # Access .start and .target properties, then .rank and .file
        move_list.append({
            "type": move.type,
            "start": {"rank": move.start.rank, "file": move.start.file},
            "target": {"rank": move.target.rank, "file": move.target.file},
            "promotion": move.promotion, # Access promotion property directly
            "san": san
        })

    return move_list

# --- API Endpoints ---

@app.route('/api/state', methods=['GET'])
def get_state():
    """Returns the current board state."""
    if not engine: return jsonify({"error": "Chess engine not initialized on server"}), 500
    try:
        with engine_lock:
             # Get the dictionary directly from the Cython property
             current_state_dict = engine.board_state
        if current_state_dict is None:
             print("[ERROR] /api/state: engine.board_state returned None")
             return jsonify({"error": "Engine state is unavailable"}), 500

        return jsonify(current_state_dict)

    except Exception as e:
        print(f"Error in /api/state: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Failed to get board state"}), 500

@app.route('/api/moves', methods=['GET'])
def get_valid_moves_api():
    """Returns a list of valid moves for the current player."""
    if not engine: return jsonify({"error": "Chess engine not initialized"}), 500
    try:
        # Assuming get_valid_moves is safe without lock if it reads consistent state
        valid_moves = engine.get_valid_moves()
        return jsonify(moves_to_list(valid_moves))
    except Exception as e:
        print(f"Error in /api/moves: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Failed to get valid moves"}), 500

@app.route('/api/apply_move', methods=['POST'])
def apply_move_api():
    """Applies a player move and returns the new state."""
    if not engine: return jsonify({"error": "Chess engine not initialized"}), 500

    move_data = request.json
    if not move_data: return jsonify({"error": "No move data provided"}), 400

    try:
        with engine_lock:
            # --- Find move object ---
            valid_moves = engine.get_valid_moves()
            move_to_apply = None
            # Get coordinates from request JSON
            start_rank = move_data.get('start', {}).get('rank'); start_file = move_data.get('start', {}).get('file')
            target_rank = move_data.get('target', {}).get('rank'); target_file = move_data.get('target', {}).get('file')
            promo_piece = move_data.get('promotion') # Might be None

            if start_rank is None or start_file is None or target_rank is None or target_file is None:
                return jsonify({"error": "Incomplete move coordinates"}), 400

            # Iterate through the Cython ChessMove objects
            for move in valid_moves:
                # *** CORRECTION HERE ***
                # Use move.start and move.target properties
                coords_match = (move.start.rank == start_rank and move.start.file == start_file and
                                move.target.rank == target_rank and move.target.file == target_file)
                # *** END CORRECTION ***

                if not coords_match: continue

                # Check promotion match if applicable
                if move.type == MoveType.PROMOTION: # Assuming MoveType is available
                    # Compare the integer value from JSON (if present) to the move's promotion property
                    if promo_piece is not None and move.promotion == promo_piece:
                         move_to_apply = move
                         break
                    # If promo_piece from frontend is None, but move IS a promotion, it doesn't match
                    elif promo_piece is None:
                         continue # This specific loop iteration doesn't match
                else: # Normal move, capture, castle, en passant
                    move_to_apply = move
                    break
            # --- End Matching Logic ---

            if not move_to_apply:
                print(f"Could not find valid move matching: {move_data}")
                return jsonify({"error": "Invalid or ambiguous move"}), 400

            print(f"Applying move via API: {engine.move_to_str(move_to_apply)}")
            updated_state_dict = engine.apply_move(move_to_apply) # Cython engine returns dict

            if updated_state_dict is None:
                 print("[API WARNING] engine.apply_move returned None")
                 return jsonify({"error": "Engine failed to apply move internally"}), 500

            print(f"Move applied successfully. New current player: {updated_state_dict.get('current_player', 'N/A')}")

        return jsonify(updated_state_dict) # Return new state dict

    except Exception as e:
        print(f"Error in /api/apply_move: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Failed to apply move due to server error"}), 500


# --- MODIFIED AI Move Endpoint (Synchronous) ---
@app.route('/api/ai_move', methods=['POST'])
def trigger_ai_move():
    """Triggers the AI to make a move SYNCHRONOUSLY and returns the new state."""
    if not engine: return jsonify({"error": "Chess engine not initialized"}), 500

    try:
        with engine_lock:
            difficulty = 5
            if request.is_json:
                difficulty = request.json.get('difficulty', 5)
            else:
                 print("Warning: /api/ai_move called without JSON body, using default difficulty.")

            print(f"Calculating AI move (blocking)...")
            start_time = time.time()

            # --- Direct call to engine ---
            engine.ai_move(difficulty=difficulty) # Blocks until C++ returns

            end_time = time.time()
            print(f"AI move calculated in {end_time - start_time:.2f} seconds.")

            # Fetch the state *after* ai_move completed (it updates internal state)
            new_state_after_ai = engine.board_state # This is ALREADY the dictionary
            if new_state_after_ai is None:
                 print("[ERROR] engine.board_state is None after AI move!")
                 return jsonify({"error": "AI move completed but engine state is unavailable"}), 500

            # --- REMOVE THIS LINE: ---
            # state_dict = board_state_to_dict(new_state_after_ai) # NO LONGER NEEDED

        # *** CHANGE HERE: Return the dictionary directly ***
        print(f"Returning state after AI move. New current player: {new_state_after_ai.get('current_player', 'N/A')}")
        return jsonify(new_state_after_ai), 200
        # *** Use new_state_after_ai directly ***

    except Exception as e:
        print(f"Error in /api/ai_move (Sync): {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Failed to calculate AI move"}), 500

@app.route('/api/reset', methods=['POST'])
def reset_game():
    """Destroys and re-initializes the chess engine."""
    global engine # Declare intent to modify the global engine variable
    with engine_lock:
        print("Received request to reset engine...")
        if engine:
            try:
                # Assuming engine has a close/destroy method exposed via Cython __dealloc__
                # Re-creating it should trigger the necessary cleanup and re-init
                print("Attempting to re-initialize engine...")
                # Get paths again (could store them globally too)
                abs_lib_path = os.path.abspath(os.path.join(project_root, LIBRARY_PATH))
                abs_model_path = os.path.abspath(os.path.join(project_root, MODEL_PATH))
                # Create new instance - the old one should be garbage collected, triggering __dealloc__
                engine = ChessEngine(library_path=abs_lib_path, model_path=abs_model_path)
                print("Engine re-initialized successfully.")
                return jsonify({"message": "Game reset successfully"}), 200
            except Exception as e:
                print(f"Error during engine reset: {e}\n{traceback.format_exc()}")
                # Try to set engine to None to indicate failure state
                engine = None
                return jsonify({"error": f"Failed to reset engine: {e}"}), 500
        else:
            # If engine was already None, try to initialize it
            try:
                abs_lib_path = os.path.abspath(os.path.join(project_root, LIBRARY_PATH))
                abs_model_path = os.path.abspath(os.path.join(project_root, MODEL_PATH))
                engine = ChessEngine(library_path=abs_lib_path, model_path=abs_model_path)
                print("Engine initialized successfully during reset request.")
                return jsonify({"message": "Game reset successfully"}), 200
            except Exception as e:
                 print(f"Error during engine initialization on reset: {e}\n{traceback.format_exc()}")
                 engine = None
                 return jsonify({"error": f"Failed to initialize engine on reset: {e}"}), 500

# --- Run the Server ---
if __name__ == '__main__':
    if not engine:
        print("Cannot start server: Chess engine failed to initialize.")
    else:
        print("Starting Flask server (Synchronous AI Mode)...")
        # Run explicitly non-threaded for synchronous blocking behavior
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=False)