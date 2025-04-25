# server.py (Workaround: Manually flip player in returned dict)
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
    from chess_dir.ai_chess import ChessEngine, Player, PieceType, GameStatus, MoveType
except ImportError as e:
    print(f"Error importing ChessEngine/Enums: {e}"); sys.exit(1)

# --- Configuration ---
LIBRARY_PATH = "build/src/libchess_cpp.so"
MODEL_PATH = "model/trained_nnue.onnx"

# --- ctypes Structures ---
class CtypesBoardPosition(Structure): _pack_=1; _fields_=[("rank",c_uint8),("file",c_uint8)]
class CtypesPiece(Structure): _pack_=1; _fields_=[("type",c_int8),("piece_player",c_int8)]
class CtypesChessMove(Structure): _pack_=1; _fields_=[("type",c_int8),("start_position",CtypesBoardPosition),("target_position",CtypesBoardPosition),("promotion_target",c_int8)]
class CtypesBoardState(Structure): _pack_=1; _fields_=[("pieces",(CtypesPiece*8)*8),("can_castle",c_bool*4),("in_check",c_bool*2),("en_passant_valid",c_bool*16),("turns_since_last_capture_or_pawn",c_int32),("current_player",c_int8),("status",c_int8),("can_claim_draw",c_bool),("_padding_",c_uint8*3)]

# --- Flask App Setup ---
app = Flask(__name__)
frontend_url = os.environ.get("FRONTEND_URL", "https://sp14-chessai-red-2025.github.io")
CORS(app, resources={r"/api/*": {"origins": [frontend_url]}})
app.logger.info(f"CORS enabled for origin: {frontend_url}")

# --- Global Engine Instance & Lock ---
engine_lock = threading.Lock()
engine = None

# --- Engine Initialization ---
try:
    abs_lib_path = os.path.abspath(os.path.join(project_root, LIBRARY_PATH))
    abs_model_path = os.path.abspath(os.path.join(project_root, MODEL_PATH))
    if not os.path.exists(abs_lib_path): raise FileNotFoundError(f"Lib not found: {abs_lib_path}")
    if not os.path.exists(abs_model_path): raise FileNotFoundError(f"Model not found: {abs_model_path}")
    engine = ChessEngine(library_path=abs_lib_path, model_path=abs_model_path)
    app.logger.info("Chess Engine loaded successfully.")
except Exception as init_error:
    # NOTE: The user reported the server still fails to start here.
    # This workaround won't help if initialization fails.
    app.logger.error(f"FATAL: Failed to initialize ChessEngine: {init_error}", exc_info=True)

# --- Helper Functions ---
# (moves_to_list_ctypes and state_address_to_dict remain the same as before)
def moves_to_list_ctypes(move_buffer_address, num_moves):
    if not engine or move_buffer_address == 0 or num_moves <= 0: return []
    move_list = []; move_ptr = cast(move_buffer_address, POINTER(CtypesChessMove))
    try:
        for i in range(num_moves):
            current_move = move_ptr[i]; current_move_addr = move_buffer_address + i * sizeof(CtypesChessMove)
            # Assuming move_to_str still exists and works with addresses
            san = engine.move_to_str(current_move_addr) if engine else "ERR!"
            move_dict = {"type":current_move.type,"start":{"rank":current_move.start_position.rank,"file":current_move.start_position.file},"target":{"rank":current_move.target_position.rank,"file":current_move.target_position.file},"promotion":current_move.promotion_target,"san":san}; move_list.append(move_dict)
    except Exception as e: app.logger.error(f"Error processing ctypes move buffer: {e}", exc_info=True); return []
    return move_list

def state_address_to_dict(address):
    if address == 0: app.logger.error("state_address_to_dict received NULL address."); return None
    try:
        state_ptr = cast(address, POINTER(CtypesBoardState));
        if not state_ptr: app.logger.error("state_address_to_dict: cast NULL ptr."); return None
        c_state = state_ptr.contents; py_state = {}; py_pieces = []
        for r in range(8): py_pieces.append([{'type':c_state.pieces[r][f].type, 'player':c_state.pieces[r][f].piece_player} for f in range(8)])
        py_state['pieces'] = py_pieces; py_state['current_player'] = c_state.current_player; py_state['can_castle'] = list(c_state.can_castle); py_state['in_check'] = list(c_state.in_check); py_state['en_passant_valid'] = list(c_state.en_passant_valid); py_state['turns_since_last_capture_or_pawn'] = c_state.turns_since_last_capture_or_pawn; py_state['status'] = c_state.status; py_state['can_claim_draw'] = c_state.can_claim_draw
        return py_state
    except Exception as e: app.logger.error(f"Error converting C state addr {address}: {e}", exc_info=True); return None

# --- API Endpoints ---
@app.route('/api/state', methods=['GET'])
def get_state():
    if not engine:
        return jsonify({"error": "Engine not initialized"}), 500
    try:
        address = 0
        with engine_lock:
            # Get the address of the current C++ board state
            address = engine.board_state_address

        # Convert the C++ state to a Python dictionary
        py_state = state_address_to_dict(address)

        if py_state is None:
            return jsonify({"error": "Failed to read state"}), 500

        # === WORKAROUND: Manually flip player in the *dictionary* ===
        # Apply the same workaround here as in apply_move/ai_move
        # This corrects the player if the C++ state pointer was stale
        # NOTE: This assumes the state read here *might* be immediately
        #       after a move where the C++ pointer wasn't updated yet.
        #       If this GET is always *before* a move, this might
        #       incorrectly flip the player. Monitor logs.
        if 'current_player' in py_state:
            original_player = py_state['current_player']
            # Flip 0 to 1, 1 to 0.
            # It's safer to check the actual player value in C++ 
            py_state['current_player'] = 1 - original_player
            app.logger.warning(f"WORKAROUND applied in GET /api/state: Flipped current_player in dict from {original_player} to {py_state['current_player']}")
        else:
             app.logger.error("Cannot apply workaround in GET /api/state: 'current_player' key missing.")
        # === END WORKAROUND ===

        # Return the MODIFIED dictionary
        app.logger.info(f"GET /api/state returning state (Python state modified). Player: {py_state.get('current_player')}, Status: {py_state.get('status')}")
        return jsonify(py_state)

    except Exception as e:
        app.logger.error(f"/api/state error: {e}", exc_info=True)
        return jsonify({"error": "Server error getting state"}), 500

@app.route('/api/moves', methods=['GET'])
def get_valid_moves_api():
    # (Unchanged - Reads valid moves based on current C++ state)
    if not engine: return jsonify({"error": "Engine not initialized"}), 500
    try:
        address, count = 0, 0; valid_moves_list = []
        with engine_lock:
            address, count = engine.get_valid_moves_address_count()
            valid_moves_list = moves_to_list_ctypes(address, count) # Needs lock for move_to_str
        return jsonify(valid_moves_list)
    except Exception as e: app.logger.error(f"/api/moves error: {e}", exc_info=True); return jsonify({"error": "Failed moves"}), 500

@app.route('/api/apply_move', methods=['POST'])
def apply_move_api(): # WORKAROUND ADDED
    """ Applies player move, gets new state address from engine, returns state dict. """
    if not engine: return jsonify({"error": "Chess engine not initialized"}), 500
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    data = request.json;
    if not isinstance(data, dict): return jsonify({"error": "Invalid JSON format"}), 400

    try: # Extract and validate move data
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
            # 1. Get current valid moves buffer
            moves_buffer_address, num_moves = engine.get_valid_moves_address_count()
            if moves_buffer_address == 0 or num_moves == 0:
                 return jsonify({"error": "No valid moves available"}), 400

            # 2. Find the matching C move address
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
                return jsonify({"error": "Move not found in valid moves"}), 400

            # 3. Call engine.apply_move - gets the address of the (potentially unchanged) C++ state
            new_state_address = engine.apply_move(move_address_to_apply) # <-- Gets potentially incorrect state address

            if new_state_address == 0:
                app.logger.error("engine.apply_move returned NULL address.")
                return jsonify({"error": "Engine failed to apply the move"}), 500

            # 4. Convert the C++ state (potentially with wrong player) to dict
            new_state_dict = state_address_to_dict(new_state_address)
            if new_state_dict is None:
                 return jsonify({"error": "Failed to convert new board state"}), 500

            # === WORKAROUND: Manually flip player in the *dictionary* ===
            if 'current_player' in new_state_dict:
                original_player = new_state_dict['current_player']
                new_state_dict['current_player'] = 1 - original_player # Flip 0 to 1, 1 to 0
                app.logger.warning(f"WORKAROUND applied: Flipped current_player in dict from {original_player} to {new_state_dict['current_player']}")
            else:
                 app.logger.error("Cannot apply workaround: 'current_player' key missing in state dict.")
            # === END WORKAROUND ===

        # 5. Return the MODIFIED dictionary
        app.logger.info(f"Move applied (Python state modified). Returning state. Player: {new_state_dict.get('current_player')}, Status: {new_state_dict.get('status')}")
        return jsonify(new_state_dict), 200

    except ValueError as e: # Catch specific errors from Cython/C++
         app.logger.error(f"ValueError during apply_move: {e}", exc_info=True)
         return jsonify({"error": f"Engine error: {e}"}), 500 # Or 400?
    except Exception as e:
        app.logger.error(f"Error in /api/apply_move: {e}", exc_info=True)
        return jsonify({"error": "Server error applying move"}), 500


@app.route('/api/ai_move', methods=['POST'])
def trigger_ai_move(): # WORKAROUND ADDED
    """Triggers AI move, gets new state address from engine, returns state dict."""
    if not engine: return jsonify({"error": "Chess engine not initialized"}), 500

    difficulty = 5
    if request.is_json and isinstance(request.json, dict):
        try: difficulty = int(request.json.get('difficulty', 5))
        except (ValueError, TypeError): difficulty = 5

    try:
        new_state_dict = None
        with engine_lock:
            app.logger.info(f"Calculating AI move (difficulty={difficulty})...")
            start_time = time.time()

            # === Call engine.ai_move - gets the address of the (potentially unchanged) C++ state ===
            new_state_address = engine.ai_move(difficulty=difficulty) # <-- Gets potentially incorrect state address
            end_time = time.time()
            app.logger.info(f"AI move calculation attempt finished in {end_time - start_time:.2f} seconds.")

            if new_state_address == 0:
                 app.logger.error("engine.ai_move returned NULL address.")
                 current_address = engine.board_state_address
                 state_dict = state_address_to_dict(current_address)
                 if state_dict: return jsonify(state_dict), 200 # Return current state if readable
                 else: return jsonify({"error": "AI move failed and subsequent state read failed"}), 500

            # Convert the C++ state (potentially with wrong player) to dict
            new_state_dict = state_address_to_dict(new_state_address)
            if new_state_dict is None:
                return jsonify({"error": "Failed to convert board state after AI move"}), 500

            # === WORKAROUND: Manually flip player in the *dictionary* ===
            if 'current_player' in new_state_dict:
                original_player = new_state_dict['current_player']
                new_state_dict['current_player'] = 1 - original_player # Flip 0 to 1, 1 to 0
                app.logger.warning(f"WORKAROUND applied: Flipped current_player in dict from {original_player} to {new_state_dict['current_player']}")
            else:
                 app.logger.error("Cannot apply workaround: 'current_player' key missing in state dict.")
            # === END WORKAROUND ===

        # Return the MODIFIED dictionary
        app.logger.info(f"AI moved (Python state modified). Returning state. Player: {new_state_dict.get('current_player')}, Status: {new_state_dict.get('status')}")
        return jsonify(new_state_dict), 200

    except Exception as e:
        app.logger.error(f"Error in /api/ai_move: {e}", exc_info=True)
        return jsonify({"error": "Server error during AI move"}), 500

# --- reset endpoint remains the same ---
@app.route('/api/reset', methods=['POST'])
def reset_game():
    global engine
    with engine_lock:
        app.logger.info("Received request to reset engine...");
        try:
            abs_lib_path=os.path.abspath(os.path.join(project_root,LIBRARY_PATH));abs_model_path=os.path.abspath(os.path.join(project_root,MODEL_PATH))
            if not os.path.exists(abs_lib_path): raise FileNotFoundError(f"Lib not found: {abs_lib_path}")
            if not os.path.exists(abs_model_path): raise FileNotFoundError(f"Model not found: {abs_model_path}")
            if engine: 
                try: 
                    engine.close() 
                except Exception as close_err: 
                    app.logger.warning(f"Error closing old engine: {close_err}")
            engine = ChessEngine(library_path=abs_lib_path, model_path=abs_model_path); app.logger.info("Engine re-initialized.")
            address = engine.board_state_address; state_dict = state_address_to_dict(address)
            if state_dict is None: return jsonify({"error": "Engine reset but failed state"}), 500
            return jsonify({"message": "Game reset", "initial_state": state_dict}), 200
        except Exception as e: app.logger.error(f"Error during reset: {e}", exc_info=True); engine = None; return jsonify({"error": f"Failed reset: {e}"}), 500

# --- Run the Server ---
if __name__ == '__main__':
    if not engine: print("FATAL: Engine failed init.", file=sys.stderr); sys.exit(1)
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask server (Sync AI) on port {port}...")
    app.run(debug=False, host='0.0.0.0', port=port, threaded=False)