# server.py
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
    # Make sure ai_chess.py has the updated methods returning POINTER(BoardState)
    from chess_dir.ai_chess import ChessEngine, Player, PieceType, GameStatus, MoveType
except ImportError as e:
    print(f"Error importing ChessEngine/Enums: {e}"); sys.exit(1)
except AttributeError as e:
     print(f"Error importing specific ctypes structures (check ai_chess.py or definitions here): {e}"); sys.exit(1)


# --- Configuration ---
LIBRARY_PATH = "build/src/libchess_cpp.so"
MODEL_PATH = "model/trained_nnue.onnx"

# --- ctypes Structures (Ensure these match ai_chess.py if defined there) ---
class CtypesBoardPosition(Structure): 
    _fields_=[("rank",c_uint8),("file",c_uint8)]
class CtypesPiece(Structure): 
    _fields_=[("type",c_int8),("piece_player",c_int8)]
class CtypesChessMove(Structure): 
    _fields_=[("type",c_int8),("start_position",CtypesBoardPosition),("target_position",CtypesBoardPosition),("promotion_target",c_int8)]
class CtypesBoardState(Structure): 
    _fields_=[("pieces",(CtypesPiece*8)*8),("can_castle",c_bool*4),
            ("in_check",c_bool*2),("en_passant_valid",c_bool*16),("turns_since_last_capture_or_pawn",c_int32),
            ("current_player",c_int8),("status",c_int8),("can_claim_draw",c_bool)]

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
    # Pass paths to ChessEngine constructor
    engine = ChessEngine(library_path=abs_lib_path, model_path=abs_model_path)
    app.logger.info("Chess Engine loaded successfully.")
except Exception as init_error:
    app.logger.error(f"FATAL: Failed to initialize ChessEngine: {init_error}", exc_info=True)
    # Consider exiting if engine init fails: sys.exit(1)


# --- Helper Functions ---
def moves_to_list_ctypes(move_buffer_address, num_moves):
    if not engine or move_buffer_address == 0 or num_moves <= 0: return []
    move_list = []; move_ptr = cast(move_buffer_address, POINTER(CtypesChessMove))
    try:
        for i in range(num_moves):
            current_move = move_ptr[i]; current_move_addr = move_buffer_address + i * sizeof(CtypesChessMove)
            san = engine.move_to_str(current_move_addr) if engine else "ERR!" # Assuming move_to_str exists
            move_dict = {"type":current_move.type,"start":{"rank":current_move.start_position.rank,"file":current_move.start_position.file},"target":{"rank":current_move.target_position.rank,"file":current_move.target_position.file},"promotion":current_move.promotion_target,"san":san}; move_list.append(move_dict)
    except Exception as e: app.logger.error(f"Error processing ctypes move buffer: {e}", exc_info=True); return []
    return move_list

def state_address_to_dict(address):
    if address == 0: app.logger.error("state_address_to_dict received NULL address."); return None
    try:
        state_ptr = cast(address, POINTER(CtypesBoardState))
        if not state_ptr: app.logger.error("state_address_to_dict: cast NULL ptr."); return None
        c_state = state_ptr.contents
        py_state = {}
        py_pieces = []
        for r in range(8): 
            py_pieces.append([{'type':c_state.pieces[r][f].type, 'player':c_state.pieces[r][f].piece_player} for f in range(8)])
        py_state['pieces'] = py_pieces
        py_state['current_player'] = c_state.current_player
        py_state['can_castle'] = list(c_state.can_castle)
        py_state['in_check'] = list(c_state.in_check)
        py_state['en_passant_valid'] = list(c_state.en_passant_valid)
        py_state['turns_since_last_capture_or_pawn'] = c_state.turns_since_last_capture_or_pawn
        py_state['status'] = c_state.status
        py_state['can_claim_draw'] = c_state.can_claim_draw
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
            address = engine.board_state_address
        py_state = state_address_to_dict(address)
        if py_state is None:
            return jsonify({"error": "Failed to read state"}), 500
        app.logger.info(f"GET /api/state returning state. Player: {py_state.get('current_player')}, Status: {py_state.get('status')}")
        return jsonify(py_state)
    except Exception as e:
        app.logger.error(f"/api/state error: {e}", exc_info=True)
        return jsonify({"error": "Server error getting state"}), 500


@app.route('/api/moves', methods=['GET'])
def get_valid_moves_api():
    if not engine: return jsonify({"error": "Engine not initialized"}), 500
    try:
        address, count = 0, 0; valid_moves_list = []
        with engine_lock:
            # Ensure get_valid_moves_address_count exists and works
            address, count = engine.get_valid_moves_address_count()
            valid_moves_list = moves_to_list_ctypes(address, count) # Needs lock for move_to_str
        return jsonify(valid_moves_list)
    except Exception as e: app.logger.error(f"/api/moves error: {e}", exc_info=True); return jsonify({"error": "Failed moves"}), 500


@app.route('/api/apply_move', methods=['POST'])
def apply_move_api():
    """ Applies player move, gets new state pointer from engine, returns corrected state dict. """
    if not engine: 
        return jsonify({"error": "Chess engine not initialized"}), 500
    if not request.is_json: 
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.json;
    if not isinstance(data, dict): 
        return jsonify({"error": "Invalid JSON format"}), 400

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
            # 1. Get current valid moves buffer address
            moves_buffer_address, num_moves = engine.get_valid_moves_address_count()
            if moves_buffer_address == 0 or num_moves == 0:
                 return jsonify({"error": "No valid moves available"}), 400

            # 2. Find the matching C move ADDRESS
            move_address_to_apply = 0 # This will be the integer address
            move_ptr = cast(moves_buffer_address, POINTER(CtypesChessMove))
            for i in range(num_moves):
                c_move = move_ptr[i]
                if (c_move.start_position.rank == start_rank and
                    c_move.start_position.file == start_file and
                    c_move.target_position.rank == target_rank and
                    c_move.target_position.file == target_file and
                    c_move.promotion_target == promotion_piece):
                    # Calculate the address for the matching move
                    move_address_to_apply = moves_buffer_address + i * sizeof(CtypesChessMove)
                    break

            if move_address_to_apply == 0:
                return jsonify({"error": "Move address not found in valid moves"}), 400

            # 3. Call engine.apply_move (from Cython) with the INTEGER ADDRESS
            app.logger.debug(f"Calling engine.apply_move with address: {move_address_to_apply}")
            # Pass the integer address directly
            app.logger.debug(f"Before apply_move: {state_address_to_dict(engine.board_state_address)}")
            new_state_address_int = engine.apply_move(move_address_to_apply)
            app.logger.debug(f"After apply_move: {state_address_to_dict(new_state_address_int)}")

            # Note: The Cython apply_move returns ptrdiff_t which is the address.
            # We rename the variable for clarity.
            if new_state_address_int == 0:
                app.logger.error("engine.apply_move (Cython) returned NULL address (0).")
                return jsonify({"error": "Engine failed to apply the move"}), 500
            app.logger.debug(f"engine.apply_move returned address: {new_state_address_int}")


            # 4. Convert the C++ state at the returned address to dict
            app.logger.debug(f"Attempting state_address_to_dict with address: {new_state_address_int}")
            new_state_dict = state_address_to_dict(new_state_address_int) # Pass the integer address

            if new_state_dict is None:
                 app.logger.error(f"state_address_to_dict failed for address {new_state_address_int}")
                 return jsonify({"error": "Failed to convert new board state"}), 500
            app.logger.debug(f"state_address_to_dict returned player: {new_state_dict.get('current_player')}")


        new_state_dict['current_player'] = (Player.BLACK if new_state_dict['current_player'] == Player.WHITE else Player.WHITE)

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


# Add a global variable to track the AI's color
ai_player = None  # Default to None for AI vs AI mode

@app.route('/api/ai_move', methods=['POST'])
def trigger_ai_move():
    """Triggers AI move, gets new state pointer from engine, returns corrected state dict."""
    global ai_player

    if not engine:
        return jsonify({"error": "Chess engine not initialized"}), 500

    difficulty = 1
    game_mode = None  # Default game mode
    previous_player = None  # Variable to remember the player of the input state_dict

    if request.is_json and isinstance(request.json, dict):
        try:
            difficulty = int(request.json.get('difficulty', 2))
            game_mode = request.json.get('game_mode', None)  # Retrieve game mode from the request
        except (ValueError, TypeError):
            difficulty = 2

    try:
        new_state_dict = None
        with engine_lock:
            # Check if it's the AI's turn, unless in AI vs AI mode
            current_address = engine.board_state_address
            current_state = state_address_to_dict(current_address)
            if current_state is None:
                return jsonify({"error": "Failed to retrieve current state"}), 500

            # Remember the current player of the input state_dict
            previous_player = current_state.get('current_player')

            if ai_player is not None and previous_player != ai_player:
                app.logger.warning("AI attempted to move out of turn. Correcting current player.")
                # Update the current player to the correct one
                if previous_player == Player.BLACK:
                    current_state['current_player'] = Player.WHITE
                else:
                    current_state['current_player'] = Player.BLACK
                return jsonify(current_state), 200

            # Proceed with AI move
            start_time = time.time()
            new_state_ptr = engine.ai_move(difficulty=difficulty)  # Assumes wrapper returns POINTER(BoardState)
            end_time = time.time()
            app.logger.info(f"AI move calculation attempt finished in {end_time - start_time:.2f} seconds.")

            if not new_state_ptr:
                app.logger.error("engine.ai_move (wrapper) returned None pointer.")
                # Try to return current state if possible
                state_dict = state_address_to_dict(current_address)
                if state_dict:
                    return jsonify(state_dict), 200
                else:
                    return jsonify({"error": "AI move failed and subsequent state read failed"}), 500

            app.logger.debug(f"engine.ai_move returned pointer: {new_state_ptr}")

            # Convert state using the returned pointer
            new_state_address = cast(new_state_ptr, c_void_p).value
            app.logger.debug(f"Attempting state_address_to_dict with address: {new_state_address}")
            new_state_dict = state_address_to_dict(new_state_address)
            if new_state_dict is None:
                app.logger.error(f"state_address_to_dict failed for address {new_state_address}")
                return jsonify({"error": "Failed to convert board state after AI move"}), 500
            app.logger.debug(f"state_address_to_dict returned player: {new_state_dict.get('current_player')}")

            # Compare the new state's current player with the previous player
            if new_state_dict['current_player'] == previous_player:
                app.logger.info("New state's current player is the same as the previous player. Switching player.")
                # Switch the player
                if new_state_dict['current_player'] == Player.BLACK:
                    new_state_dict['current_player'] = Player.WHITE
                else:
                    new_state_dict['current_player'] = Player.BLACK

        # Return the potentially modified dictionary
        current_p_final = new_state_dict.get('current_player', 'N/A')
        status_final = new_state_dict.get('status', 'N/A')
        app.logger.info(f"AI moved. Returning state. Player: {current_p_final}, Status: {status_final}")
        return jsonify(new_state_dict), 200

    except Exception as e:
        app.logger.error(f"Error in /api/ai_move: {e}", exc_info=True)
        return jsonify({"error": "Server error during AI move"}), 500

@app.route('/api/evaluate', methods=['GET'])
def evaluate_board():
    global engine
    with engine_lock:
        try:
            evaluation = engine.evaluate_board()  # Call the Python wrapper method
            return jsonify({"evaluation": evaluation}), 200
        except Exception as e:
            app.logger.error(f"Error evaluating board: {e}", exc_info=True)
            return jsonify({"error": f"Failed to evaluate board: {e}"}), 500

# --- reset endpoint remains the same ---
@app.route('/api/reset', methods=['POST'])
def reset_game():
    global engine
    with engine_lock:
        app.logger.info("Received request to reset engine...")
        try:
            # Call the reset method on the engine
            engine.reset()  # Assuming `reset` is a method in the ChessEngine class

            # Get the initial board state after resetting
            address = engine.board_state_address
            state_dict = state_address_to_dict(address)
            if state_dict is None:
                return jsonify({"error": "Engine reset but failed to retrieve state"}), 500

            app.logger.info("Engine successfully reset.")
            return jsonify({"message": "Game reset", "initial_state": state_dict}), 200
        except Exception as e:
            app.logger.error(f"Error during reset: {e}", exc_info=True)
            return jsonify({"error": f"Failed reset: {e}"}), 500

# --- Run the Server ---
if __name__ == '__main__':
    if not engine:
        print("FATAL: Engine failed init during script setup.", file=sys.stderr)
        sys.exit(1)
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask server (Sync AI) on port {port}...")
    # Ensure debug=False for production/Render
    app.run(debug=False, host='0.0.0.0', port=port, threaded=False) # threaded=False might be important with C extensions