import sys
import os
import traceback
import argparse
import time
import numpy as np
import h5py
from ctypes import *

# Need python-chess for the new feature extraction logic
try:
    import chess
except ImportError:
    print("Error: This script requires the 'python-chess' library.")
    print("Install it using: pip install python-chess")
    sys.exit(1)

# --- Import from the C++ wrapper ---
try:
    from chess_dir.ai_chess import ChessEngine, BoardState, BoardStateCType, BoardPosition, Piece, ChessMove
    print("Successfully imported components from chess package.")
except ImportError as e:
    print(f"Error importing from chess.ai_chess module: {e}")
    print("Ensure ai_chess.py exists inside the 'chess' subdirectory relative to this script,")
    print("and that the 'chess' directory contains an __init__.py file.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

# --- Constants for EmbeddingBag features ---
TOTAL_FEATURES = 768
FEATURES_PER_COLOR = TOTAL_FEATURES // 2
# Use python-chess constants directly
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
PIECE_TO_INDEX = {piece_type: i for i, piece_type in enumerate(PIECE_TYPES)}

# --- Feature Extraction (EmbeddingBag Indices) ---
def extract_features_indices(board_state: BoardState) -> list[int]:
    """
    Extracts feature indices for the EmbeddingBag NNUE input layer,
    adapted to work with the BoardState wrapper.

    Args:
        board_state: The BoardState object from the chess engine wrapper.

    Returns:
        A list of integer indices representing active features.
    """
    indices = []
    try:
        # Map C++ piece types/players to python-chess constants if necessary
        # C++: 0=None, 1=Pawn, ..., 6=King | 0=White, 1=Black
        # python-chess: 1=Pawn, ..., 6=King | True=White, False=Black

        # White pieces (offset 0 to FEATURES_PER_COLOR - 1)
        for r in range(8):
            for f in range(8):
                piece_c = board_state.pieces[r][f]
                if piece_c.piece_player == 0 and piece_c.piece_type != 0: # White piece
                    if 1 <= piece_c.piece_type <= 6:
                        piece_type_py = PIECE_TYPES[piece_c.piece_type - 1] # Map C++ type to python-chess type
                        piece_index = PIECE_TO_INDEX[piece_type_py]
                        square = r * 8 + f
                        base_index = piece_index * 64
                        feature_index = base_index + square
                        if 0 <= feature_index < FEATURES_PER_COLOR:
                            indices.append(feature_index)
                        else:
                             print(f"[WARN] White feature index out of bounds: {feature_index}")
                    else:
                         print(f"[WARN] Invalid white piece type {piece_c.piece_type}")


        # Black pieces (offset FEATURES_PER_COLOR to TOTAL_FEATURES - 1)
        black_offset = FEATURES_PER_COLOR
        for r in range(8):
            for f in range(8):
                 piece_c = board_state.pieces[r][f]
                 if piece_c.piece_player == 1 and piece_c.piece_type != 0: # Black piece
                    if 1 <= piece_c.piece_type <= 6:
                        piece_type_py = PIECE_TYPES[piece_c.piece_type - 1] # Map C++ type to python-chess type
                        piece_index = PIECE_TO_INDEX[piece_type_py]
                        square = r * 8 + f
                        base_index = piece_index * 64 + black_offset
                        feature_index = base_index + square
                        if FEATURES_PER_COLOR <= feature_index < TOTAL_FEATURES:
                             indices.append(feature_index)
                        else:
                             print(f"[WARN] Black feature index out of bounds: {feature_index}")
                    else:
                         print(f"[WARN] Invalid black piece type {piece_c.piece_type}")

    except IndexError:
        print("[ERROR] IndexError during feature extraction. Board state might be corrupt.")
        return []
    except Exception as e:
        print(f"[ERROR] Unexpected error during feature extraction: {e}")
        return []

    # Check for missing kings - essential for valid position
    white_king_found = any(p.piece_type == 6 and p.piece_player == 0 for row in board_state.pieces for p in row)
    black_king_found = any(p.piece_type == 6 and p.piece_player == 1 for row in board_state.pieces for p in row)
    if not white_king_found or not black_king_found:
        # print("[WARN] Position missing a king, returning empty features.")
        return [] # Invalid position

    return indices

# --- Game Outcome Determination (Adjusted) ---
def get_game_outcome(final_board_state: BoardState) -> float:
    """
    Determines the game outcome: 1.0 White Win, 0.5 Draw, 0.0 Black Win.

    Args:
        final_board_state: The BoardState object at the end of the game.

    Returns:
        1.0, 0.5, or 0.0.
    """
    status = final_board_state.status
    current_player = final_board_state.current_player # Player whose turn it WOULD be

    if status == 1: # Draw
        return 0.5
    elif status == 2: # Checkmate
        # Player whose turn it IS got checkmated.
        return 0.0 if current_player == 0 else 1.0 # 0.0 if White lost, 1.0 if Black lost
    elif status == 3: # Resigned
        # Player whose turn it IS resigned.
        return 0.0 if current_player == 0 else 1.0 # 0.0 if White lost, 1.0 if Black lost
    else:
        print(f"[WARN] Game ended with non-terminal status: {status}. Assuming draw.")
        return 0.5

# --- HDF5 Data Handling (Adjusted for VLEN) ---
def append_to_hdf5(h5_file, features_list_of_lists, outcomes_list):
    """Appends buffered data (feature index lists) to HDF5 datasets."""
    if not features_list_of_lists:
        return

    try:
        features_dataset = h5_file["features"]
        outcomes_dataset = h5_file["outcomes"]

        current_size = features_dataset.shape[0]
        new_size = current_size + len(features_list_of_lists)

        # Resize datasets
        features_dataset.resize(new_size, axis=0)
        outcomes_dataset.resize(new_size, axis=0)

        # --- FIX for VLEN: Convert list of lists to NumPy object array ---
        # Create an empty object array and fill it
        features_object_array = np.empty(len(features_list_of_lists), dtype=object)
        for i, lst in enumerate(features_list_of_lists):
             # Store each list as an element, ensure inner type is suitable (e.g., int64)
             features_object_array[i] = np.array(lst, dtype=np.int64)
        # --- End FIX ---

        outcomes_array = np.array(outcomes_list, dtype=np.float32).reshape(-1, 1)

        # Append data
        features_dataset[current_size:] = features_object_array # Write object array
        outcomes_dataset[current_size:] = outcomes_array

        h5_file.flush()
    except Exception as e:
        print(f"[ERROR] Failed to append data to HDF5 file: {e}")
        traceback.print_exc()


# --- Main Self-Play Function ---
def run_self_play(library_path: str, model_path: str, num_games: int, hdf5_path: str, ai_difficulty: int):
    """Runs self-play games and saves training data."""

    print(f"Starting self-play for {num_games} games.")
    print(f"Using library: {library_path}")
    print(f"Using model: {model_path}")
    print(f"AI Difficulty/Depth: {ai_difficulty}")
    print(f"Output HDF5: {hdf5_path}")

    games_processed = 0
    positions_saved = 0
    start_time_total = time.time()

    # Buffers for batch writing to HDF5
    feature_buffer = [] # Now stores lists of ints
    outcome_buffer = []
    BUFFER_SIZE = 10000

    try:
        # Define the HDF5 vlen integer type
        vlen_int_dtype = h5py.vlen_dtype(np.int64) # Use int64 for indices

        with h5py.File(hdf5_path, 'a') as h5f:
            # Create datasets if they don't exist, using vlen for features
            if "features" not in h5f:
                h5f.create_dataset("features", (0,), maxshape=(None,), # Shape is 1D for vlen
                                   dtype=vlen_int_dtype, # Use vlen type
                                   chunks=(1024,), # Chunks are 1D
                                   compression="gzip")
                print(f"Created 'features' dataset (vlen) in {hdf5_path}")
            if "outcomes" not in h5f:
                h5f.create_dataset("outcomes", (0, 1), maxshape=(None, 1),
                                   dtype='float32', chunks=(1024, 1), compression="gzip")
                print(f"Created 'outcomes' dataset in {hdf5_path}")

            initial_positions = h5f["features"].shape[0]
            print(f"HDF5 file initially contains {initial_positions} positions.")

            # --- Game Generation Loop ---
            for game_num in range(num_games):
                start_time_game = time.time()
                game_positions = [] # Store (BoardStateCType)

                try:
                    with ChessEngine(library_path, model_path) as engine:
                        while engine.board_state.status == 0:
                            current_state_copy = BoardStateCType()
                            memmove(byref(current_state_copy), byref(engine.board_state.board_state_impl), sizeof(BoardStateCType))
                            game_positions.append(current_state_copy)
                            engine.ai_move(ai_difficulty)

                        final_state_copy = BoardStateCType()
                        memmove(byref(final_state_copy), byref(engine.board_state.board_state_impl), sizeof(BoardStateCType))
                        outcome = get_game_outcome(BoardState(final_state_copy)) # Get 0.0, 0.5, or 1.0

                        if game_positions:
                             for state_c_data in game_positions:
                                 board_wrapper = BoardState(state_c_data)
                                 # --- Use new feature extraction ---
                                 features_indices = extract_features_indices(board_wrapper)
                                 # Only add if features were successfully extracted (e.g., kings present)
                                 if features_indices:
                                     feature_buffer.append(features_indices) # Append list of ints
                                     outcome_buffer.append(outcome)

                             positions_this_game = len(game_positions) # Count attempts, not necessarily saved positions
                             positions_saved_this_batch = len(outcome_buffer) - (positions_saved % BUFFER_SIZE) # How many new items in buffer
                             games_processed += 1
                             duration_game = time.time() - start_time_game

                             print(f"Game {game_num+1}/{num_games} finished. Outcome: {outcome:.1f}. Positions: {positions_this_game}. Time: {duration_game:.2f}s")

                             if len(feature_buffer) >= BUFFER_SIZE:
                                 print(f"Writing {len(feature_buffer)} positions to HDF5...")
                                 positions_saved += len(feature_buffer) # Update total count *before* clearing
                                 append_to_hdf5(h5f, feature_buffer, outcome_buffer)
                                 feature_buffer.clear()
                                 outcome_buffer.clear()
                        else:
                             print(f"Game {game_num+1}/{num_games} had no moves. Skipping.")

                except (RuntimeError, FileNotFoundError, OSError, AttributeError) as e:
                    print(f"[ERROR] Failed to initialize or run ChessEngine for game {game_num+1}: {e}")
                    traceback.print_exc()
                    break
                except Exception as e:
                    print(f"[ERROR] Unexpected error during game {game_num+1}: {e}")
                    traceback.print_exc()

            # Write remaining buffer
            if feature_buffer:
                print(f"Writing remaining {len(feature_buffer)} positions to HDF5...")
                positions_saved += len(feature_buffer)
                append_to_hdf5(h5f, feature_buffer, outcome_buffer)
                feature_buffer.clear()
                outcome_buffer.clear()

    except Exception as e:
        print(f"[FATAL ERROR] Error opening or writing to HDF5 file {hdf5_path}: {e}")
        traceback.print_exc()

    # --- Final Summary ---
    duration_total = time.time() - start_time_total
    print("\n--- Self-Play Summary ---")
    print(f"Games processed: {games_processed}/{num_games}")
    print(f"Total valid positions saved: {positions_saved}") # Reflects positions with valid features
    print(f"Total time: {duration_total:.2f} seconds")
    if games_processed > 0:
        print(f"Average time per game: {duration_total / games_processed:.2f} seconds")
    print(f"Data saved to: {hdf5_path}")


# --- Argument Parsing and Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate chess self-play data (EmbeddingBag format) and save to HDF5.")

    parser.add_argument("library_path", help="Path to the compiled C++ shared library (e.g., chess_cpp_pybind.so)")
    parser.add_argument("model_path", help="Path to the ONNX NNUE model file used by the engine.")
    parser.add_argument("hdf5_output_path", help="Path to the output HDF5 file (will be created or appended).")
    parser.add_argument("-n", "--num_games", type=int, default=100, help="Number of self-play games to generate.")
    parser.add_argument("-d", "--difficulty", type=int, default=3, help="AI difficulty/depth hint passed to the engine (Default: 3).")

    args = parser.parse_args()

    if not os.path.exists(args.library_path):
         print(f"Error: Shared library file not found at '{args.library_path}'")
         sys.exit(1)
    if not os.path.exists(args.model_path):
        print(f"Error: ONNX model file not found at '{args.model_path}'")
        sys.exit(1)

    run_self_play(
        library_path=args.library_path,
        model_path=args.model_path,
        num_games=args.num_games,
        hdf5_path=args.hdf5_output_path,
        ai_difficulty=args.difficulty
    )

    print("Self-play data generation finished.")
