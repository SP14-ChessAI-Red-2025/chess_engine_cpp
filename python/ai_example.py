# Example script to run the Chess Engine with AI moves powered by an ONNX model,
# using the modified ai_chess.py wrapper located inside the 'chess_dir' package directory.

import sys
import os
import glob
import platform
import traceback
import argparse
import time
import chess_dir.ai_chess as ai
from ctypes import *

MODEL_FILENAME = "trained_nnue.onnx"
LIBRARY_PATTERN = "chess_cpp_pybind*.{ext}"
DEFAULT_AI_DIFFICULTY = 3

piece_strings = [["♙", "♘", "♗", "♖", "♕", "♔"], ["♟", "♞", "♝", "♜", "♛", "♚"]]

def find_project_paths():
    """
    Attempts to automatically find the shared library and model path
    based on the script's location and typical project structure.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    build_src_dir = os.path.join(project_root, "build", "src")
    library_path = None
    system = platform.system()
    if system == "Windows":
        search_pattern = os.path.join(build_src_dir, LIBRARY_PATTERN.format(ext="pyd"))
        found_libs = glob.glob(search_pattern)
        if not found_libs:
             search_pattern = os.path.join(build_src_dir, LIBRARY_PATTERN.format(ext="dll"))
             found_libs = glob.glob(search_pattern)
    elif system == "Darwin":
        search_pattern = os.path.join(build_src_dir, LIBRARY_PATTERN.format(ext="dylib"))
        found_libs = glob.glob(search_pattern)
    else: # Linux and other Unix-like
        search_pattern = os.path.join(build_src_dir, LIBRARY_PATTERN.format(ext="so"))
        found_libs = glob.glob(search_pattern)

    if found_libs:
        found_libs.sort(key=len, reverse=True)
        library_path = found_libs[0]
        print(f"Found library: {library_path}")
    else:
        print(f"Warning: Could not automatically find library in '{build_src_dir}' matching pattern '{LIBRARY_PATTERN}'.")
        print("Please ensure the project is built and the library exists.")

    model_dir = os.path.join(project_root, "model")
    default_model_path = os.path.join(model_dir, MODEL_FILENAME)
    if os.path.exists(default_model_path):
        print(f"Found default model: {default_model_path}")
    else:
         print(f"Warning: Could not find default model at '{default_model_path}'.")
         default_model_path = None

    return library_path, default_model_path


def print_board(board: ai.BoardState) -> None:
    """Prints the current board state to the console."""
    board_str = "  a b c d e f g h\n"
    board_str += " +---------------+\n"
    for rank_idx in range(7, -1, -1):
        board_str += f"{rank_idx + 1}|"
        rank = board.pieces[rank_idx]
        for piece in rank:
            if piece.piece_type == 0: board_str += '. '
            elif 0 < piece.piece_type <= 6: board_str += piece_strings[piece.piece_player][piece.piece_type - 1] + ' '
            else: board_str += '? '
        board_str += f"|{rank_idx + 1}\n"
    board_str += " +---------------+\n"
    board_str += "  a b c d e f g h\n"
    print(board_str)

def main() -> None:
    """Main function to run the player vs AI game."""

    parser = argparse.ArgumentParser(description="Run Player vs AI Chess Game.")
    parser.add_argument(
        "model_path", nargs='?', default=None,
        help=f"Optional. Path to the ONNX model file. Defaults to searching in '../model/{MODEL_FILENAME}'."
    )
    parser.add_argument(
        "-d", "--difficulty", type=int, default=DEFAULT_AI_DIFFICULTY,
        help=f"AI difficulty level (maps to depth in C++). Default: {DEFAULT_AI_DIFFICULTY}"
    )
    parser.add_argument(
        "--book_path", type=str, default="../model/Human.bin",
        help="Path to the Polyglot opening book (.bin file)."
    )
    args = parser.parse_args()
    ai_difficulty = args.difficulty

    library_path, default_model_path = find_project_paths()
    model_path = args.model_path
    if model_path:
        if not os.path.exists(model_path):
            print(f"Warning: Model path from command line not found: '{model_path}'. Falling back.")
            model_path = None
    if model_path is None: model_path = default_model_path
    if model_path is None:
        print(f"Error: Could not find model '{MODEL_FILENAME}' and no valid path provided.")
        return
    if library_path is None:
        print("Error: Could not automatically find the chess_dir engine library. Ensure project is built.")
        return

    print(f"\nLoading Chess Engine from: {library_path}")
    print(f"Using ONNX model from: {model_path}")
    print(f"AI Difficulty set to: {ai_difficulty}")

    try:
        with ai.ChessEngine(library_path, model_path) as chess_engine:
            print("Chess Engine and AI Initialized.")
            while True:
                print("-" * 30)
                current_player = chess_engine.board_state.current_player
                current_player_str = "White" if current_player == 0 else "Black"
                print(f"{current_player_str}'s turn (Difficulty: {ai_difficulty}):")
                print_board(chess_engine.board_state)

                in_check_status = chess_engine.board_state.in_check
                if in_check_status[current_player]: print(f"*** {current_player_str} is in Check! ***")
                status = chess_engine.board_state.status
                status_str = ["Normal", "Draw", "Checkmate", "Resigned"][status]
                print(f"Game Status: {status_str}")
                if status != 0:
                    if status == 1: print("\nGame over: Draw")
                    elif status == 2: winner = "Black" if current_player == 0 else "White"; print(f"\nGame over: Checkmate! {winner} wins.")
                    elif status == 3: winner = "Black" if current_player == 0 else "White"; print(f"\nGame over: Resignation! {winner} wins.")
                    break

                if current_player == 0: # Player's Turn (White)
                    valid_moves = []
                    try: valid_moves = chess_engine.get_valid_moves()
                    except RuntimeError as e: print(f"Error getting valid moves: {e}"); break
                    if not valid_moves: print("No valid moves available, ending game."); break
                    print("\nValid moves:")
                    move_dict = {}
                    for idx, move in enumerate(valid_moves): move_str = chess_engine.move_to_str(move); print(f"  {idx+1}: {move_str}"); move_dict[idx+1] = move
                    while True:
                        try:
                            move_number_str = input("Enter move number (or 'q' to quit): ")
                            if move_number_str.lower() == 'q': print("Quitting game."); return
                            move_number = int(move_number_str)
                            if move_number in move_dict:
                                chosen_move = move_dict[move_number]
                                if chosen_move.type == 6: print("Player resigns."); chess_engine.apply_move(chosen_move); break
                                print(f"\nApplying move: {chess_engine.move_to_str(chosen_move)}"); chess_engine.apply_move(chosen_move); break
                            else: print("Invalid move number.")
                        except ValueError: print("Invalid input.")
                        except Exception as e: print(f"Error applying move: {e}"); traceback.print_exc(); break

                else: # AI's Turn (Black)
                    print("\nAI is thinking...")
                    try:
                        start_time = time.perf_counter()
                        chess_engine.ai_move(ai_difficulty)
                        end_time = time.perf_counter()
                        duration = end_time - start_time

                        if ai_difficulty <= 1: current_depth = 2
                        elif ai_difficulty <= 2: current_depth = 3
                        else: current_depth = 4
                        print(f"AI moved (Difficulty {ai_difficulty}, Depth {current_depth}). Took {duration:.2f}s wall-clock time.")

                    except RuntimeError as e: print(f"Runtime error during AI move: {e}"); break
                    except Exception as e: print(f"Unexpected error during AI move: {e}"); traceback.print_exc(); break

    except FileNotFoundError as e: print(f"Initialization Error: {e}")
    except OSError as e: print(f"Initialization Error: Could not load library: {e}")
    except RuntimeError as e:
        print(f"Initialization Error: {e}")
        if "libcudnn" in str(e): print("\nCheck CUDA/cuDNN installation and library paths (LD_LIBRARY_PATH/PATH).")
    except AttributeError as e: print(f"Initialization Error: Library function mismatch: {e}. Ensure C++ library is compiled correctly.")
    except Exception as e: print(f"\nAn unexpected error occurred: {e}"); traceback.print_exc()

if __name__ == "__main__":
    main()