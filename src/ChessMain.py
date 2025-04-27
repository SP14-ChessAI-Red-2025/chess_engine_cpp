# src/ChessMain.py (Modified for Cython Interface)
import pygame
from sys import exit, argv
from pygame import Color # Added Color import explicitly
import argparse
import queue
import threading
import time
import sys
import os
import traceback

# --- Path Setup ---
# Ensure the directory containing the compiled Cython module (chess_dir)
# is accessible. This might involve adding the parent directory or the
# directory containing 'chess_dir' to sys.path if running from 'src'.
script_dir_main = os.path.dirname(os.path.abspath(__file__))
# If ChessMain.py is in src and chess_dir is sibling to src:
project_root = os.path.abspath(os.path.join(script_dir_main, '..'))
# If chess_dir is inside src (where setup.py built it):
chess_dir_parent = script_dir_main # Or adjust as necessary

# Add the directory containing 'chess_dir' to the path
if chess_dir_parent not in sys.path:
    sys.path.append(chess_dir_parent)
# print(f"DEBUG: Added to sys.path: {chess_dir_parent}")
# print(f"DEBUG: Current sys.path: {sys.path}")


# --- Import the Cython module ---
try:
    # Import the Cython class and enums/wrappers
    # The module path 'chess_dir.ai_chess' comes from setup.py Extension name
    from chess_dir.ai_chess import (
        ChessEngine, Player, PieceType, MoveType, GameStatus,
        BoardPosition, ChessMove # Assuming these wrappers are defined in Cython
    )
    # print("DEBUG: Successfully imported from chess_dir.ai_chess")
except ImportError as e:
    print(f"CRITICAL ERROR importing Cython module 'chess_dir.ai_chess': {e}")
    print("Ensure the Cython extension was compiled correctly using:")
    print("  cd src && python setup.py build_ext --inplace") # Example build command
    print(f"Check sys.path: {sys.path}")
    exit(1)
except Exception as e:
    print(f"CRITICAL ERROR during import: {e}")
    print(traceback.format_exc())
    exit(1)

# Global constants and variables
BOARD_SIZE = 8
SQ_SIZE = 0
# Changed SELECTED_PIECE format: (piece_info_tuple or None, display_row, display_col)
# piece_info_tuple could be (type, player) or None
SELECTED_PIECE = (None, 0, 0)
icons = [] # Global list to hold piece images

# loads images (No changes needed if it uses PieceType enums correctly)
def load_images():
    """Loads piece icons and scales them based on calculated SQ_SIZE."""
    if SQ_SIZE == 0:
        print("ERROR: SQ_SIZE not set before loading images.")
        return False
    icons.clear()
    pcs = ["wht_pawn", "wht_knight", "wht_bishop", "wht_rook", "wht_queen", "wht_king",
           "blk_pawn", "blk_knight", "blk_bishop", "blk_rook", "blk_queen", "blk_king"]
    script_dir = os.path.dirname(__file__)
    image_folder = os.path.join(script_dir, "../images")
    # print(f"DEBUG: Loading images from: {image_folder}")
    loaded_successfully = True
    for pc in pcs:
        try:
            img_path = os.path.join(image_folder, pc + ".png")
            if not os.path.exists(img_path):
                print(f"Error: Image file not found at {img_path}")
                loaded_successfully = False
                continue
            loaded_image = pygame.image.load(img_path).convert_alpha()
            icons.append(pygame.transform.scale(loaded_image, (SQ_SIZE, SQ_SIZE)))
        except pygame.error as e:
            print(f"Error loading image {img_path}: {e}")
            loaded_successfully = False
        except Exception as e:
            print(f"Unexpected error loading image {img_path}: {e}")
            loaded_successfully = False
    if not loaded_successfully:
        print("ERROR: Failed to load one or more piece images.")
    return loaded_successfully

# draws game board pattern (No changes needed)
def draw_board(screen):
    """Draws the checkered board."""
    colors = [Color("antiquewhite"), Color("cadetblue")] # Use explicit Color
    for row in range(BOARD_SIZE):
        for column in range(BOARD_SIZE):
            color = colors[(row + column) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(column * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

# icon_for_piece (Updated to use imported enums)
def icon_for_piece(type_val, player_val):
    """Gets the correct icon from the loaded list based on piece type and color."""
    # Use imported enums for validation and indexing
    if not isinstance(type_val, int) or not (PieceType.PAWN <= type_val <= PieceType.KING):
        return None
    if not isinstance(player_val, int) or player_val not in [Player.WHITE, Player.BLACK]:
        return None
    # Calculate index: (type - 1) for 0-based index, add 6 if black (Player.BLACK == 1)
    index = (type_val - 1) + (6 if player_val == Player.BLACK else 0)
    if 0 <= index < len(icons):
        return icons[index]
    else:
        print(f"Error: Icon index {index} out of bounds (icons list length: {len(icons)}). Type: {type_val}, Player: {player_val}")
        return None

# target_for_move (Updated to use ChessMove wrapper properties)
def target_for_move(move: ChessMove) -> tuple[int, int] | None:
    """Calculates the target (internal rank, internal file) tuple for a move."""
    try:
        move_type = move.type # Access property from wrapper
        target_pos = move.target # Access property from wrapper (BoardPosition object)
        start_pos = move.start   # Access property from wrapper

        if move_type == MoveType.CASTLE:
            king_start_file = 4
            # Determine king's target square based on side
            king_target_file = 6 if target_pos.file > king_start_file else 2
            return (start_pos.rank, king_target_file) # King's final square
        elif move_type == MoveType.CLAIM_DRAW or move_type == MoveType.RESIGN:
             return None # No board target
        else:
            # Use the move's target position directly from the wrapper
            if 0 <= target_pos.rank < 8 and 0 <= target_pos.file < 8:
                 return (target_pos.rank, target_pos.file)
            else:
                 print(f"Warning: Invalid target position ({target_pos.rank},{target_pos.file}) in target_for_move.")
                 return None
    except AttributeError as e:
        print(f"Warning: Invalid move object or missing attributes passed to target_for_move: {move}, Error: {e}")
        return None
    except Exception as e:
         print(f"Error in target_for_move: {e}")
         return None

# moves_for_position (Updated to use ChessMove wrapper properties)
def moves_for_position(valid_moves, rank, file):
    """Filters a list of ChessMove objects to find those starting at the given internal rank/file."""
    if not isinstance(valid_moves, list): # Basic type check
        print("Warning: Invalid valid_moves list passed to moves_for_position.")
        return []
    # Filter moves where start_position matches the given rank and file
    matching_moves = []
    for move in valid_moves:
        # Check if it's a ChessMove object and access properties
        if isinstance(move, ChessMove) and \
           move.start.rank == rank and move.start.file == file:
            matching_moves.append(move)
    return matching_moves

# draw_pcs (Updated to use board_state_dict and new SELECTED_PIECE format)
def draw_pcs(screen, board_state_dict, valid_moves_for_player):
    """Draws the board squares, highlights, and pieces using the board state dictionary."""
    highlight_targets = []
    selected_piece_info, selected_display_row, selected_display_col = SELECTED_PIECE # Unpack new format

    if selected_piece_info is not None: # Check if piece_info tuple is valid
        selected_rank, selected_file = 7 - selected_display_row, selected_display_col
        piece_moves = moves_for_position(valid_moves_for_player, selected_rank, selected_file)
        try:
            highlight_targets = [target for move in piece_moves if (target := target_for_move(move)) is not None]
        except Exception as e:
            print(f"Error getting highlight targets: {e}")
            print(traceback.format_exc())
            highlight_targets = []

    # Draw pieces and highlights
    for row in range(BOARD_SIZE): # Display row (0=top)
        for column in range(BOARD_SIZE): # Display column (0=left)
            internal_rank = 7 - row
            internal_file = column
            try:
                # Access piece info dict from the board state dict
                pc_info = board_state_dict['pieces'][internal_rank][internal_file] # e.g., {'type': 1, 'player': 0}
            except (IndexError, KeyError, TypeError):
                 print(f"Error accessing board_state_dict['pieces'] at [{internal_rank}][{internal_file}].")
                 continue # Skip drawing this square

            # Highlight valid move squares
            if (internal_rank, internal_file) in highlight_targets:
                highlight_color = Color("lightseagreen") # Use explicit Color
                center_x = column * SQ_SIZE + SQ_SIZE / 2
                center_y = row * SQ_SIZE + SQ_SIZE / 2
                pygame.draw.circle(screen, highlight_color, center=(center_x, center_y), radius=SQ_SIZE / 4, width=0)

            # Draw the piece itself
            if pc_info['type'] != PieceType.NONE: # Use imported enum
                icon = icon_for_piece(pc_info['type'], pc_info['player'])
                if icon is None: continue

                # Check if this is the currently selected piece
                is_selected = (selected_piece_info is not None and
                               selected_display_row == row and
                               selected_display_col == column)

                if is_selected:
                     mouse_pos = pygame.mouse.get_pos()
                     icon_rect = icon.get_rect(center=mouse_pos)
                     screen.blit(icon, icon_rect)
                else:
                     icon_rect = pygame.Rect(SQ_SIZE * column, SQ_SIZE * row, SQ_SIZE, SQ_SIZE)
                     screen.blit(icon, icon_rect)


# AI thread function (Updated to use engine object)
def run_ai_move(engine, difficulty, result_queue):
    """Runs the AI move calculation in a separate thread using the engine object."""
    try:
        print("AI thinking...")
        start_time = time.perf_counter()
        # Direct call to the Cython object's method
        engine.ai_move(difficulty) # Modifies engine's state internally
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"AI move found in {duration:.2f} seconds.")
        # The result (new board state) is fetched by reading engine.board_state
        # *after* the thread completes in the main loop.
        result_queue.put(("done", duration))
    except Exception as e:
        print(f"[ERROR] Exception in AI thread: {e}\n{traceback.format_exc()}")
        result_queue.put(("error", 0))


# initializes application window/provides game loop (Main modifications here)
def main():
    """Main game function: initialization, game loop, event handling."""
    parser = argparse.ArgumentParser(description="Chess AI Game")
    # Library path is now mainly informational if setup.py linked correctly
    parser.add_argument("library_path", type=str, help="Path to the C++ chess engine library (informational).")
    parser.add_argument("model_path", type=str, help="Path to the NNUE model file (e.g., model/trained_nnue.onnx)")
    args = parser.parse_args()

    pygame.init()
    screen = None
    screenWidth = 0
    screenHeight = 0

    # --- Main Try/Except/Finally Block ---
    engine = None # Initialize engine variable outside the try block
    try:
        # --- Screen Setup ---
        try:
            monitorInfo = pygame.display.Info()
            max_dim = 1000
            screenHeight = min(monitorInfo.current_h / 1.25, max_dim)
            screenWidth = screenHeight
        except pygame.error:
            print("Warning: Could not get display info. Using default size.")
            screenWidth = screenHeight = 640

        global SQ_SIZE
        SQ_SIZE = int(screenHeight / BOARD_SIZE)
        if SQ_SIZE == 0: raise ValueError("Calculated SQ_SIZE is 0.")

        screen = pygame.display.set_mode((int(screenWidth), int(screenHeight)))
        pygame.display.set_caption("ChessAI")

        # --- Load Window Icon ---
        try:
            script_dir = os.path.dirname(__file__)
            icon_path = os.path.join(script_dir, "../images/icon.png")
            if os.path.exists(icon_path):
                pygame.display.set_icon(pygame.image.load(icon_path))
            else: print(f"Warning: Window icon not found at {icon_path}")
        except pygame.error as e: print(f"Warning: Could not load window icon: {e}")

        # --- Load Piece Images ---
        if not load_images():
            raise RuntimeError("Failed to load piece images.")

        # --- Select game mode ---
        print("Select game mode:")
        print("1. AI vs AI")
        print("2. Player vs AI (Player is White)")
        print("3. Player vs AI (Player is Black)")
        mode = ""
        while mode not in ["1", "2", "3"]:
            mode = input("Enter 1, 2, or 3: ").strip()

        player_turn = -1 # Represents AI/No specific player
        if mode == '2': player_turn = Player.WHITE
        elif mode == '3': player_turn = Player.BLACK

        # --- Instantiate the Cython Engine ---
        # Use a context manager ('with') for automatic cleanup via __exit__/__dealloc__
        with ChessEngine(args.library_path, args.model_path) as engine:
            # print("DEBUG: Cython ChessEngine initialized.")

            # --- Get Initial State from Engine ---
            board_state_dict = engine.board_state # Access the property
            if not board_state_dict:
                 raise RuntimeError("Failed to get initial board state dictionary from engine.")
            # print(f"DEBUG: Initial player from state dict: {board_state_dict.get('current_player', 'N/A')}")

            all_legal_moves_for_player = engine.get_valid_moves() # Call the method
            # print(f"DEBUG: Initial valid moves count: {len(all_legal_moves_for_player)}")
            # --- END OF INITIAL STATE ---

            global SELECTED_PIECE # Use the global for UI selection state
            SELECTED_PIECE = (None, 0, 0) # Reset selection state

            ai_thread = None
            ai_thinking = False
            ai_result_queue = queue.Queue()
            game_over = False
            # Initial status message based on fetched state
            status_message = getGameStatusMessage(board_state_dict, ai_thinking)

            # --- Main Game Loop ---
            while True:
                # Update current player from the potentially modified board_state_dict
                current_player_internal = board_state_dict.get('current_player', -1) # Default to -1 if key missing

                # --- Event Handling ---
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        if ai_thinking and ai_thread and ai_thread.is_alive():
                            print("QUIT received, attempting to cancel AI search...")
                            engine.cancel_search() # Use engine object method
                            ai_thread.join(timeout=1.0)
                        raise SystemExit("Pygame Quit event received")

                    # Handle mouse clicks only if it's the human player's turn and game not over
                    elif event.type == pygame.MOUSEBUTTONDOWN and \
                        mode in ["2", "3"] and \
                        current_player_internal == player_turn and \
                        not ai_thinking and not game_over:

                        position = pygame.mouse.get_pos()
                        column = int(position[0] // SQ_SIZE)
                        row = int(position[1] // SQ_SIZE)
                        clicked_rank = 7 - row
                        clicked_file = column
                        # print(f"DEBUG Click: Display=({row},{column}), Internal=({clicked_rank},{clicked_file})")

                        # --- Click Logic (Adapted for Cython) ---
                        selected_piece_info, selected_display_row, selected_display_col = SELECTED_PIECE

                        if selected_piece_info is not None: # A piece is currently selected
                             # print(f"DEBUG Click: Piece previously selected at Display=({selected_display_row},{selected_display_col})")
                             if (row, column) == (selected_display_row, selected_display_col):
                                 # print("DEBUG Click: Clicked same square, deselecting.")
                                 SELECTED_PIECE = (None, 0, 0)
                             else:
                                 selected_rank, selected_file = 7 - selected_display_row, selected_display_col
                                 # print(f"DEBUG Click: Trying to move from Internal=({selected_rank},{selected_file}) to Internal=({clicked_rank},{clicked_file})")
                                 # Find the corresponding Cython ChessMove object
                                 move_to_apply = None
                                 for move_obj in all_legal_moves_for_player:
                                      # Access attributes via the ChessMove wrapper class properties
                                      if isinstance(move_obj, ChessMove) and \
                                         move_obj.start.rank == selected_rank and move_obj.start.file == selected_file and \
                                         move_obj.target.rank == clicked_rank and move_obj.target.file == clicked_file:
                                           move_to_apply = move_obj
                                           # print(f"DEBUG Click: Found matching move object: {move_obj}")
                                           break

                                 if move_to_apply:
                                     try:
                                         move_str_log = engine.move_to_str(move_to_apply) # Use engine method
                                         print(f"Applying move: {move_str_log}")
                                         ai_thinking = True # Block input during apply
                                         status_message = "Applying move..."
                                         # Call Cython method - it modifies internal state and returns the new state dict
                                         new_board_state_dict = engine.apply_move(move_to_apply)
                                         ai_thinking = False # Unblock

                                         # --- Update local state from returned dict ---
                                         board_state_dict = new_board_state_dict
                                         all_legal_moves_for_player = engine.get_valid_moves() # Get moves for next player
                                         SELECTED_PIECE = (None, 0, 0) # Deselect after move

                                         # Update status and check game end
                                         status_message = getGameStatusMessage(board_state_dict, ai_thinking)
                                         current_status = board_state_dict['status']
                                         game_over = (current_status != GameStatus.NORMAL)
                                         if game_over: print(f"Game Over! Status: {status_message}")
                                         elif board_state_dict['in_check'][board_state_dict['current_player']]: print("Check!")
                                         # --- End Status Check ---

                                     except Exception as e_apply:
                                         print(f"Error applying move {move_str_log}: {e_apply}")
                                         print(traceback.format_exc())
                                         # Attempt to resync state
                                         board_state_dict = engine.board_state
                                         all_legal_moves_for_player = engine.get_valid_moves()
                                         status_message = f"Error: {e_apply}"
                                         SELECTED_PIECE = (None, 0, 0)
                                         ai_thinking = False # Ensure unblocked on error

                                 else: # Clicked different square, not a valid move target
                                     # print("DEBUG Click: Clicked square is not a valid target.")
                                     try:
                                         clicked_piece_dict = board_state_dict['pieces'][clicked_rank][clicked_file]
                                         # Check if clicked on another piece of the player's color
                                         if clicked_piece_dict['type'] != PieceType.NONE and clicked_piece_dict['player'] == player_turn:
                                             # print(f"DEBUG Click: Selecting new piece at Display=({row},{column})")
                                             # Store piece info (type, player) tuple
                                             SELECTED_PIECE = ((clicked_piece_dict['type'], clicked_piece_dict['player']), row, column)
                                         else:
                                             # print("DEBUG Click: Clicked empty square or opponent piece, deselecting.")
                                             SELECTED_PIECE = (None, 0, 0)
                                     except (IndexError, KeyError):
                                          # print("DEBUG Click: Error accessing clicked piece or out of bounds, deselecting.")
                                          SELECTED_PIECE = (None, 0, 0)

                        else: # No piece was selected...
                             # print("DEBUG Click: No piece previously selected.")
                             try:
                                 clicked_piece_dict = board_state_dict['pieces'][clicked_rank][clicked_file]
                                 if clicked_piece_dict['type'] != PieceType.NONE and clicked_piece_dict['player'] == player_turn:
                                     # print(f"DEBUG Click: Selecting piece at Display=({row},{column})")
                                     SELECTED_PIECE = ((clicked_piece_dict['type'], clicked_piece_dict['player']), row, column)
                                 else:
                                     # print("DEBUG Click: Clicked empty square or opponent piece.")
                                     SELECTED_PIECE = (None, 0, 0)
                             except (IndexError, KeyError):
                                  # print("DEBUG Click: Clicked out of bounds.")
                                  SELECTED_PIECE = (None, 0, 0)
                         # --- End of Click Logic ---

                # --- AI Turn Logic ---
                is_ai_turn = not game_over and (
                            (mode == "1") or
                            (mode == "2" and current_player_internal == Player.BLACK) or
                            (mode == "3" and current_player_internal == Player.WHITE)
                            )

                if is_ai_turn and not ai_thinking:
                    print(f"Starting AI thread for player {current_player_internal}")
                    while not ai_result_queue.empty():
                        try: ai_result_queue.get_nowait()
                        except queue.Empty: break
                    # Start AI thread using the 'engine' object
                    ai_thread = threading.Thread(target=run_ai_move, args=(engine, 5, ai_result_queue), daemon=True) # Difficulty 5
                    ai_thread.start()
                    ai_thinking = True
                    status_message = "AI is thinking..." # Update status display

                # Check AI result queue
                if ai_thinking:
                    try:
                        result, _ = ai_result_queue.get_nowait() # Check if thread put something in queue
                        ai_thinking = False # Thread finished (successfully or with error)
                        ai_thread = None    # Clear thread reference
                        if result == "done":
                            print("AI thread finished. Fetching updated state...")
                            # AI move modifies internal engine state, get the new state dict
                            board_state_dict = engine.board_state
                            all_legal_moves_for_player = engine.get_valid_moves() # Get moves for next player
                            status_message = getGameStatusMessage(board_state_dict, ai_thinking) # Update status message

                            # Check game status
                            current_status = board_state_dict['status']
                            game_over = (current_status != GameStatus.NORMAL)
                            if game_over: print(f"Game Over! Status: {status_message}")
                            elif board_state_dict['in_check'][board_state_dict['current_player']]: print("Check!")
                            # --- End Status Check ---

                        elif result == "error":
                            game_over = True; status_message = "AI Error"
                            print("AI thread reported an error.")

                    except queue.Empty:
                        pass # AI still thinking, continue loop
                    except Exception as q_e:
                        print(f"Error processing AI queue: {q_e}")
                        print(traceback.format_exc())
                        ai_thinking = False # Stop waiting if queue error
                        ai_thread = None
                        game_over = True; status_message = "System Error"


                # --- Drawing ---
                if screen:
                    draw_board(screen)
                    if board_state_dict:
                        # Pass the dict and moves list to draw_pcs
                        draw_pcs(screen, board_state_dict, all_legal_moves_for_player)
                    else:
                        print("Warning: board_state_dict invalid for drawing.")

                    # Display game over message
                    if game_over and status_message:
                        try:
                            font = pygame.font.SysFont(None, 48)
                            text_surface = font.render(status_message, True, Color('red')) # Use explicit Color
                            text_rect = text_surface.get_rect(center=(int(screenWidth / 2), int(screenHeight / 2)))
                            bg_surface = pygame.Surface(text_rect.size, pygame.SRCALPHA)
                            bg_surface.fill((200, 200, 200, 180))
                            screen.blit(bg_surface, text_rect.topleft)
                            screen.blit(text_surface, text_rect)
                        except Exception as font_e:
                            print(f"Error displaying game over message: {font_e}")

                    pygame.display.update()
                # --- End of Game Loop ---

        # --- End of 'with ChessEngine' block --- (Engine cleanup handled automatically)

    except SystemExit as e:
        print(f"Exiting gracefully: {e}")
    except ImportError as e: # Catch import errors specifically
        print(f"Import Error: {e}")
        print(traceback.format_exc())
    except RuntimeError as e: # Catch runtime errors like engine init failure
        print(f"Runtime Error: {e}")
        print(traceback.format_exc())
    except ValueError as e:
        print(f"Value Error: {e}")
        print(traceback.format_exc())
    except Exception as e: # Catch any other unexpected errors
        print(f"\n--- An unexpected error occurred in main ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print(traceback.format_exc())
        print("---------------------------------------------")
    finally:
        print("Exiting Pygame.")
        # Ensure AI thread is handled if loop exited unexpectedly
        if 'engine' in locals() and engine is not None and ai_thread and ai_thread.is_alive():
             print("Attempting final AI search cancellation...")
             try: engine.cancel_search()
             except Exception as cancel_e: print(f"Error during final cancel: {cancel_e}")
             ai_thread.join(timeout=0.5)
        pygame.quit()

# Function to get status message (Handles None state)
def getGameStatusMessage(state_dict, is_thinking):
    if is_thinking: return "AI is thinking..." # Override if AI is active
    if not state_dict: return "Loading State..."

    status = state_dict.get('status', GameStatus.NORMAL)
    current_player = state_dict.get('current_player', -1)
    in_check = state_dict.get('in_check', [False, False])

    if status == GameStatus.NORMAL:
        player_name = "White" if current_player == Player.WHITE else "Black" if current_player == Player.BLACK else "N/A"
        check_str = " (Check!)" if current_player != -1 and in_check[current_player] else ""
        return f"{player_name}'s Turn{check_str}"
    elif status == GameStatus.CHECKMATE: return "Checkmate!"
    elif status == GameStatus.DRAW: return "Draw (Stalemate/50-Move/Insufficient Material)"
    elif status == GameStatus.DRAW_BY_REPETITION: return "Draw (Repetition)"
    elif status == GameStatus.RESIGNED: return "Resigned"
    else: return f"Unknown Status ({status})"


# Entry point
if __name__ == '__main__':
    if len(argv) < 3:
        print("Usage: python src/ChessMain.py <path_to_libchess_cpp.(so|dylib|dll)> <path_to_trained_nnue.onnx>")
        # Note: The library path is less critical now if linking works in setup.py
        exit()
    main()