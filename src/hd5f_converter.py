import chess
import chess.pgn
import chess.engine
import numpy as np
import zstandard as zstd
import io
import argparse
import os
from tqdm import tqdm
import traceback
import multiprocessing
import time
import sys
import math
import gc
import signal

# Global flag to indicate termination
terminate_flag = False

def handle_termination_signal(signum, frame):
    global terminate_flag
    print("\n[INFO] Termination signal received. Finalizing and exiting...")
    terminate_flag = True

try:
    import h5py
    print(f"Using h5py version: {h5py.__version__}")
    sys.stdout.flush()
except ImportError:
    print("Error: h5py library not found.")
    print("Please install it using: pip install h5py")
    sys.exit(1)

try:
    print(f"Using python-chess version: {chess.__version__}")
    sys.stdout.flush()
except AttributeError:
    print("Could not determine python-chess version automatically.")
    sys.stdout.flush()


PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
PIECE_TO_INDEX = {piece_type: i for i, piece_type in enumerate(PIECE_TYPES)}
TOTAL_FEATURES = 768
FEATURES_PER_COLOR = TOTAL_FEATURES // 2
CHUNK_SIZE = 25000  # Default, can be overridden by args

# --- Feature Extraction ---
def extract_features(board: chess.Board) -> list[int]:
    """
    Extracts feature indices for a simple 768-feature NNUE input layer.
    """
    indices = []
    for piece_type in PIECE_TYPES:
        piece_index = PIECE_TO_INDEX[piece_type]
        base_index = piece_index * 64
        for square in board.pieces(piece_type, chess.WHITE):
            feature_index = base_index + square
            if 0 <= feature_index < FEATURES_PER_COLOR:
                indices.append(feature_index)

    black_offset = FEATURES_PER_COLOR
    for piece_type in PIECE_TYPES:
        piece_index = PIECE_TO_INDEX[piece_type]
        base_index = piece_index * 64 + black_offset
        for square in board.pieces(piece_type, chess.BLACK):
            feature_index = base_index + square
            if FEATURES_PER_COLOR <= feature_index < TOTAL_FEATURES:
                indices.append(feature_index)

    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
        return []

    indices.sort()  # Optional: Ensure consistent order
    return indices

# --- Stockfish Evaluation ---
def evaluate_position(board: chess.Board, engine_path: str = "stockfish") -> float:
    """
    Evaluate a chess position using Stockfish.
    :param board: A chess.Board object representing the position.
    :param engine_path: Path to the Stockfish executable.
    :return: Evaluation score in centipawns (positive for White, negative for Black).
    """
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        result = engine.analyse(board, chess.engine.Limit(depth=4))
        score = result["score"].white().score(mate_score=10000)  # Mate scores are capped
        return score if score is not None else 0  # Return 0 if evaluation is unavailable

def evaluate_positions(boards: list[chess.Board], engine_path: str = "stockfish", depth: int = 4) -> list[float]:
    """
    Evaluate multiple chess positions using Stockfish in a batch.
    :param boards: A list of chess.Board objects representing the positions.
    :param engine_path: Path to the Stockfish executable.
    :param depth: Depth for Stockfish evaluation.
    :return: A list of evaluation scores in centipawns.
    """
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        results = []
        for board in boards:
            try:
                result = engine.analyse(board, chess.engine.Limit(depth=depth))
                score = result["score"].white().score(mate_score=10000)  # Mate scores are capped
                results.append(score if score is not None else 0)
            except Exception as e:
                print(f"[ERROR] Failed to evaluate position: {e}", file=sys.stderr)
                results.append(0)  # Default to 0 if evaluation fails
        return results

def write_hdf5_chunk(hf, features_chunk, outcomes_chunk, plies_chunk, first_chunk_in_file):
    """
    Writes a chunk of data to the HDF5 file. Creates datasets if it's the first chunk.
    """
    count = len(outcomes_chunk)
    if count == 0:
        return 0

    try:
        # Convert list of lists to object array containing int32 numpy arrays
        np_features_list = [np.array(lst, dtype=np.int32) for lst in features_chunk]
        np_features_obj = np.array(np_features_list, dtype=object)
    except Exception as e:
         pid = os.getpid()
         print(f"\n[PID {pid}] Error converting feature chunk lists to NumPy arrays: {e}", file=sys.stderr)
         traceback.print_exc(file=sys.stderr)
         return 0

    np_outcomes = np.array(outcomes_chunk, dtype=np.float32)
    np_plies = np.array(plies_chunk, dtype=np.uint16)

    try:
        vlen_dtype = h5py.special_dtype(vlen=np.int32) # VLEN for features

        if first_chunk_in_file:
            hf.create_dataset('features', data=np_features_obj, dtype=vlen_dtype,
                              maxshape=(None,), chunks=(max(1, CHUNK_SIZE // 10),),
                              compression="gzip")
            hf.create_dataset('outcomes', data=np_outcomes, maxshape=(None,),
                              chunks=(CHUNK_SIZE,), compression="gzip")
            hf.create_dataset('plies', data=np_plies, maxshape=(None,),
                              chunks=(CHUNK_SIZE,), compression="gzip")
        else:
            # Append features
            features_dset = hf['features']
            features_dset.resize((features_dset.shape[0] + count,))
            features_dset[-count:] = np_features_obj
            # Append outcomes
            outcomes_dset = hf['outcomes']
            outcomes_dset.resize((outcomes_dset.shape[0] + count,))
            outcomes_dset[-count:] = np_outcomes
            # Append plies
            plies_dset = hf['plies']
            plies_dset.resize((plies_dset.shape[0] + count,))
            plies_dset[-count:] = np_plies

        return count
    except Exception as e:
        pid = os.getpid()
        print(f"\n[PID {pid}] Error writing chunk to HDF5 file '{hf.filename}': {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 0


# --- PGN Processing (Worker Task) ---
def process_pgn_file_worker(input_args):
    """
    Reads a single PGN/ZST file, extracts features/outcomes,
    evaluates positions using Stockfish, and saves incrementally to an HDF5 file.
    """
    pgn_path, output_hdf5_path, min_ply, engine_path = input_args
    worker_pid = os.getpid()
    base_pgn_name = os.path.basename(pgn_path)
    base_hdf5_name = os.path.basename(output_hdf5_path)
    tqdm.write(f"[PID {worker_pid}] Processing {base_pgn_name} -> {base_hdf5_name}...")
    sys.stdout.flush()
    start_time = time.time()

    total_positions_written = 0
    processed_game_count_in_file = 0
    first_chunk_in_file = True

    try:
        os.makedirs(os.path.dirname(output_hdf5_path), exist_ok=True)
    except OSError as e:
        print(f"\n[PID {worker_pid}] Error creating output directory for {base_hdf5_name}: {e}", file=sys.stderr)
        return False

    try:
        with h5py.File(output_hdf5_path, 'w') as hf:
            def hdf5_write_callback(features_chunk, outcomes_chunk, plies_chunk):
                nonlocal total_positions_written, first_chunk_in_file
                written_count = write_hdf5_chunk(hf, features_chunk, outcomes_chunk, plies_chunk, first_chunk_in_file)
                total_positions_written += written_count
                if written_count > 0:
                    first_chunk_in_file = False

            features_current_chunk = []
            outcomes_current_chunk = []
            plies_current_chunk = []

            is_compressed = pgn_path.lower().endswith(".zst")
            compressed_file_handle = None

            try:
                stream_opener = None
                if is_compressed:
                    dctx = zstd.ZstdDecompressor()
                    compressed_file_handle = open(pgn_path, 'rb')
                    stream_opener = dctx.stream_reader(compressed_file_handle)
                else:
                    stream_opener = open(pgn_path, 'rt', encoding='utf-8', errors='replace')

                with stream_opener as reader:
                    pgn_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace') if is_compressed else reader
                    processed_game_count_in_file = _process_stream_chunked(
                        pgn_stream,
                        features_current_chunk,
                        outcomes_current_chunk,
                        plies_current_chunk,
                        min_ply,
                        worker_pid,
                        base_pgn_name,
                        CHUNK_SIZE,
                        hdf5_write_callback
                    )
                    # Write final remaining chunk
                    hdf5_write_callback(features_current_chunk, outcomes_current_chunk, plies_current_chunk)


            except (MemoryError, ValueError, KeyError, IndexError, chess.IllegalMoveError, chess.InvalidMoveError, AttributeError) as e:
                 print(f"\n[PID {worker_pid}] Error during PGN stream processing for {base_pgn_name}: {e}", file=sys.stderr)
                 traceback.print_exc(file=sys.stderr)
                 # Continue, partial file might be useful
            except Exception as e:
                print(f"\n[PID {worker_pid}] Critical Error processing file {base_pgn_name}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                return False # Indicate failure
            finally:
                if compressed_file_handle and not compressed_file_handle.closed:
                    compressed_file_handle.close()
                del features_current_chunk, outcomes_current_chunk, plies_current_chunk
                gc.collect()

    except Exception as e:
        print(f"\n[PID {worker_pid}] Critical HDF5 file error for {base_hdf5_name}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False

    if total_positions_written == 0:
        tqdm.write(f"[PID {worker_pid}] No valid positions extracted from {base_pgn_name}. HDF5 file created: {base_hdf5_name}")
        sys.stdout.flush()

    duration = time.time() - start_time
    tqdm.write(f"[PID {worker_pid}] Finished {base_pgn_name}. Extracted {total_positions_written} positions from {processed_game_count_in_file} valid games in {duration:.2f}s. Saved to {base_hdf5_name}")
    sys.stdout.flush()
    return True


def _process_stream_chunked(pgn_stream, features_chunk, outcomes_chunk, plies_chunk, min_ply, worker_pid, filename, chunk_size, write_callback):
    global terminate_flag
    game_count = 0
    position_count_total = 0
    position_in_current_chunk = 0
    skipped_games = 0
    processed_games_count = 0
    error_count = 0

    desc = f"[PID {worker_pid}] File {filename}"
    pbar = tqdm(desc=desc, unit=" game", leave=False, position=worker_pid % 30, mininterval=5.0, smoothing=0.1)

    while not terminate_flag:
        game = None
        try:
            game = chess.pgn.read_game(pgn_stream)
            if game is None:
                break
            game_count += 1
            pbar.update(1)

            result = game.headers.get("Result", "*")
            if result == "1-0":
                outcome = 1.0
            elif result == "0-1":
                outcome = 0.0
            elif result == "1/2-1/2":
                outcome = 0.5
            else:
                skipped_games += 1
                continue

            for node in game.mainline():
                if terminate_flag:
                    break

                parent_node = node.parent
                if parent_node is None:
                    continue

                board_before_move = parent_node.board()
                current_ply = parent_node.ply()

                if current_ply < min_ply:
                    continue

                comment = node.comment
                try:
                    if "Stockfish eval:" in comment:
                        eval_str = comment.split("Stockfish eval:")[1].split()[0]
                        evaluation = float(eval_str) * 100
                    else:
                        continue
                except (ValueError, IndexError):
                    tqdm.write(f"[PID {worker_pid}] Skipping position due to malformed comment: {comment}")
                    continue

                features = extract_features(board_before_move)
                if features:
                    features_chunk.append(features)
                    outcomes_chunk.append(evaluation)
                    plies_chunk.append(current_ply)
                    position_count_total += 1
                    position_in_current_chunk += 1

                    if position_in_current_chunk >= chunk_size:
                        write_callback(features_chunk, outcomes_chunk, plies_chunk)
                        features_chunk.clear()
                        outcomes_chunk.clear()
                        plies_chunk.clear()
                        position_in_current_chunk = 0

            processed_games_count += 1

        except (ValueError, KeyError, IndexError, chess.IllegalMoveError, chess.InvalidMoveError, AttributeError) as e:
            error_count += 1
            skipped_games += 1
            tqdm.write(f"[PID {worker_pid}] Skipping game {game_count} due to error: {e}")
            continue
        except Exception as e:
            error_count += 1
            tqdm.write(f"[PID {worker_pid}] Unexpected error in game {game_count}: {e}")
            continue

    if features_chunk:
        write_callback(features_chunk, outcomes_chunk, plies_chunk)

    pbar.close()
    tqdm.write(f"[PID {worker_pid}] Processed {processed_games_count} games, skipped {skipped_games} games, with {error_count} errors.")
    return processed_games_count

def convert_json_to_hdf5(json_path, output_hdf5_path):
    """
    Convert a JSON file containing chess evaluations to an HDF5 file.
    :param json_path: Path to the input JSON file.
    :param output_hdf5_path: Path to the output HDF5 file.
    """
    print(f"Converting JSON file: {json_path} to HDF5 file: {output_hdf5_path}")
    features = []
    outcomes = []
    plies = []

    try:
        with open(json_path, "r", encoding="utf-8") as json_file:
            for line in tqdm(json_file, desc="Processing JSON lines"):
                data = json.loads(line)
                fen = data["fen"]
                evals = data.get("evals", [])

                if evals:
                    best_eval = evals[0]
                    pvs = best_eval.get("pvs", [])
                    if pvs:
                        cp = pvs[0].get("cp")
                        mate = pvs[0].get("mate")
                        if cp is not None:
                            evaluation = cp / 100.0
                        elif mate is not None:
                            evaluation = 10000 if mate > 0 else -10000
                        else:
                            continue

                        board = chess.Board(fen)
                        feature_indices = extract_features(board)
                        if feature_indices:
                            features.append(feature_indices)
                            outcomes.append(evaluation)

                            # Calculate ply count correctly
                            ply_count = (board.fullmove_number - 1) * 2
                            if not board.turn:  # If it's Black's turn, add 1
                                ply_count += 1
                            plies.append(ply_count)

        with h5py.File(output_hdf5_path, "w") as hf:
            vlen_dtype = h5py.special_dtype(vlen=np.int32)
            hf.create_dataset("features", data=np.array([np.array(f, dtype=np.int32) for f in features], dtype=object), dtype=vlen_dtype)
            hf.create_dataset("outcomes", data=np.array(outcomes, dtype=np.float32))
            hf.create_dataset("plies", data=np.array(plies, dtype=np.uint16))

        print(f"Successfully converted JSON to HDF5: {output_hdf5_path}")

    except Exception as e:
        print(f"Error during JSON to HDF5 conversion: {e}")
        traceback.print_exc()

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON files to HDF5 or PGN.")
    parser.add_argument("input_path", help="Path to the input JSON file.")
    parser.add_argument("output_path", help="Path to the output HDF5 or PGN file.")
    parser.add_argument("--format", choices=["hdf5", "pgn"], required=True, help="Output format ('hdf5' or 'pgn').")

    args = parser.parse_args()

    if args.format == "hdf5":
        convert_json_to_hdf5(args.input_path, args.output_path)
    elif args.format == "pgn":
        print("Error: The 'process_json_evaluations' function is not implemented.")
        sys.exit(1)