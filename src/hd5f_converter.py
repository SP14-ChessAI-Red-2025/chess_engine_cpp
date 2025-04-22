import chess
import numpy as np
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
import orjson
from multiprocessing import Queue, Process

# Global flag to indicate termination (optional, but good practice for long processes)
terminate_flag = False

def handle_termination_signal(signum, frame):
    global terminate_flag
    print("\n[INFO] Termination signal received. Signalling workers...")
    terminate_flag = True

signal.signal(signal.SIGINT, handle_termination_signal)
signal.signal(signal.SIGTERM, handle_termination_signal)


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
# CHUNK_SIZE = 25000 # This was for PGN chunking, less relevant now


# --- Feature Extraction ---
def extract_features(board: chess.Board) -> list[int]:
    indices = []
    try:
        if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
            return []

        for square, piece in board.piece_map().items():
            piece_index = PIECE_TO_INDEX.get(piece.piece_type)
            if piece_index is not None:
                base_index = piece_index * 64
                if piece.color == chess.WHITE:
                    indices.append(base_index + square)
                else:
                    indices.append(base_index + square + FEATURES_PER_COLOR)

    except Exception as e:
        print(f"Error extracting features for FEN {board.fen()}: {e}", file=sys.stderr)
        return []

    return indices

# --- Functions below are related to PGN/Stockfish eval, keep if needed, otherwise remove ---
# def evaluate_position(board: chess.Board, engine_path: str = "stockfish") -> float: ...
# def evaluate_positions(boards: list[chess.Board], engine_path: str = "stockfish", depth: int = 4) -> list[float]: ...
# def write_hdf5_chunk(hf, features_chunk, outcomes_chunk, plies_chunk, first_chunk_in_file): ...
# def process_pgn_file_worker(input_args): ...
# def _process_stream_chunked(pgn_stream, features_chunk, outcomes_chunk, plies_chunk, min_ply, worker_pid, filename, chunk_size, write_callback): ...
# def convert_json_to_hdf5(json_path, output_hdf5_path): # Original single-threaded version
# --- Functions related to byte-based parallel JSON processing (now superseded) ---
# def process_chunk(start, end, json_path, chunk_id): ...
# def parallel_convert_json_to_hdf5(json_path, output_hdf5_path, num_workers=4, chunk_size=1000000): ...
# --- End of potentially removable functions ---


# --- Parallel JSON Processing (Line-Based) ---

def process_lines_chunk(lines_chunk, chunk_id):
    """
    Process a chunk of lines (list of strings) from the JSON file.
    :param lines_chunk: A list of strings, each expected to be a JSON object.
    :param chunk_id: ID of the chunk being processed.
    :return: Tuple of (features, outcomes, plies).
    """
    global terminate_flag
    features = []
    outcomes = []
    plies = []
    processed_count = 0

    for line in lines_chunk:
        if terminate_flag: # Check if termination is requested
            break

        # Skip empty lines that might result from splitting/reading
        line = line.strip()
        if not line:
            continue

        try:
            data = orjson.loads(line) # Parse one line at a time
            fen = data["fen"]
            evals = data.get("evals", [])

            if evals:
                # Assuming the first evaluation is the desired one
                best_eval = evals[0]
                pvs = best_eval.get("pvs", [])
                if pvs:
                    # Assuming the first PV line has the score
                    score_info = pvs[0]
                    cp = score_info.get("cp")
                    mate = score_info.get("mate")

                    evaluation = 0.0 # Default evaluation

                    if cp is not None:
                        # Convert centipawns to pawn units (adjust if your NNUE expects different scale)
                        evaluation = cp / 100.0
                    elif mate is not None:
                        # Assign large scores for mate, ensure sign is correct
                        # Adjust scale (e.g., 100.0 or 10000) based on NNUE training range
                        evaluation = 100.0 if mate > 0 else -100.0
                    else:
                        continue # Skip if no 'cp' or 'mate' key found in the first PV

                    board = chess.Board(fen)
                    # Basic validity check - does it have kings? Is position legal?
                    # board.is_valid() can be slow, maybe just check kings
                    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
                        # print(f"[Chunk {chunk_id}] Skipping invalid FEN (no king): {fen}", file=sys.stderr)
                        continue

                    feature_indices = extract_features(board) # Use your existing function
                    if feature_indices: # Ensure features were extracted successfully
                        features.append(feature_indices)
                        outcomes.append(evaluation)

                        # Calculate ply count from FEN
                        ply_count = (board.fullmove_number - 1) * 2
                        if board.turn == chess.BLACK: # If it's Black's turn
                            ply_count += 1
                        plies.append(ply_count)
                        processed_count += 1

        except KeyError as e:
            # Handle missing keys like 'fen' or 'evals'
            # print(f"[Chunk {chunk_id}] Skipping line due to missing key {e}: '{line[:100]}...'", file=sys.stderr)
            pass
        except (orjson.JSONDecodeError, TypeError, ValueError) as e:
            # Handle malformed JSON or issues converting data (e.g., FEN parsing)
            # print(f"[Chunk {chunk_id}] Error processing line: '{line[:100]}...': {e}", file=sys.stderr)
            pass # Optionally log or count errors
        except Exception as e:
            # Catch any other unexpected errors during line processing
            # print(f"[Chunk {chunk_id}] Unexpected error on line: '{line[:100]}...': {e}", file=sys.stderr)
            # traceback.print_exc(file=sys.stderr) # More detail if needed
            pass

    # print(f"[Chunk {chunk_id}] Processed {processed_count} valid positions.") # Debug output per chunk
    return features, outcomes, plies

def hdf5_writer(queue, output_hdf5_path):
    with h5py.File(output_hdf5_path, "w") as hf:
        # Initialize datasets here
        while True:
            data = queue.get()
            if data is None:  # Sentinel value to terminate
                break
            # Write data to HDF5

def parallel_convert_json_to_hdf5_by_lines(json_path, output_hdf5_path, num_workers=4, lines_per_chunk=200000):
    """
    Convert a large JSON Lines file to HDF5 format using parallel processing,
    distributing lines instead of bytes. Writes results sequentially.
    :param json_path: Path to the input JSON file.
    :param output_hdf5_path: Path to the output HDF5 file.
    :param num_workers: Number of parallel workers.
    :param lines_per_chunk: How many lines to process in each task sent to a worker.
    """
    global terminate_flag
    print(f"Converting JSON file (line by line): {json_path} to HDF5 file: {output_hdf5_path}")
    start_time = time.time()

    # --- HDF5 File Initialization ---
    try:
        # <<< FIX: Only call makedirs if dirname is not empty >>>
        output_dir = os.path.dirname(output_hdf5_path)
        if output_dir: # Check if directory part exists
            print(f"Ensuring output directory exists: {output_dir}")
            os.makedirs(output_dir, exist_ok=True) # Create intermediate dirs if needed

        with h5py.File(output_hdf5_path, "w") as hf:
            vlen_dtype = h5py.special_dtype(vlen=np.int32)
            # Initialize resizable datasets with chunking for efficiency
            # Adjust chunk size based on expected data size and access patterns
            h5_chunk_size = max(1, lines_per_chunk // 10) # Example chunk size
            features_dset = hf.create_dataset("features", (0,), maxshape=(None,), dtype=vlen_dtype, chunks=(h5_chunk_size,), compression="gzip")
            outcomes_dset = hf.create_dataset("outcomes", (0,), maxshape=(None,), dtype=np.float32, chunks=(h5_chunk_size * 5,), compression="gzip") # Larger chunk for simpler types
            plies_dset = hf.create_dataset("plies", (0,), maxshape=(None,), dtype=np.uint16, chunks=(h5_chunk_size * 5,), compression="gzip")

            # --- Processing in Parallel ---
            # Use try-with-resources for Pool and file handle
            try:
                with multiprocessing.Pool(num_workers) as pool, open(json_path, "r", encoding="utf-8") as f:
                    tasks = []
                    lines_buffer = []
                    total_lines_read = 0
                    tasks_submitted = 0

                    print("Reading input file and submitting tasks...")
                    # Use tqdm for reading progress (can be slow for line counting)
                    # Getting total lines first can be slow:
                    # total_lines = sum(1 for line in open(json_path, 'r'))
                    # pbar_read = tqdm(total=total_lines, desc="Reading JSONL", unit=" lines", smoothing=0.1)
                    pbar_read = tqdm(desc="Reading JSONL", unit=" lines", smoothing=0.1, mininterval=2.0)

                    for line in f:
                        if terminate_flag:
                            print("\nTermination requested, stopping reading.")
                            break
                        lines_buffer.append(line)
                        total_lines_read += 1
                        pbar_read.update(1)
                        if len(lines_buffer) >= lines_per_chunk:
                            tasks.append(pool.apply_async(process_lines_chunk, (lines_buffer.copy(), tasks_submitted)))
                            lines_buffer.clear()
                            tasks_submitted += 1
                            # Optional: Limit number of pending tasks to manage memory
                            # while len(tasks) >= num_workers * 2: # Example limit
                            #     time.sleep(0.1) # Wait briefly

                    # Add any remaining lines as a final task
                    if lines_buffer and not terminate_flag:
                        tasks.append(pool.apply_async(process_lines_chunk, (lines_buffer, tasks_submitted)))
                        tasks_submitted += 1

                    pbar_read.close()
                    print(f"Submitted {len(tasks)} tasks for {total_lines_read} lines read.")

                    # --- Collecting Results and Writing to HDF5 ---
                    print("Processing tasks and writing to HDF5...")
                    pbar_write = tqdm(total=len(tasks), desc="Processing Chunks", unit=" chunk", mininterval=2.0)
                    total_positions_written = 0
                    error_count = 0

                    for task in tasks:
                        if terminate_flag and not task.ready():
                             # If termination requested, don't wait indefinitely for tasks
                             # This part needs careful handling, maybe Pool.terminate() is better
                             print("Skipping remaining tasks due to termination signal.")
                             break
                        try:
                            # Get results from worker (with a timeout?)
                            features, outcomes, plies = task.get(timeout=300) # 5 min timeout
                            count = len(outcomes)

                            if count > 0:
                                # Convert features to the required format (list of numpy arrays)
                                np_features_list = [np.array(lst, dtype=np.int32) for lst in features]
                                # This conversion can be memory intensive, do it just before writing
                                np_features_obj = np.array(np_features_list, dtype=object)
                                np_outcomes = np.array(outcomes, dtype=np.float32)
                                np_plies = np.array(plies, dtype=np.uint16)

                                # Resize datasets and append data (this needs file to be open)
                                current_size = features_dset.shape[0]
                                features_dset.resize((current_size + count,))
                                features_dset[current_size:] = np_features_obj
                                # Clear intermediate large objects
                                del np_features_list, np_features_obj

                                outcomes_dset.resize((outcomes_dset.shape[0] + count,))
                                outcomes_dset[-count:] = np_outcomes
                                del np_outcomes

                                plies_dset.resize((plies_dset.shape[0] + count,))
                                plies_dset[-count:] = np_plies
                                del np_plies

                                total_positions_written += count
                                gc.collect() # Force garbage collection periodically?

                        except multiprocessing.TimeoutError:
                             print(f"\nWarning: Timeout getting result from worker chunk.", file=sys.stderr)
                             error_count += 1
                        except Exception as e:
                            print(f"\nError retrieving result from worker or writing to HDF5: {e}", file=sys.stderr)
                            traceback.print_exc(file=sys.stderr) # Print traceback for debugging
                            error_count += 1
                        finally:
                             pbar_write.update(1) # Update progress bar

                    pbar_write.close()
                    # Attempt to gracefully close the pool
                    pool.close()
                    pool.join()

            except KeyboardInterrupt: # Catch Ctrl+C here too
                 print("\nKeyboardInterrupt detected during processing. Cleaning up...")
                 terminate_flag = True # Ensure flag is set
                 # Pool termination might be needed here if hangs
                 # pool.terminate()
                 # pool.join()
            except Exception as e: # Catch errors during pool/file setup
                 print(f"\nError during parallel processing setup or collection: {e}", file=sys.stderr)
                 traceback.print_exc(file=sys.stderr)


            if not terminate_flag and error_count == 0:
                 print(f"\nSuccessfully converted JSON to HDF5: {output_hdf5_path}")
            elif terminate_flag:
                 print(f"\nConversion terminated early.")
            else:
                 print(f"\nConversion finished with {error_count} errors during result collection/writing.")

            print(f"Total positions written: {total_positions_written}")
            duration = time.time() - start_time
            print(f"Total time: {duration:.2f} seconds")


    except Exception as e:
         print(f"Critical HDF5 file error for {output_hdf5_path}: {e}", file=sys.stderr)
         traceback.print_exc(file=sys.stderr)
    except Exception as e:
        # Catch errors like directory creation failure if permissions are wrong
        print(f"Critical error during JSON to HDF5 conversion setup: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr) # <<< USES traceback


# --- Main Execution ---
if __name__ == "__main__":
    # Set start method for multiprocessing (optional, 'fork' is default on Linux)
    # try:
    #    multiprocessing.set_start_method('fork') # or 'spawn' or 'forkserver'
    # except RuntimeError:
    #    pass # Already set or not applicable

    parser = argparse.ArgumentParser(description="Convert JSON Lines files with chess FENs and evals to HDF5.") # <<< USES argparse
    parser.add_argument("input_path", help="Path to the input JSON Lines file (.jsonl).")
    parser.add_argument("output_path", help="Path to the output HDF5 file.")
    parser.add_argument("--format", choices=["hdf5"], required=True, help="Output format (currently only 'hdf5' supported).")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of parallel workers (default: number of CPUs).")
    parser.add_argument("--lines_per_chunk", type=int, default=200000, help="Number of lines per processing chunk (default: 50,000).")

    args = parser.parse_args()

    # Set default num_workers if provided value is invalid
    if args.num_workers <= 0:
        print(f"Invalid --num_workers value ({args.num_workers}). Using default.")
        args.num_workers = os.cpu_count()
        print(f"Using default number of workers: {args.num_workers}")


    if args.format == "hdf5":
        # Call the corrected line-based parallel function
        parallel_convert_json_to_hdf5_by_lines(args.input_path, args.output_path, args.num_workers, args.lines_per_chunk)
    # Elif args.format == "pgn": # Add PGN support later if needed
    #    print("Error: PGN conversion from JSON is not implemented in this example.")
    #    sys.exit(1)
    else:
         print(f"Error: Unsupported format '{args.format}' specified.")
         sys.exit(1)

    print("Script finished.")