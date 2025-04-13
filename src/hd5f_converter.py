import chess
import chess.pgn
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

# --- HDF5 Dependency ---
try:
    import h5py
    print(f"Using h5py version: {h5py.__version__}")
    sys.stdout.flush()
except ImportError:
    print("Error: h5py library not found.")
    print("Please install it using: pip install h5py")
    sys.exit(1)


# --- Print Library Version ---
try:
    print(f"Using python-chess version: {chess.__version__}")
    sys.stdout.flush()
except AttributeError:
    print("Could not determine python-chess version automatically.")
    sys.stdout.flush()


# --- Configuration ---
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
PIECE_TO_INDEX = {piece_type: i for i, piece_type in enumerate(PIECE_TYPES)}
TOTAL_FEATURES = 768
FEATURES_PER_COLOR = TOTAL_FEATURES // 2
CHUNK_SIZE = 100000 # Number of positions to accumulate before writing to HDF5

# --- Feature Extraction ---
def extract_features(board: chess.Board) -> list[int]:
    """
    Extracts feature indices for a simple 768-feature NNUE input layer.
    """
    indices = []
    for piece_type in PIECE_TYPES:
        piece_index = PIECE_TO_INDEX[piece_type]
        base_index = piece_index * 64
        piece_bb = board.pieces(piece_type, chess.WHITE)
        for square in piece_bb:
            feature_index = base_index + square
            if 0 <= feature_index < FEATURES_PER_COLOR:
                indices.append(feature_index)
            else: pass

    # Process Black pieces
    black_offset = FEATURES_PER_COLOR
    for piece_type in PIECE_TYPES:
        piece_index = PIECE_TO_INDEX[piece_type]
        base_index = piece_index * 64 + black_offset
        piece_bb = board.pieces(piece_type, chess.BLACK)
        for square in piece_bb: # Direct iteration
            feature_index = base_index + square
            if FEATURES_PER_COLOR <= feature_index < TOTAL_FEATURES:
                indices.append(feature_index)
            else: pass

    # King presence check
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)
    if white_king_square is None or black_king_square is None:
        return []

    indices.sort()
    return indices


def write_hdf5_chunk(hf, features_chunk, outcomes_chunk, plies_chunk, first_chunk_in_file):
    """
    Writes a chunk of data to the HDF5 file. Creates datasets if it's the first chunk.
    """
    count = len(outcomes_chunk)
    if count == 0:
        return 0

    try:
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
        vlen_dtype = h5py.special_dtype(vlen=np.int32) # Define vlen type

        if first_chunk_in_file:
            hf.create_dataset('features', data=np_features_obj, dtype=vlen_dtype,
                              maxshape=(None,), chunks=(max(1, CHUNK_SIZE // 10),), # Ensure chunk size >= 1
                              compression="gzip")
            hf.create_dataset('outcomes', data=np_outcomes, maxshape=(None,),
                              chunks=(CHUNK_SIZE,), compression="gzip")
            hf.create_dataset('plies', data=np_plies, maxshape=(None,),
                              chunks=(CHUNK_SIZE,), compression="gzip")
        else:
            features_dset = hf['features']
            features_dset.resize((features_dset.shape[0] + count,))
            features_dset[-count:] = np_features_obj

            outcomes_dset = hf['outcomes']
            outcomes_dset.resize((outcomes_dset.shape[0] + count,))
            outcomes_dset[-count:] = np_outcomes

            plies_dset = hf['plies']
            plies_dset.resize((plies_dset.shape[0] + count,))
            plies_dset[-count:] = np_plies

        return count
    except Exception as e:
        pid = os.getpid()
        print(f"\n[PID {pid}] Error writing chunk to HDF5 file '{hf.filename}': {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 0


# --- PGN Processing (Single File - Worker Task using HDF5) ---
def process_pgn_file_worker(input_args):
    """
    Reads a single PGN or PGN.ZST file, extracts features/outcomes,
    and saves incrementally to an HDF5 file.
    Designed to be called by multiprocessing.Pool. Handles one file entirely.

    Args:
        input_args (tuple): A tuple containing (pgn_path, output_hdf5_path, min_ply).

    Returns:
        bool: True on success or if no positions extracted, False on critical error.
    """
    pgn_path, output_hdf5_path, min_ply = input_args
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

    # --- Open HDF5 file ---
    try:
        with h5py.File(output_hdf5_path, 'w') as hf:
            def hdf5_write_callback(features_chunk, outcomes_chunk, plies_chunk):
                nonlocal total_positions_written, first_chunk_in_file
                written_count = write_hdf5_chunk(hf, features_chunk, outcomes_chunk, plies_chunk, first_chunk_in_file)
                total_positions_written += written_count
                if written_count > 0:
                    first_chunk_in_file = False

            # --- Stream Processing Logic ---
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
                    if is_compressed:
                        pgn_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
                    else:
                        pgn_stream = reader

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
                    hdf5_write_callback(features_current_chunk, outcomes_current_chunk, plies_current_chunk)


            except (MemoryError, ValueError, KeyError, IndexError, chess.IllegalMoveError, chess.InvalidMoveError, AttributeError) as e:
                 print(f"\n[PID {worker_pid}] Error during PGN stream processing for {base_pgn_name}: {e}", file=sys.stderr)
                 traceback.print_exc(file=sys.stderr)

            except Exception as e:
                print(f"\n[PID {worker_pid}] Critical Error processing file {base_pgn_name}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                return False
            finally:
                if compressed_file_handle and not compressed_file_handle.closed:
                    compressed_file_handle.close()
                del features_current_chunk, outcomes_current_chunk, plies_current_chunk
                gc.collect()

    except Exception as e:
        # Catch errors related to opening/closing HDF5 file
        print(f"\n[PID {worker_pid}] Critical HDF5 file error for {base_hdf5_name}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False


    if total_positions_written == 0:
        tqdm.write(f"[PID {worker_pid}] No valid positions found or extracted from {base_pgn_name}. HDF5 file created but may be empty: {base_hdf5_name}")
        sys.stdout.flush()

    duration = time.time() - start_time
    tqdm.write(f"[PID {worker_pid}] Finished processing {base_pgn_name}. Extracted {total_positions_written} positions from {processed_game_count_in_file} valid games in {duration:.2f}s. Saved to {base_hdf5_name}")
    sys.stdout.flush()
    return True


def _process_stream_chunked(pgn_stream, features_chunk, outcomes_chunk, plies_chunk, min_ply, worker_pid, filename, chunk_size, write_callback):
    """
    Helper function to process a PGN stream, adding to chunk lists and calling
    write_callback when chunk size is reached. Returns processed game count.
    (Largely unchanged logic from previous example, just calls the callback)
    """
    game_count = 0
    position_count_total = 0
    position_in_current_chunk = 0
    skipped_games = 0
    processed_games_count = 0
    error_count = 0

    desc = f"[PID {worker_pid}] Processing {filename}"
    pbar = tqdm(desc=desc, unit=" game", leave=False, position=worker_pid % 30, mininterval=5.0, smoothing=0.1)

    while True:
        game = None
        node_being_processed = None
        try:
            game = chess.pgn.read_game(pgn_stream)
            if game is None:
                break # End of stream

            game_count += 1
            pbar.update(1)

            result = game.headers.get("Result", "*")
            if result == "1-0": outcome = 1.0
            elif result == "0-1": outcome = 0.0
            elif result == "1/2-1/2": outcome = 0.5
            else:
                skipped_games += 1
                continue

            # --- Process the valid game ---
            processed_positions_in_game = 0
            for node in game.mainline():
                node_being_processed = node
                parent_node = node.parent
                if parent_node is None:
                    continue

                board_before_move = parent_node.board()
                current_ply = parent_node.ply()

                if current_ply >= min_ply:
                    features = extract_features(board_before_move)
                    if features:
                        features_chunk.append(features)
                        outcomes_chunk.append(outcome)
                        plies_chunk.append(current_ply)
                        position_count_total += 1
                        position_in_current_chunk += 1
                        processed_positions_in_game += 1

                        # --- Check if chunk is full ---
                        if position_in_current_chunk >= chunk_size:
                            write_callback(features_chunk, outcomes_chunk, plies_chunk)
                            features_chunk.clear()
                            outcomes_chunk.clear()
                            plies_chunk.clear()
                            position_in_current_chunk = 0


            if processed_positions_in_game > 0:
                processed_games_count += 1

        except (ValueError, KeyError, IndexError, chess.IllegalMoveError, chess.InvalidMoveError, AttributeError) as e:
             error_count += 1
             if error_count % 1000 == 0:
                 move_san = "N/A"
                 if node_being_processed and node_being_processed.move:
                     try:
                         move_san = node_being_processed.parent.board().san(node_being_processed.move)
                     except:
                         move_san = str(node_being_processed.move)
                 tqdm.write(f"\n[PID {worker_pid}] Error processing game {game_count} near move {move_san} in {filename} (error #{error_count}): {e}")
             continue
        except Exception as e:
             error_count += 1
             tqdm.write(f"\n[PID {worker_pid}] An unexpected error occurred processing game {game_count} in {filename} (error #{error_count}): {e}")
             traceback.print_exc(file=sys.stderr)
             continue

        if game_count % 1000 == 0:
             pbar.set_postfix({"pos": f"{position_count_total/1000:.1f}k", "skip": skipped_games, "err": error_count}, refresh=False) # Refresh=False might reduce overhead

    pbar.close()
    tqdm.write(f"[PID {worker_pid}] Stream processed for {filename}: {processed_games_count} games yielded positions ({position_count_total} total), {skipped_games} skipped, {error_count} errors.")
    sys.stdout.flush()
    return processed_games_count


# --- Main Execution ---
if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Convert PGN/ZST files to HDF5 using file-level multiprocessing and chunking.")
    parser.add_argument("input_path", help="Path to input PGN/ZST file or directory.")
    parser.add_argument("output_dir", help="Directory to save output HDF5 file(s).")
    parser.add_argument("--min_ply", type=int, default=8, help="Minimum ply count (default: 8).")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (concurrent files) (default: CPU count).")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE, help=f"Number of positions per HDF5 write chunk (default: {CHUNK_SIZE}).")

    args = parser.parse_args()

    # --- Update CHUNK_SIZE from arguments ---
    CHUNK_SIZE = args.chunk_size
    print(f"Using HDF5 chunk size (positions per write): {CHUNK_SIZE}")
    sys.stdout.flush()

    os.makedirs(args.output_dir, exist_ok=True)

    overall_start_time = time.time()

    files_to_process = []
    if os.path.isdir(args.input_path):
        print(f"Looking for PGN/PGN.ZST files in directory: {args.input_path}")
        sys.stdout.flush()
        try:
            all_files = os.listdir(args.input_path)
            for f in sorted(all_files):
                if f.lower().endswith((".pgn", ".pgn.zst")):
                    files_to_process.append(os.path.join(args.input_path, f))
        except OSError as e:
             print(f"Error reading input directory {args.input_path}: {e}")
             sys.exit(1)

        if not files_to_process:
            print("No PGN or PGN.ZST files found in the input directory.")
            sys.exit(1)
        else:
             print(f"Found {len(files_to_process)} files to process.")
             sys.stdout.flush()

        # --- Multiprocessing Logic ---
        tasks = []
        for input_file in files_to_process:
            base_name = os.path.basename(input_file)
            if base_name.lower().endswith(".pgn.zst"): output_filename_base = base_name[:-8]
            elif base_name.lower().endswith(".pgn"): output_filename_base = base_name[:-4]
            else: output_filename_base = os.path.splitext(base_name)[0] # Fallback
            output_file = os.path.join(args.output_dir, f"{output_filename_base}.hdf5")
            tasks.append((input_file, output_file, args.min_ply))

        if args.workers is None:
            try:
                num_workers = os.cpu_count()
                if num_workers is None: num_workers = 4
                print(f"Number of workers not specified, defaulting to CPU count: {num_workers}")
            except NotImplementedError:
                num_workers = 4
                print(f"Could not detect CPU count, defaulting to {num_workers} workers.")
        else:
            num_workers = max(1, args.workers)

        pool_size = min(num_workers, len(files_to_process))
        if tasks and pool_size == 0:
            pool_size = 1
        print(f"Starting multiprocessing pool with {pool_size} workers (requested: {num_workers}, files: {len(files_to_process)})...")
        sys.stdout.flush()

        if not tasks:
             print("No tasks to process.")
             sys.exit(0)

        results = []
        try:
            with multiprocessing.Pool(processes=pool_size) as pool:
                with tqdm(total=len(tasks), desc="Overall Progress", smoothing=0.1, unit="file") as pbar:
                    for result in pool.imap_unordered(process_pgn_file_worker, tasks):
                        results.append(result)
                        pbar.update(1)
        except KeyboardInterrupt:
            print("\nCaught KeyboardInterrupt! Terminating pool...")
            pool.close()
            pool.join()
            print("Pool closed.")
            sys.exit(1)
        except Exception as e:
            print(f"\nError during pool processing: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            try: pool.close(); pool.join()
            except: pass

        success_count = sum(1 for r in results if r is True)
        failure_count = len(tasks) - success_count
        print(f"\nMultiprocessing finished. Successful files: {success_count}, Failed/Skipped files: {failure_count}")
        sys.stdout.flush()

    elif os.path.isfile(args.input_path):
        if args.input_path.lower().endswith((".pgn", ".pgn.zst")):
            base_name = os.path.basename(args.input_path)
            if base_name.lower().endswith(".pgn.zst"): output_filename_base = base_name[:-8]
            elif base_name.lower().endswith(".pgn"): output_filename_base = base_name[:-4]
            else: output_filename_base = os.path.splitext(base_name)[0]
            output_file = os.path.join(args.output_dir, f"{output_filename_base}.hdf5")
            print(f"Processing single file: {args.input_path} -> {output_file}")
            sys.stdout.flush()
            process_pgn_file_worker((args.input_path, output_file, args.min_ply))
        else:
            print(f"Error: Input file '{args.input_path}' is not a .pgn or .pgn.zst file.", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Error: Input path '{args.input_path}' is not a valid file or directory.", file=sys.stderr)
        sys.exit(1)

    overall_duration = time.time() - overall_start_time
    print(f"\nTotal execution time: {overall_duration:.2f} seconds.")