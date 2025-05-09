import math
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import argparse
import os
import time
import io
import sys

# --- Imports needed for data processing ---
import chess
import chess.pgn
import zstandard as zstd
from tqdm import tqdm
import h5py

# --- Constants ---
TOTAL_FEATURES = 768
FEATURES_PER_COLOR = TOTAL_FEATURES // 2
# --- Constants for PGN processing ---
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
PIECE_TO_INDEX = {piece_type: i for i, piece_type in enumerate(PIECE_TYPES)}
DEFAULT_MIN_PLY = 8
# --------------------------------

# --- NNUE Model Definition ---
class NNUE(nn.Module):
    def __init__(self, feature_dim=TOTAL_FEATURES, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.input_layer = nn.EmbeddingBag(feature_dim, embed_dim, mode="sum", sparse=False)
        self.hidden1 = nn.Linear(embed_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(hidden_dim // 2, 1)

    def forward(self, features_indices, offsets=None):
        x = self.input_layer(features_indices, offsets=offsets)
        x = self.relu1(self.hidden1(x))
        x = self.relu2(self.hidden2(x))
        x = self.output(x)
        return x

# --- Feature Extraction Function (For PGN processing) ---
def extract_features(board: chess.Board) -> list[int]:
    """
    Extracts feature indices for a simple 768-feature NNUE input layer.
    """
    indices = []
    for piece_type in PIECE_TYPES:
        piece_index = PIECE_TO_INDEX[piece_type]
        base_index = piece_index * 64
        piece_bb = board.pieces(piece_type, chess.WHITE)
        for square in piece_bb: # Direct iteration
            feature_index = base_index + square
            if 0 <= feature_index < FEATURES_PER_COLOR:
                indices.append(feature_index)
            else: pass
    black_offset = FEATURES_PER_COLOR
    for piece_type in PIECE_TYPES:
        piece_index = PIECE_TO_INDEX[piece_type]
        base_index = piece_index * 64 + black_offset
        piece_bb = board.pieces(piece_type, chess.BLACK)
        for square in piece_bb:
            feature_index = base_index + square
            if FEATURES_PER_COLOR <= feature_index < TOTAL_FEATURES:
                indices.append(feature_index)
            else: pass
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)
    if white_king_square is None or black_king_square is None:
        return []
    return indices

# --- PGN Iterable Dataset Definition ---
class IterablePgnDataset(IterableDataset):
    def __init__(self, pgn_files: list[str], min_ply: int):
        """
        PyTorch IterableDataset for streaming data directly from PGN/ZST files.

        Args:
            pgn_files (list[str]): List of paths to PGN or PGN.ZST files.
            min_ply (int): Minimum ply count for positions to be included.
        """
        super().__init__()
        self.pgn_files = pgn_files
        self.min_ply = min_ply
        if not self.pgn_files:
            raise ValueError("No PGN files provided to IterablePgnDataset.")
        print(f"IterablePgnDataset initialized with {len(self.pgn_files)} PGN/ZST files.")

    def _parse_stream(self, pgn_stream):
        """Generator function to yield features/outcomes from a single PGN stream."""
        game_count = 0
        while True:
            game = None
            try:
                game = chess.pgn.read_game(pgn_stream)
                if game is None:
                    break

                game_count += 1
                result = game.headers.get("Result", "*")
                if result == "1-0": outcome = 1.0
                elif result == "0-1": outcome = 0.0
                elif result == "1/2-1/2": outcome = 0.5
                else:
                    continue

                for node in game.mainline():
                    parent_node = node.parent
                    if parent_node is None: continue

                    current_ply = parent_node.ply()
                    if current_ply >= self.min_ply:
                        board = parent_node.board()
                        features = extract_features(board)
                        if features:
                            yield features, outcome

            except (ValueError, KeyError, IndexError, AttributeError, chess.IllegalMoveError, chess.InvalidMoveError) as e:
                 print(f"Warning: Error processing game #{game_count} in PGN stream: {e}", file=sys.stderr)
                 continue
            except Exception as e:
                 print(f"Warning: Unexpected error processing game #{game_count} in PGN stream: {e}", file=sys.stderr)
                 traceback.print_exc(file=sys.stderr)
                 continue

    def __iter__(self):
        """Iterator method called by DataLoader."""
        worker_info = torch.utils.data.get_worker_info()
        files_to_process = self.pgn_files

        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            files_per_worker = int(math.ceil(len(self.pgn_files) / float(num_workers)))
            start_idx = worker_id * files_per_worker
            end_idx = min(start_idx + files_per_worker, len(self.pgn_files))
            files_to_process = self.pgn_files[start_idx:end_idx]

        for pgn_path in files_to_process:
            pgn_stream = None
            reader = None
            compressed_file = None
            try:
                is_compressed = pgn_path.lower().endswith(".zst")
                if is_compressed:
                    dctx = zstd.ZstdDecompressor()
                    compressed_file = open(pgn_path, 'rb')
                    reader = dctx.stream_reader(compressed_file)
                    pgn_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
                else:
                    pgn_stream = open(pgn_path, 'rt', encoding='utf-8', errors='replace')

                yield from self._parse_stream(pgn_stream)

            except Exception as e:
                 print(f"Error opening/reading file {pgn_path}: {e}", file=sys.stderr)
                 traceback.print_exc(file=sys.stderr)
            finally:
                 if pgn_stream is not None:
                     try: pgn_stream.close()
                     except Exception: pass
                 if reader is not None:
                     try: reader.close()
                     except Exception: pass
                 if compressed_file is not None:
                     try: compressed_file.close()
                     except Exception: pass

# --- HDF5 Iterable Dataset Definition ---
class IterableHdf5Dataset(IterableDataset):
    def __init__(self, hdf5_files: list[str]):
        """
        PyTorch IterableDataset for streaming data from multiple HDF5 files.

        Args:
            hdf5_files (list[str]): List of paths to HDF5 files.
        """
        super().__init__()
        self.hdf5_files = hdf5_files
        if not self.hdf5_files:
            raise ValueError("No HDF5 files provided to IterableHdf5Dataset.")
        print(f"IterableHdf5Dataset initialized with {len(self.hdf5_files)} HDF5 files.")

    def __iter__(self):
        """Iterator method called by DataLoader."""
        worker_info = torch.utils.data.get_worker_info()
        files_to_process = self.hdf5_files

        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            files_per_worker = int(math.ceil(len(self.hdf5_files) / float(num_workers)))
            start_idx = worker_id * files_per_worker
            end_idx = min(start_idx + files_per_worker, len(self.hdf5_files))
            files_to_process = self.hdf5_files[start_idx:end_idx]

        # Iterate through assigned files
        for hdf5_path in files_to_process:
            try:
                with h5py.File(hdf5_path, 'r') as hf:
                    features_dset = hf['features']
                    outcomes_dset = hf['outcomes']

                    num_samples_in_file = len(outcomes_dset)

                    for idx in range(num_samples_in_file):
                        features = features_dset[idx]
                        outcome = outcomes_dset[idx]
                        yield features.tolist(), float(outcome)

            except KeyError as e:
                 print(f"Error: Missing dataset '{e}' in file {hdf5_path}. Skipping file.", file=sys.stderr)
            except Exception as e:
                 print(f"Error opening/reading file {hdf5_path}: {e}", file=sys.stderr)
                 traceback.print_exc(file=sys.stderr)


# --- Collate Function for DataLoader ---
def collate_fn(batch):
    indices = []
    offsets = [0]
    outcomes = []
    for features_list, outcome in batch:
        np_features = np.array(features_list, dtype=np.int64)
        if np_features.size > 0:
             indices.extend(np_features.tolist())
        offsets.append(offsets[-1] + len(np_features))
        outcomes.append(outcome)

    offsets.pop()

    if not indices:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float32).unsqueeze(1)

    batch_indices = torch.tensor(indices, dtype=torch.long)
    batch_offsets = torch.tensor(offsets, dtype=torch.long)
    batch_outcomes = torch.tensor(outcomes, dtype=torch.float32).unsqueeze(1)
    return batch_indices, batch_offsets, batch_outcomes


# --- Training Function ---
def train_model(input_path, data_type, model_save_path, epochs=10, batch_size=1024, lr=0.001, num_data_workers=4, pgn_min_ply=DEFAULT_MIN_PLY):
    """
    Main training loop. Handles HDF5 or PGN data.

    Args:
        input_path (str): Path to HDF5 directory OR PGN/ZST file/directory.
        data_type (str): 'hdf5' or 'pgn'.
        model_save_path (str): Path to save the trained model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        num_data_workers (int): Number of workers for DataLoader.
        pgn_min_ply (int): Min ply for PGN processing (used only if data_type='pgn').
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = None
    train_loader = None
    val_loader = None

    # --- Load Data based on type ---
    if data_type == 'hdf5':
        print(f"Streaming data from HDF5 files in: {input_path}")
        if not os.path.isdir(input_path):
             print(f"Error: Input path '{input_path}' is not a directory for HDF5 data.")
             return
        hdf5_files = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith((".hdf5", ".h5"))])
        if not hdf5_files:
            print(f"Error: No .hdf5 or .h5 files found in {input_path}")
            return

        dataset = IterableHdf5Dataset(hdf5_files)
        train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_data_workers, pin_memory=True)

    elif data_type == 'pgn':
        print(f"Streaming data from PGN/ZST path: {input_path}")
        pgn_files = []
        if os.path.isdir(input_path):
             pgn_files = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith((".pgn", ".pgn.zst"))])
        elif os.path.isfile(input_path) and input_path.lower().endswith((".pgn", ".pgn.zst")):
             pgn_files = [input_path]

        if not pgn_files:
             print(f"Error: No PGN or PGN.ZST files found at path: {input_path}")
             return

        dataset = IterablePgnDataset(pgn_files, min_ply=pgn_min_ply)
        train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_data_workers, pin_memory=True)

    else:
        print(f"Error: Invalid data_type '{data_type}'. Choose 'hdf5' or 'pgn'.")
        return

    # --- Initialize Model, Loss, Optimizer ---
    model = NNUE().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) # Example scheduler

    # --- Training Loop ---
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        processed_samples = 0
        processed_batches = 0

        # Wrap train_loader with tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_indices, batch_offsets, batch_outcomes in pbar:
            if batch_indices.numel() == 0:
                 continue

            batch_indices = batch_indices.to(device)
            batch_offsets = batch_offsets.to(device)
            batch_outcomes = batch_outcomes.to(device)

            optimizer.zero_grad()
            outputs = model(batch_indices, batch_offsets)
            loss = criterion(outputs, batch_outcomes)
            loss.backward()
            optimizer.step()

            current_batch_size = batch_outcomes.size(0)
            running_loss += loss.item() * current_batch_size
            processed_samples += current_batch_size
            processed_batches += 1

            if processed_samples > 0:
                 pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{running_loss/processed_samples:.4f}"})

        epoch_loss = running_loss / processed_samples if processed_samples > 0 else 0
        epoch_time = time.time() - start_time
        print(f'\nEpoch [{epoch+1}/{epochs}] completed in {epoch_time:.2f}s - Training Loss: {epoch_loss:.4f}')
        sys.stdout.flush()

        print(f'Saving model after epoch {epoch+1}...')
        sys.stdout.flush()
        torch.save(model.state_dict(), model_save_path)

        scheduler.step()
        print("-" * 30)
        sys.stdout.flush()

    print('Finished Training')


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NNUE model using data from HDF5 or PGN files.")
    # Modified arguments
    parser.add_argument("input_path", help="Path to the input directory (HDF5) or file/directory (PGN/ZST).")
    parser.add_argument("model_save_path", help="Path to save the trained PyTorch model (.pt or .pth).")
    parser.add_argument("--data_type", choices=['hdf5', 'pgn'], required=True, help="Type of input data ('hdf5' or 'pgn').")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20).")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size (default: 4096).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001).")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers (default: 4).")
    parser.add_argument("--min_ply", type=int, default=DEFAULT_MIN_PLY, help=f"Minimum ply for PGN positions (PGN mode only) (default: {DEFAULT_MIN_PLY}).")

    args = parser.parse_args()

    # Basic input validation
    if args.data_type == 'hdf5' and not os.path.isdir(args.input_path):
         print(f"Error: For data_type 'hdf5', input_path must be a directory: {args.input_path}")
         sys.exit(1)
    elif args.data_type == 'pgn' and not os.path.exists(args.input_path):
         print(f"Error: Input path not found for data_type 'pgn': {args.input_path}")
         sys.exit(1)

    # Ensure model save directory exists
    save_dir = os.path.dirname(args.model_save_path)
    if save_dir and not os.path.exists(save_dir):
         os.makedirs(save_dir)

    train_model(
        input_path=args.input_path,
        data_type=args.data_type,
        model_save_path=args.model_save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_data_workers=args.num_workers,
        pgn_min_ply=args.min_ply
    )