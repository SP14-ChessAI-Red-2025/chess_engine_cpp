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

import chess
import chess.pgn
import zstandard as zstd
from tqdm import tqdm
import h5py

TOTAL_FEATURES = 768
FEATURES_PER_COLOR = TOTAL_FEATURES // 2
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
PIECE_TO_INDEX = {piece_type: i for i, piece_type in enumerate(PIECE_TYPES)}
DEFAULT_MIN_PLY = 8

# --- NNUE Model Definition ---
class NNUE(nn.Module):
    def __init__(self, feature_dim=TOTAL_FEATURES, embed_dim=128, hidden_dim=256):
        super().__init__()
        # Use sparse=False as ONNX export might handle sparse better this way
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

# --- Feature Extraction (for PGN processing) ---
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
        return [] # Invalid position if a king is missing
    return indices

# --- PGN Iterable Dataset ---
class IterablePgnDataset(IterableDataset):
    def __init__(self, pgn_files: list[str], min_ply: int):
        """
        PyTorch IterableDataset for streaming data directly from PGN/ZST files.
        """
        super().__init__()
        self.pgn_files = pgn_files
        self.min_ply = min_ply
        if not self.pgn_files:
            raise ValueError("No PGN files provided to IterablePgnDataset.")
        print(f"IterablePgnDataset initialized with {len(self.pgn_files)} PGN/ZST files.")

    def _parse_stream(self, pgn_stream):
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
                else: continue # Skip games with unknown results

                for node in game.mainline():
                    parent_node = node.parent
                    if parent_node is None: continue

                    current_ply = parent_node.ply()
                    if current_ply >= self.min_ply:
                        board = parent_node.board()
                        features = extract_features(board)
                        if features: # Ensure features were extracted
                            yield features, outcome

            except (ValueError, KeyError, IndexError, AttributeError, chess.IllegalMoveError, chess.InvalidMoveError) as e:
                 # Log errors less frequently to avoid spamming console
                 if game_count % 1000 == 0:
                      print(f"Warning: Error processing game #{game_count} in PGN stream: {e}", file=sys.stderr)
                 continue
            except Exception as e:
                 print(f"Warning: Unexpected error processing game #{game_count} in PGN stream: {e}", file=sys.stderr)
                 traceback.print_exc(file=sys.stderr)
                 continue

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files_to_process = self.pgn_files

        if worker_info is not None: # Distribute files among workers
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
            finally: # Ensure resources are closed
                 if pgn_stream is not None:
                     try: pgn_stream.close()
                     except Exception: pass
                 if reader is not None:
                     try: reader.close()
                     except Exception: pass
                 if compressed_file is not None:
                     try: compressed_file.close()
                     except Exception: pass

# --- HDF5 Iterable Dataset ---
class IterableHdf5Dataset(IterableDataset):
    def __init__(self, hdf5_files: list[str]):
        """
        PyTorch IterableDataset for streaming data from multiple HDF5 files.
        """
        super().__init__()
        self.hdf5_files = hdf5_files
        if not self.hdf5_files:
            raise ValueError("No HDF5 files provided to IterableHdf5Dataset.")
        print(f"IterableHdf5Dataset initialized with {len(self.hdf5_files)} HDF5 files.")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files_to_process = self.hdf5_files

        if worker_info is not None: # Distribute files
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            files_per_worker = int(math.ceil(len(self.hdf5_files) / float(num_workers)))
            start_idx = worker_id * files_per_worker
            end_idx = min(start_idx + files_per_worker, len(self.hdf5_files))
            files_to_process = self.hdf5_files[start_idx:end_idx]

        for hdf5_path in files_to_process:
            try:
                with h5py.File(hdf5_path, 'r') as hf:
                    features_dset = hf['features']
                    outcomes_dset = hf['outcomes']
                    num_samples_in_file = len(outcomes_dset)

                    for idx in range(num_samples_in_file):
                        # Assume features are stored as variable-length integers
                        features = features_dset[idx]
                        outcome = outcomes_dset[idx]
                        # Yield directly, collation handles tensor conversion
                        yield features.tolist(), float(outcome)

            except KeyError as e:
                 print(f"Error: Missing dataset '{e}' in file {hdf5_path}. Skipping file.", file=sys.stderr)
            except Exception as e:
                 print(f"Error opening/reading file {hdf5_path}: {e}", file=sys.stderr)
                 traceback.print_exc(file=sys.stderr)

# --- Collate Function ---
def collate_fn(batch):
    """
    Prepares batches for the EmbeddingBag layer.
    Input: list of (features_list, outcome) tuples.
    Output: batch_indices, batch_offsets, batch_outcomes tensors.
    """
    indices = []
    offsets = [0]
    outcomes = []
    for features_list, outcome in batch:
        # Ensure features_list contains integers
        np_features = np.array(features_list, dtype=np.int64)
        if np_features.size > 0:
             indices.extend(np_features.tolist()) # Flatten feature lists into one list
        offsets.append(offsets[-1] + len(np_features)) # Calculate offset for next item
        outcomes.append(outcome)

    offsets.pop() # Remove the last offset which is the total count

    # Handle empty batches gracefully
    if not indices:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float32).unsqueeze(1)

    batch_indices = torch.tensor(indices, dtype=torch.long)
    batch_offsets = torch.tensor(offsets, dtype=torch.long)
    batch_outcomes = torch.tensor(outcomes, dtype=torch.float32).unsqueeze(1) # Ensure shape [batch_size, 1]
    return batch_indices, batch_offsets, batch_outcomes


# --- Training ---
def train_model(input_path, data_type, model_save_path, epochs=10, batch_size=1024, lr=0.001, num_data_workers=4, pgn_min_ply=DEFAULT_MIN_PLY):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = None
    train_loader = None

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

    else:
        print(f"Error: Invalid data_type '{data_type}'. Choose 'hdf5' or 'pgn'.")
        return

    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_data_workers, pin_memory=True if device.type == 'cuda' else False)

    model = NNUE().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        processed_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for batch_indices, batch_offsets, batch_outcomes in pbar:
            # Skip empty batches that might result from filtering in datasets
            if batch_indices.numel() == 0 or batch_outcomes.numel() == 0:
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

            if processed_samples > 0:
                 # Display current batch loss and running average loss
                 pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{running_loss/processed_samples:.4f}"})

        epoch_loss = running_loss / processed_samples if processed_samples > 0 else 0
        epoch_time = time.time() - start_time
        print(f'\nEpoch [{epoch+1}/{epochs}] completed in {epoch_time:.2f}s - Training Loss: {epoch_loss:.4f}')
        sys.stdout.flush()

        print(f'Saving model after epoch {epoch+1}...')
        sys.stdout.flush()
        torch.save(model.state_dict(), model_save_path)

        scheduler.step() # Adjust learning rate
        print("-" * 30)
        sys.stdout.flush()

    print('Finished Training')


# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NNUE model using data from HDF5 or PGN files.")
    parser.add_argument("input_path", help="Path to the input directory (HDF5) or file/directory (PGN/ZST).")
    parser.add_argument("model_save_path", help="Path to save the trained PyTorch model (.pt or .pth).")
    parser.add_argument("--data_type", choices=['hdf5', 'pgn'], required=True, help="Type of input data ('hdf5' or 'pgn').")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20).")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size (default: 4096).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001).")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers (default: 4).")
    parser.add_argument("--min_ply", type=int, default=DEFAULT_MIN_PLY, help=f"Minimum ply for PGN positions (PGN mode only) (default: {DEFAULT_MIN_PLY}).")

    args = parser.parse_args()

    if args.data_type == 'hdf5' and not os.path.isdir(args.input_path):
         print(f"Error: For data_type 'hdf5', input_path must be a directory: {args.input_path}")
         sys.exit(1)
    elif args.data_type == 'pgn' and not os.path.exists(args.input_path):
         print(f"Error: Input path not found for data_type 'pgn': {args.input_path}")
         sys.exit(1)

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