import math
import traceback
import random

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
import json
import re

from cython_feature_extractor import board_to_features_cython

TOTAL_FEATURES = 768
FEATURES_PER_COLOR = TOTAL_FEATURES // 2
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
PIECE_TO_INDEX = {piece_type: i for i, piece_type in enumerate(PIECE_TYPES)}
DEFAULT_MIN_PLY = 8

# --- NNUE Model Definition ---
class NNUE(nn.Module):
    def __init__(self, feature_dim=TOTAL_FEATURES, embed_dim=256, hidden_dim=32, dropout_prob=0.4):
        """
        NNUE Model Architecture with Dropout:
        - Input Layer: EmbeddingBag for sparse input features.
        - Hidden Layers: Fully connected layers with ReLU activation and Dropout.
        - Output Layer: Single scalar output for evaluation.
        """
        super().__init__()
        # Input layer: EmbeddingBag to process sparse features
        self.input_layer = nn.EmbeddingBag(feature_dim, embed_dim, mode="sum", sparse=False)

        # First hidden layer
        self.hidden1 = nn.Linear(embed_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_prob)  # Dropout after first hidden layer

        # Second hidden layer
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_prob)  # Dropout after second hidden layer

        # Output layer
        self.output = nn.Linear(hidden_dim // 2, 1)

    def forward(self, features_indices, offsets=None):
        x = self.input_layer(features_indices, offsets=offsets)
        x = self.relu1(self.hidden1(x))
        x = self.dropout1(x)
        x = self.relu2(self.hidden2(x))
        x = self.dropout2(x)
        x = self.output(x)
        return x
    
# --- Feature Extraction (for PGN processing) ---
def extract_features(board: chess.Board) -> list[int]:
    """
    Extracts feature indices for a simple 768-feature NNUE input layer.
    """
    features = board_to_features_cython(board)
    return features

# --- PGN Iterable Dataset ---
class IterablePgnDataset(IterableDataset):
    def __init__(self, pgn_files: list[str], min_ply: int, is_validation: bool = False, sample_rate=1.0):
        """
        PyTorch IterableDataset for streaming data directly from PGN/ZST files.
        :param pgn_files: List of PGN file paths.
        :param min_ply: Minimum ply for positions to include.
        :param is_validation: If True, all workers process the validation set.
        :param sample_rate: Fraction of data to include (default: 1.0, i.e., no sampling).
        """
        super().__init__()
        self.pgn_files = pgn_files
        self.min_ply = min_ply
        self.is_validation = is_validation
        self.sample_rate = sample_rate
        if not self.pgn_files:
            raise ValueError("No PGN files provided to IterablePgnDataset.")
        print(f"IterablePgnDataset initialized with {len(self.pgn_files)} PGN/ZST files.")

    def _parse_stream(self, pgn_stream):
        game_count = 0
        eval_pattern = re.compile(r"Stockfish eval: ([+-]?\d+(\.\d+)?) \(centipawns\)")

        while True:
            game = None
            try:
                game = chess.pgn.read_game(pgn_stream)
                if game is None:
                    break

                game_count += 1
                result = game.headers.get("Result", "*")
                if result not in ["1-0", "0-1", "1/2-1/2"]:
                    continue  # Skip games with unknown results

                outcome = 1.0 if result == "1-0" else 0.0 if result == "0-1" else 0.5

                for node in game.mainline():
                    parent_node = node.parent
                    if parent_node is None:
                        continue

                    current_ply = parent_node.ply()
                    if current_ply >= self.min_ply:
                        board = parent_node.board()
                        features = extract_features(board)

                        # Extract Stockfish evaluation from the comment
                        comment = node.comment
                        match = eval_pattern.search(comment)
                        if match:
                            evaluation = float(match.group(1)) / 100.0  # Convert centipawns to float
                            if features and random.random() <= self.sample_rate:  # Apply sampling
                                yield features, evaluation

            except Exception as e:
                print(f"Warning: Error processing game #{game_count} in PGN stream: {e}", file=sys.stderr)
                continue

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files_to_process = self.pgn_files

        if worker_info is not None and not self.is_validation:  # Distribute files among workers during training
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
            finally:  # Ensure resources are closed
                if pgn_stream is not None:
                    try:
                        pgn_stream.close()
                    except Exception:
                        pass
                if reader is not None:
                    try:
                        reader.close()
                    except Exception:
                        pass
                if compressed_file is not None:
                    try:
                        compressed_file.close()
                    except Exception:
                        pass

# --- HDF5 Iterable Dataset ---
class IterableHdf5Dataset(IterableDataset):
    def __init__(self, hdf5_files: list[str], is_validation: bool = False, sample_rate=1.0, split_ratio=0.9):
        """
        PyTorch IterableDataset for streaming data from multiple HDF5 files.
        :param hdf5_files: List of HDF5 file paths.
        :param is_validation: If True, this dataset is for validation.
        :param sample_rate: Fraction of data to include (default: 1.0, i.e., no sampling).
        :param split_ratio: Ratio of data to use for training (default: 0.9 for training, 0.1 for validation).
        """
        super().__init__()
        self.hdf5_files = hdf5_files
        self.is_validation = is_validation
        self.sample_rate = sample_rate
        self.split_ratio = split_ratio
        if not self.hdf5_files:
            raise ValueError("No HDF5 files provided to IterableHdf5Dataset.")
        print(f"IterableHdf5Dataset initialized with {len(self.hdf5_files)} HDF5 files.")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files_to_process = self.hdf5_files

        if worker_info is not None and not self.is_validation:  # Distribute files among workers during training
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
                        # Determine if this sample belongs to training or validation
                        if (hash(idx) % 100 < self.split_ratio * 100) != self.is_validation:
                            # Apply sampling
                            if random.random() <= self.sample_rate:
                                features = features_dset[idx]
                                outcome = outcomes_dset[idx]
                                yield features.tolist(), float(outcome)

            except KeyError as e:
                print(f"Error: Missing dataset '{e}' in file {hdf5_path}. Skipping file.", file=sys.stderr)
            except Exception as e:
                print(f"Error opening/reading file {hdf5_path}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

# --- HDF5 Dataset ---
class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path: str):
        """
        PyTorch Dataset for loading data from an HDF5 file.
        :param hdf5_path: Path to the HDF5 file.
        """
        self.hdf5_path = hdf5_path
        with h5py.File(hdf5_path, "r") as hf:
            self.num_samples = len(hf["evaluations"])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, "r") as hf:
            features = hf["features"][idx]
            evaluation = hf["evaluations"][idx]
        return torch.tensor(features, dtype=torch.long), torch.tensor(evaluation, dtype=torch.float32)

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
def train_nnue_from_hdf5(hdf5_path: str, model_save_path: str, epochs: int = 10, batch_size: int = 1024, lr: float = 0.001):
    """
    Train the NNUE model using data from an HDF5 file.
    :param hdf5_path: Path to the HDF5 file.
    :param model_save_path: Path to save the trained model.
    :param epochs: Number of training epochs.
    :param batch_size: Batch size.
    :param lr: Learning rate.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HDF5Dataset(hdf5_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = NNUE().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for features, evaluations in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            features = features.to(device)
            evaluations = evaluations.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, evaluations)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Move the JsonDataset class definition here
class JsonDataset(Dataset):
    def __init__(self, json_path):
        """
        Dataset for training directly from a JSON file.
        :param json_path: Path to the JSON file containing evaluations.
        """
        self.data = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                fen = entry["fen"]
                evals = entry.get("evals", [])
                if evals:
                    # Use the first evaluation and its first principal variation
                    best_eval = evals[0]
                    pvs = best_eval.get("pvs", [])
                    if pvs:
                        cp = pvs[0].get("cp")
                        mate = pvs[0].get("mate")
                        if cp is not None:
                            evaluation = cp / 100.0  # Convert centipawns to a float
                        elif mate is not None:
                            evaluation = 10000 if mate > 0 else -10000  # Large value for mate
                        else:
                            continue
                        self.data.append((fen, evaluation))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, evaluation = self.data[idx]
        board = chess.Board(fen)
        features = self.extract_features(board)
        return torch.tensor(features, dtype=torch.long), torch.tensor(evaluation, dtype=torch.float32)

    @staticmethod
    def extract_features(board):
        """
        Extracts feature indices for a simple 768-feature NNUE input layer.
        """
        TOTAL_FEATURES = 768
        FEATURES_PER_COLOR = TOTAL_FEATURES // 2
        PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        PIECE_TO_INDEX = {piece_type: i for i, piece_type in enumerate(PIECE_TYPES)}

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
            return []  # Invalid position if a king is missing
        return indices

# Define IterableJsonDataset before train_model
class IterableJsonDataset(IterableDataset):
    def __init__(self, json_path, sample_rate=1.0, is_validation=False, split_ratio=0.9):
        """
        PyTorch IterableDataset for streaming data directly from a large JSON file.
        Includes train/validation split logic.
        :param json_path: Path to the JSON file containing evaluations.
        :param sample_rate: Fraction of data to include (default: 1.0, i.e., no sampling).
        :param is_validation: If True, yield samples for the validation set.
        :param split_ratio: Ratio of data to use for training (0.0 to 1.0).
        """
        super().__init__()
        self.json_path = json_path
        self.sample_rate = sample_rate
        self.is_validation = is_validation
        self.split_ratio = split_ratio
        if not 0 <= split_ratio <= 1:
            raise ValueError("split_ratio must be between 0 and 1")
        print(f"IterableJsonDataset initialized for {'validation' if is_validation else 'training'}.")

    def parse_json_line(self, line):
        """
        Parse a single line of JSON and extract features and evaluation.
        Applies clipping.
        :param line: A JSON string representing a single position.
        :return: (features, evaluation) tuple or None if parsing/extraction fails.
        """
        try:
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
                        evaluation = 10.0 if mate > 0 else -10.0
                    else:
                        return None

                    # Clip evaluation
                    if abs(evaluation) > 10.0:
                        evaluation = 10.0 if evaluation > 0 else -10.0

                    board = chess.Board(fen)
                    features = self.extract_features(board)
                    if features:
                        return features, evaluation
            return None
        except (json.JSONDecodeError, KeyError, IndexError):
            return None

    def __iter__(self):
        """
        Iterate over the JSON file line by line, applying sampling and train/val split.
        """
        line_number = 0
        with open(self.json_path, "r", encoding="utf-8") as f:
            for line in f:
                belongs_to_train = (hash(line_number) % 100 < self.split_ratio * 100)
                line_number += 1

                if belongs_to_train == self.is_validation:
                    continue

                if random.random() <= self.sample_rate:
                    result = self.parse_json_line(line)
                    if result:
                        yield result

    @staticmethod
    def extract_features(board):
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
            return []  # Invalid position if a king is missing
        return indices

# Ensure train_model is defined after IterableJsonDataset
def train_model(input_path, model_save_path, train_data_type, load_model_path=None, epochs=10, batch_size=1024, lr=0.001, num_data_workers=12, sample_rate=1.0):
    """
    Main training loop with training and validation datasets derived from the same data type.
    :param input_path: Path to the training dataset (JSON or HDF5).
    :param model_save_path: Path to save the trained model.
    :param train_data_type: Data type for training ('json' or 'hdf5').
    :param load_model_path: Path to load a pre-trained model (optional).
    :param epochs: Number of training epochs.
    :param batch_size: Batch size.
    :param lr: Learning rate.
    :param num_data_workers: Number of workers for DataLoader.
    :param sample_rate: Fraction of data to include (default: 1.0, i.e., no sampling).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize datasets 
    train_dataset = None
    val_dataset = None

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 5
    best_model_path = model_save_path.replace('.pt', '_best.pt').replace('.pth', '_best.pth')

    # Training and validation datasets
    if train_data_type == 'json':
        print(f"Loading training data from JSON file: {input_path}")
        train_dataset = IterableJsonDataset(input_path, sample_rate=sample_rate)
        val_dataset = IterableJsonDataset(input_path, sample_rate=sample_rate, is_validation=True, split_ratio=0.9)  # Use same file for validation with different split
    elif train_data_type == 'hdf5':
        print(f"Loading training data from HDF5 files in: {input_path}")
        hdf5_files = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith((".hdf5", ".h5"))])
        if not hdf5_files:
            print(f"Error: No .hdf5 or .h5 files found in {input_path}")
            return
        # Training dataset (90%)

        train_dataset = IterableHdf5Dataset(
            hdf5_files=hdf5_files,
            is_validation=False,
            sample_rate=sample_rate,
            split_ratio=0.9
        )

        # Validation dataset (10%)
        val_dataset = IterableHdf5Dataset(
            hdf5_files=hdf5_files,
            is_validation=True,
            sample_rate=sample_rate,
            split_ratio=0.9
        )
    else:
        print(f"Error: Unsupported training data type '{train_data_type}'.")
        return

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_data_workers, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_data_workers, pin_memory=True if device.type == 'cuda' else False)

    # Initialize model, optimizer, and scheduler
    model = NNUE().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Load model weights if path is provided
    if load_model_path:
        if os.path.exists(load_model_path):
            print(f"Loading model state from: {load_model_path}")
            try:
                model.load_state_dict(torch.load(load_model_path, map_location=device))
                print("Model weights loaded successfully.")
            except Exception as e:
                print(f"Error loading model weights from {load_model_path}: {e}")
                print("Proceeding with freshly initialized model.")
        else:
            print(f"Warning: Load model path specified, but file not found: {load_model_path}")
            print("Starting training from scratch.")
    else:
        print("No load model path provided. Starting training from scratch.")

    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        processed_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for batch_indices, batch_offsets, batch_outcomes in pbar:
            if batch_indices.numel() == 0 or batch_outcomes.numel() == 0:
                continue

            batch_indices = batch_indices.to(device)
            batch_offsets = batch_offsets.to(device)
            batch_outcomes = batch_outcomes.to(device)

            optimizer.zero_grad()
            outputs = model(batch_indices, batch_offsets)
            loss = criterion(outputs, batch_outcomes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            current_batch_size = batch_outcomes.size(0)
            running_loss += loss.item() * current_batch_size
            processed_samples += current_batch_size

            if processed_samples > 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{running_loss/processed_samples:.4f}"})

        epoch_loss = running_loss / processed_samples if processed_samples > 0 else 0
        epoch_time = time.time() - start_time
        print(f'\nEpoch [{epoch+1}/{epochs}] completed in {epoch_time:.2f}s - Training Loss: {epoch_loss:.4f}')

        # Validation loop
        if val_loader is not None:  # Check if the validation loader exists
            model.eval()
            val_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for batch_indices, batch_offsets, batch_outcomes in val_loader:
                    if batch_indices.numel() == 0 or batch_outcomes.numel() == 0:
                        continue

                    batch_indices = batch_indices.to(device)
                    batch_offsets = batch_offsets.to(device)
                    batch_outcomes = batch_outcomes.to(device)

                    outputs = model(batch_indices, batch_offsets)
                    loss = criterion(outputs, batch_outcomes)

                    current_batch_size = batch_outcomes.size(0)
                    val_loss += loss.item() * current_batch_size
                    val_samples += current_batch_size

            avg_val_loss = val_loss / val_samples if val_samples > 0 else 0
            print(f'Epoch [{epoch+1}/{epochs}] - Validation Loss: {avg_val_loss:.4f}')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # Save the best model specifically
                best_model_path = model_save_path.replace('.pt', '_best.pt')
                torch.save(model.state_dict(), best_model_path)
                print(f"New best validation loss: {best_val_loss:.4f}. Saved best model to {best_model_path}")
            else:
                epochs_no_improve += 1
                print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                print(f"Stopping early after {patience} epochs without validation improvement.")
                break # Exit the epoch loop

            # Save model after each epoch
            print(f'Saving model after epoch {epoch+1}...')
            torch.save(model.state_dict(), model_save_path)

            scheduler.step(avg_val_loss)  # Adjust learning rate
            print("-" * 30)

            print('Finished Training')

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NNUE model using training and validation datasets.")
    parser.add_argument("input_path", help="Path to the training dataset (JSON or HDF5).")
    parser.add_argument("model_save_path", help="Path to save the trained PyTorch model (.pt or .pth).")
    parser.add_argument("--load_model_path", type=str, default=None, help="Path to a previously saved PyTorch model (.pt or .pth) to load weights from (optional).")
    parser.add_argument("--train_data_type", choices=['json', 'hdf5'], required=True, help="Type of training data ('json' or 'hdf5').")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20).")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size (default: 2048).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001).")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers (default: 4).")
    parser.add_argument("--sample_rate", type=float, default=1.0, help="Fraction of data to include (default: 1.0, i.e., no sampling).")

    args = parser.parse_args()

    train_model(
        input_path=args.input_path,
        model_save_path=args.model_save_path,
        train_data_type=args.train_data_type,
        load_model_path=args.load_model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_data_workers=args.num_workers,
        sample_rate=args.sample_rate
    )
