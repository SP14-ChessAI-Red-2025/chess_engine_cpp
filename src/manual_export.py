# python manual_export.py ../model/trained_nnue.pt ../model/trained_nnue.onnx

import torch
import torch.nn as nn
import torch.onnx
from torch import Tensor
import onnx
import onnxruntime as ort
import numpy as np
import argparse
import os
import sys

# --- Define the NNUE model structure ---
# This class implements the architecture provided in the reference code block:
# Input(feature_dim) -> Embed(embed_dim) -> Linear(hidden_dim) -> ReLU -> Dropout ->
# -> Linear(hidden_dim // 2) -> ReLU -> Dropout -> Linear(1)
class NNUE(nn.Module):
    # Parameters based on the reference structure:
    # feature_dim: Input feature size (e.g., 768)
    # embed_dim: Output size of EmbeddingBag (e.g., 256)
    # hidden_dim: Output size of the first hidden layer (e.g., 32)
    # Layer sizes: 768 -> 256 -> 32 -> 16 -> 1
    def __init__(self, feature_dim: int = 768, embed_dim: int = 256, hidden1_dim: int = 32, dropout_prob: float = 0.4):
        """
        Initializes the NNUE model layers based on the reference architecture.

        Args:
            feature_dim (int): The total number of unique input features. Default: 768.
            embed_dim (int): The dimension of the embedding layer. Default: 256.
            hidden_dim (int): The dimension of the first hidden layer. The second hidden
                              layer will have dimension hidden_dim // 2. Default: 32.
            dropout_prob (float): The dropout probability used after ReLU activations
                                  in hidden layers. Default: 0.4.
        """
        super(NNUE, self).__init__()

        hidden2_dim = hidden1_dim // 2 # Calculate the size of the second hidden layer

        # Layer 1: Input (EmbeddingBag)
        # Input: feature_dim, Output: embed_dim
        self.input_layer = nn.EmbeddingBag(feature_dim, embed_dim, mode='sum', sparse=False)

        # Layer 2: Hidden Linear 1 + Activation + Dropout
        # Input: embed_dim, Output: hidden_dim
        self.hidden1 = nn.Linear(embed_dim, hidden1_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_prob)

        # Layer 3: Hidden Linear 2 + Activation + Dropout
        # Input: hidden_dim, Output: hidden2_dim (hidden_dim // 2)
        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_prob)

        # Layer 4: Output Linear
        # Input: hidden2_dim, Output: 1
        self.output = nn.Linear(hidden2_dim, 1)

        print(f"Initialized NNUE with structure: Embed({feature_dim}>{embed_dim}) -> "
              f"Linear({embed_dim}>{hidden1_dim}) -> ReLU -> Dropout({dropout_prob}) -> "
              f"Linear({hidden1_dim}>{hidden2_dim}) -> ReLU -> Dropout({dropout_prob}) -> "
              f"Linear({hidden2_dim}>1)")

    def forward(self, feature_indices: Tensor, offsets: Tensor = None) -> Tensor:
        """
        Defines the forward pass of the NNUE model.

        Args:
            feature_indices (Tensor): A 1D tensor containing the indices of all active features
                                      across the entire batch, concatenated together.
            offsets (Tensor, optional): A 1D tensor indicating the starting index in
                                         `feature_indices` for each sample in the batch.
                                         Starts with 0. Required if batch size > 1.
                                         Defaults to None, assuming a single sample if omitted.

        Returns:
            Tensor: The output of the network (e.g., evaluation score) for each sample in the batch.
                    Shape: (batch_size, 1)
        """
        # 1. Input processing
        x = self.input_layer(feature_indices, offsets=offsets) # Shape: (batch_size, embed_dim)

        # 2. First hidden layer block
        x = self.hidden1(x)
        x = self.relu1(x)
        x = self.dropout1(x) # Shape: (batch_size, hidden_dim)

        # 3. Second hidden layer block
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.dropout2(x) # Shape: (batch_size, hidden_dim // 2)

        # 4. Output layer
        x = self.output(x) # Shape: (batch_size, 1)

        return x

# --- Main Conversion Function ---
# Updated function signature to accept new hidden dimensions
def convert_pt_to_onnx(pt_path, onnx_path, input_size, embed_dim, hidden1_dim, hidden2_dim, opset_version=11, verify=True):
    """
    Loads a PyTorch state_dict from a .pt file, exports the model to ONNX,
    and optionally verifies the ONNX model. Uses the updated NNUE structure.

    Args:
        pt_path (str): Path to the input PyTorch state_dict (.pt) file.
        onnx_path (str): Path to save the output ONNX (.onnx) file.
        input_size (int): Input dimension (vocabulary size).
        embed_dim (int): Embedding dimension (output of first layer).
        hidden1_dim (int): Output dimension of the first hidden layer.
        hidden2_dim (int): Output dimension of the second hidden layer.
        opset_version (int): ONNX opset version to use for export.
        verify (bool): Whether to perform ONNX model checking and runtime verification.
    """
    print(f"Starting conversion:")
    print(f"  Input PyTorch model (.pt): {pt_path}")
    print(f"  Output ONNX model (.onnx): {onnx_path}")
    print(f"  Model Params: input_size={input_size}, embed_dim={embed_dim}, hidden1_dim={hidden1_dim}, hidden2_dim={hidden2_dim}")
    print(f"  ONNX Opset Version: {opset_version}")

    # --- Input Validation ---
    if not os.path.exists(pt_path):
        print(f"[ERROR] Input PyTorch file not found: {pt_path}")
        sys.exit(1)

    output_dir = os.path.dirname(onnx_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"[ERROR] Could not create output directory '{output_dir}': {e}")
            sys.exit(1)

    try:
        # --- Prepare Model for Export ---
        # 1. Instantiate the *updated* model structure
        model = NNUE(feature_dim=input_size, embed_dim=embed_dim, hidden1_dim=hidden1_dim, dropout_prob=0.4)

        # 2. Load the state dictionary from the .pt file
        print("Loading state dictionary...")
        state_dict = torch.load(pt_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print("State dictionary loaded successfully.")

        # 3. Set the model to evaluation mode
        model.eval()
        # 4. Move model to CPU for export consistency
        model.to('cpu')
        print("Model set to evaluation mode on CPU.")

        # --- Create Dummy Input ---
        # Create dummy input matching the model's forward signature (indices, offsets)
        dummy_num_features = 25  # Example number of active features
        dummy_indices = torch.randint(0, input_size, (dummy_num_features,), dtype=torch.long)
        dummy_offsets = torch.tensor([0], dtype=torch.long)  # Batch size 1
        dummy_input_tuple = (dummy_indices, dummy_offsets)  # Input must be a tuple

        print("Using Dummy Input Shapes for ONNX Export:")
        print(f"  Indices: {dummy_input_tuple[0].shape}")
        print(f"  Offsets: {dummy_input_tuple[1].shape}")

        # --- Perform ONNX Export ---
        print(f"Exporting to ONNX (opset {opset_version})...")
        torch.onnx.export(
            model,                      # model being run
            dummy_input_tuple,          # model input (tuple)
            onnx_path,                  # where to save the model
            export_params=True,         # store weights in the model file
            opset_version=opset_version,  # ONNX version
            do_constant_folding=True,   # optimize constants
            input_names=['feature_indices', 'offsets'],  # input names
            output_names=['evaluation'],  # output names
            dynamic_axes={'feature_indices': {0: 'num_total_features'},  # variable length axes
                          'offsets': {0: 'batch_size'},
                          'evaluation': {0: 'batch_size'}}
        )
        print(f"ONNX export successful to: {onnx_path}")

        # --- Optional Verification ---
        if verify:
            print("\n--- Starting ONNX Verification ---")
            # 1. Check ONNX model structure
            print("Verifying ONNX model structure...")
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model check passed structural validity.")

            # 2. Test Inference with ONNX Runtime
            print("Testing inference with ONNX Runtime...")
            providers = ['CPUExecutionProvider'] # Default to CPU
            print(f"Using ONNX Runtime providers: {providers}")

            ort_session = ort.InferenceSession(onnx_path, providers=providers)

            # Prepare input for ONNX Runtime (numpy arrays)
            ort_inputs = {
                ort_session.get_inputs()[0].name: dummy_input_tuple[0].numpy(),
                ort_session.get_inputs()[1].name: dummy_input_tuple[1].numpy()
            }
            ort_outputs = ort_session.run(None, ort_inputs)
            print(f"ONNX Runtime inference test successful. Output shape: {ort_outputs[0].shape}")

            # 3. Compare ONNX Runtime output with PyTorch output
            print("Comparing ONNX Runtime output with PyTorch output...")
            with torch.no_grad():
                pytorch_output = model(*dummy_input_tuple) # Run PyTorch model with same dummy input
                np.testing.assert_allclose(pytorch_output.numpy(), ort_outputs[0], rtol=1e-03, atol=1e-05)
                print("ONNX Runtime and PyTorch outputs match within tolerance!")

            print("--- ONNX Verification Complete ---")

    except FileNotFoundError:
        print(f"[ERROR] Input file not found: {pt_path}")
        sys.exit(1)
    except KeyError as e:
         print(f"[ERROR] Mismatched keys loading state_dict (model structure might be wrong): {e}")
         sys.exit(1)
    except RuntimeError as e:
        print(f"[ERROR] PyTorch runtime error during load/export: {e}")
        sys.exit(1)
    except onnx.checker.ValidationError as e:
        print(f"[ERROR] ONNX model failed validation: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manually convert a PyTorch NNUE model (.pt state_dict) to ONNX format.")

    parser.add_argument("pt_input_path", type=str, default="../model/trained_nnue.pt",
                        help="Path to the input PyTorch state_dict file (.pt).")
    parser.add_argument("onnx_output_path", type=str, default="../model/trained_nnue.onnx",
                        help="Path to save the output ONNX model file (.onnx).")

    # Updated arguments to match the new NNUE structure based on the image
    # Layer sizes: 768 -> 256 -> 32 -> 16 -> 1
    parser.add_argument("--input_size", type=int, default=768,
                        help="Input dimension (feature vocabulary size).")
    parser.add_argument("--embed_dim", type=int, default=256,
                        help="Embedding dimension (output of first layer).")
    parser.add_argument("--hidden1_dim", type=int, default=32,
                        help="Output dimension of the first hidden layer.")
    parser.add_argument("--hidden2_dim", type=int, default=16,
                        help="Output dimension of the second hidden layer.")
    parser.add_argument("--opset", type=int, default=11,
                        help="ONNX opset version to use for export (default: 11).")
    parser.add_argument("--no_verify", action="store_true",
                        help="Skip the ONNX model verification step.")

    args = parser.parse_args()

    # Run the conversion function with updated parameters
    convert_pt_to_onnx(
        pt_path=args.pt_input_path,
        onnx_path=args.onnx_output_path,
        input_size=args.input_size,
        embed_dim=args.embed_dim,
        hidden1_dim=args.hidden1_dim, # Pass new hidden dim
        hidden2_dim=args.hidden2_dim, # Pass new hidden dim
        opset_version=args.opset,
        verify=not args.no_verify
    )
