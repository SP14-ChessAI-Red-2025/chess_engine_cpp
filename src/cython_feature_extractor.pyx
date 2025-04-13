# cython: language_level=3

# cython: boundscheck=False
# Faster array access (unsafe if indices are wrong)

# cython: wraparound=False
# Faster array access (unsafe if indices are wrong)

# cython: cdivision=True
# Use C-style division (faster for integers)

# cython: nonecheck=False
# Don't check for None assignments (faster, use carefully)

import numpy as np
# cimport numpy tells Cython to use the efficient C API for NumPy
cimport numpy as np
import chess

# Define C types for potentially faster variables
ctypedef np.float32_t FLOAT32_t # C type for float32
ctypedef int INT_t             # C type for standard integers

# Define constants efficiently (or import them if defined elsewhere)
DEF INPUT_FEATURES = 768

# Pre-define the order list (avoids creating it repeatedly)
cdef list PIECE_TYPE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

# Mark function for C-level calls (cpdef makes it callable from Python and C)
cpdef np.ndarray[FLOAT32_t, ndim=1] board_to_features_cython(object board):
    """
    Cython implementation of board_to_features.
    Accepts a python-chess_dir Board object.
    """
    # Declare C variables for loop indices etc.
    cdef INT_t sq, piece_idx, color_idx, base_idx, feature_idx

    # Declare 'piece' as a Python object, as we interact with the Board object
    cdef object piece

    # Initialize the NumPy array. Type is known.
    cdef np.ndarray[FLOAT32_t, ndim=1] features = np.zeros(INPUT_FEATURES, dtype=np.float32)

    # Iterate using range for potentially faster C loop
    for sq in range(64):
        # --- This section still calls Python methods ---
        # board.piece_at() is a Python call, limiting pure C speed.
        piece = board.piece_at(sq)
        # Check if piece is not None
        if piece is not None:
            try:
                # piece.piece_type and piece.color are also Python property accesses
                piece_idx = PIECE_TYPE_ORDER.index(piece.piece_type)
                color_idx = 0 if piece.color == chess.WHITE else 1
                # Calculations are fast C operations
                base_idx = (piece_idx * 2 + color_idx) * 64
                feature_idx = base_idx + sq
                # Direct C-level access to NumPy array data buffer
                features[feature_idx] = 1.0
            except ValueError:
                # Handle cases where piece_type might not be in the list
                # In production, might log or raise, but pass is fastest if tolerable
                pass

    return features # Return the NumPy array