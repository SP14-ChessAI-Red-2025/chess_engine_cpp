# python/chess_dir/__init__.py (Corrected)

# Import directly from the compiled Cython module (ai_chess...so)
# Python finds the .so file automatically when importing 'ai_chess'
from .ai_chess import ChessEngine, Player, PieceType, GameStatus, ChessMove, BoardPosition

# Define what names are exported when someone does 'from chess_dir import *'
# OR defines what names are considered part of the public API
__all__ = [
    'ChessEngine',
    'Player',       # Export the enums if needed by users of the package
    'PieceType',
    'GameStatus',
    'ChessMove',    # Export the Cython wrapper class
    'BoardPosition' # Export the Cython wrapper class
]