How to use the Python API:

See python_example.py for an example of using the API

Please don't use any methods, variables, or classes not described in this document, as I may end up changing them

## ChessEngine class

Its constructor takes the path to the shared library as its only argument.

This class wraps the C++ code.  It allocates memory, so you must either call the free_memory method when you are done with it, or use the context manager protocol.

Example:

    with ChessEngine("/libraray_path/chess_cpp.dll") as chess_engine:
        # your code here
    
    # free_memory is automatically called at the end of the with block

Alternate example:

    chess_engine = ChessEngine("/path/to/lib/chess_cpp.dll")
    # your code here
    chess_engine.free_memory()

ChessEngine has 4 methods in its API:

- def free_memory(self) -> None:
  - as described above
- def get_valid_moves(self) -> list[ChessMove]:
  - Returns a list of all possible ChessMove instances
- def apply_move(self, move: ChessMove) -> None:
  - Applies the move to the current board state
- def ai_move(self, difficulty: int) -> None:
  - Have the chess AI make a move

ChessEngine has 1 instance variable:
- board_state: BoardState
  - This contains the current state of the board.  See below for more details


## Other classes and enums

### Piece
Contains 2 instance variables:
- piece_type: An integer determining the type of piece
  - 0: None.  USed to represent an empty square
  - 1: Pawn
  - 2: Knight
  - 3: Bishop
  - 4: Rook
  - 5: Queen
  - 6: King
- piece_player: 0 for white, 1 for black


### BoardPosition
Contains 2 instance variables:
- rank: an integer from 0-7 representing the rank.  0 for rank 1, 7 for rank 8
- file: an integer from 0-7 representing the file.  0 for file A, 7 for file H

### ChessMove
Contains 4 instance variables:
- start_position:
  - A BoardPosition representing the location of the piece to be moved
  - If the move is a castle, then this refers to the location of the rook, not the king
- target_position
  - A BoardPosition representing the location the piece is moving to
  - If the move is a castle, then this value is meaningless
  - If type == en_passant, this refers to the position where the capturing pawn will move to, not the location of the captured pawn
- promotion_target
  - An integer representing the piece that a pawn promotes to.  Meaningless if type != promotion
  - The integer is interpreted as described in Piece.piece_type
- type
  - The type of move. Possible values are:
    - 0: A normal move.  The piece at start_position moves to target_position
    - 1: A capture.  The piece at target_position is removed, and the piece at start_position moves to target_position
    - 2: En passant
    - 3: Castling
    - 4: Pawn promotion
    - 5: Claim a draw
    - 6: Resignation


### BoardState
Contains 3 properties:
- status
  - An integer representing the current game status
    - 0: The game is currently being played 
    - 1: The game is a draw
    - 2: The game is over due to checkmate.  The current_player is the loser
    - 3: The game is over due to resignation.  The current_player is the loser
- current_player
  - An integer representing the current player
    - 0 for white, 1 for black
- pieces
  - An 8x8 2D array containing the pieces on the board
    - The first index is the rank, the second index is the file
    - For example, board_state.pieces[0][0] is A1, pieces[2][3] is D3