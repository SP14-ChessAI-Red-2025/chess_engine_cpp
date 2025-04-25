// src/Board.jsx
import React from 'react';
import Piece from './Piece'; // Assuming Piece component exists

function Board({ boardPieces, onSquareClick, selectedSquare, highlightSquares }) {
  const renderSquares = () => {
    const squares = [];
    for (let r = 7; r >= 0; r--) { // Internal rank (8 to 1)
      for (let f = 0; f < 8; f++) { // Internal file (a to h)
        const isDark = (r + f) % 2 !== 0;
        const displayRow = 7 - r; // Convert internal rank to display row (0-7)
        const displayCol = f;     // Internal file = display col (0-7)

        const piece = boardPieces[r][f];
        const isSelected = selectedSquare && selectedSquare.rank === r && selectedSquare.file === f;
        const isHighlight = highlightSquares.some(sq => sq.rank === r && sq.file === f);

        squares.push(
          <div
            key={`${r}-${f}`}
            className={`square ${isDark ? 'dark' : 'light'} ${isSelected ? 'selected' : ''} ${isHighlight ? 'highlight' : ''}`}
            onClick={() => onSquareClick(r, f)} // Pass internal rank/file
            style={{ gridRow: displayRow + 1, gridColumn: displayCol + 1 }} // CSS Grid position
          >
            {piece.type !== 0 && <Piece type={piece.type} player={piece.player} />}
          </div>
        );
      }
    }
    return squares;
  };

  return <div className="board">{renderSquares()}</div>;
}

export default Board;