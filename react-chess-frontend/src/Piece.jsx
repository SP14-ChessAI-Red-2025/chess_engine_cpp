// src/Piece.jsx (Updated to use images)
import React from 'react';

// Base path for images (assuming they are in public/images/)
const imageBase = '/website/images/';

// Map piece type and player to image filenames
const pieceImageMap = {
  // PieceType: 1=Pawn, 2=Knight, 3=Bishop, 4=Rook, 5=Queen, 6=King
  // Player: 0=White, 1=Black
  '1-0': `${imageBase}wht_pawn.png`,
  '2-0': `${imageBase}wht_knight.png`,
  '3-0': `${imageBase}wht_bishop.png`,
  '4-0': `${imageBase}wht_rook.png`,
  '5-0': `${imageBase}wht_queen.png`,
  '6-0': `${imageBase}wht_king.png`,
  '1-1': `${imageBase}blk_pawn.png`,
  '2-1': `${imageBase}blk_knight.png`,
  '3-1': `${imageBase}blk_bishop.png`,
  '4-1': `${imageBase}blk_rook.png`,
  '5-1': `${imageBase}blk_queen.png`,
  '6-1': `${imageBase}blk_king.png`,
};

function Piece({ type, player }) {
  const pieceKey = `${type}-${player}`;
  const imageUrl = pieceImageMap[pieceKey];

  // Render an img tag if a mapping exists, otherwise render nothing
  return imageUrl ? (
    <img
      src={imageUrl}
      alt={`Piece type ${type} player ${player}`}
      className="piece-image" // Use a specific class for styling images
      draggable="false" // Prevent native image dragging
    />
  ) : null; // Don't render anything if type/player is invalid (e.g., PieceType.NONE)
}

export default Piece;