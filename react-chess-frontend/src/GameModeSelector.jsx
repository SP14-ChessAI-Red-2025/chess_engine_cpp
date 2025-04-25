// src/GameModeSelector.jsx
import React from 'react';

// Constants must be accessible here too
const GameMode = { SELECT: 0, AI_VS_AI: 1, PLAYER_VS_AI_WHITE: 2, PLAYER_VS_AI_BLACK: 3 };


function GameModeSelector({ onSelectMode }) {
    return (
        <div className="game-mode-selector">
            <h2>Select Game Mode</h2>
            <button onClick={() => onSelectMode(GameMode.PLAYER_VS_AI_WHITE)}>
                Play as White (vs AI)
            </button>
            <button onClick={() => onSelectMode(GameMode.PLAYER_VS_AI_BLACK)}>
                Play as Black (vs AI)
            </button>
            <button onClick={() => onSelectMode(GameMode.AI_VS_AI)}>
                AI vs AI
            </button>
        </div>
    );
}

export default GameModeSelector;