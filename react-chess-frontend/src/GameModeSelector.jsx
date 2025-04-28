// src/GameModeSelector.jsx
import React from 'react';

// Constants must be accessible here too
const GameMode = { SELECT: 0, AI_VS_AI: 1, PLAYER_VS_AI_WHITE: 2, PLAYER_VS_AI_BLACK: 3 };


function GameModeSelector({ onSelectMode, onReturnToCoverPage }) {
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
            <button
                onClick={onReturnToCoverPage}
                style={{
                    marginTop: '20px',
                    padding: '10px 20px',
                    backgroundColor: '#007bff',
                    color: 'white',
                    border: 'none',
                    borderRadius: '5px',
                    cursor: 'pointer',
                }}
            >
                Return to Cover Page
            </button>
        </div>
    );
}

export default GameModeSelector;