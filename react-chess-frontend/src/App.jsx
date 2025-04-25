// src/App.jsx (FIXED State Handling)
import React, { useState, useEffect, useCallback } from 'react';
import Board from './Board';
import GameModeSelector from './GameModeSelector';
import './App.css';

// --- Constants ---
const Player = { WHITE: 0, BLACK: 1 };
const PieceType = { NONE: 0, PAWN: 1, KNIGHT: 2, BISHOP: 3, ROOK: 4, QUEEN: 5, KING: 6 };
const GameStatus = { NORMAL: 0, DRAW: 1, CHECKMATE: 2, RESIGNED: 3, DRAW_BY_REPETITION: 4 };
const GameMode = { SELECT: 0, AI_VS_AI: 1, PLAYER_VS_AI_WHITE: 2, PLAYER_VS_AI_BLACK: 3 };

// Ensure this points to your Render backend URL
const API_URL = 'https://chess-engine-cpp.onrender.com/api';

function App() {
  // --- State Variables ---
  const [boardState, setBoardState] = useState(null);
  const [validMoves, setValidMoves] = useState([]);
  const [selectedSquare, setSelectedSquare] = useState(null);
  const [statusMessage, setStatusMessage] = useState("Select Game Mode");
  const [isLoading, setIsLoading] = useState(false);
  const [gameMode, setGameMode] = useState(GameMode.SELECT);
  const [playerColor, setPlayerColor] = useState(null);

  // --- Helper to Get Status String ---
  const getGameStatusMessage = (state) => {
    // (Keep existing implementation)
    if (!state) return "Loading State...";
    if (isLoading && statusMessage.startsWith("AI is") || statusMessage.startsWith("Applying")) return statusMessage; // Keep specific loading messages

    switch (state.status) {
      case GameStatus.NORMAL:
        const player = state.current_player === Player.WHITE ? "White" : "Black";
        const check = state.in_check && state.in_check[state.current_player] ? " (Check!)" : "";
        return `${player}'s Turn${check}`;
      case GameStatus.CHECKMATE:
        const winner = state.current_player === Player.BLACK ? "White" : "Black";
        return `Checkmate! ${winner} Wins.`;
      case GameStatus.DRAW: return "Draw (Stalemate/50-Move/Material)";
      case GameStatus.DRAW_BY_REPETITION: return "Draw (Repetition)";
      case GameStatus.RESIGNED: return "Resigned";
      default: return `Unknown Status (${state.status})`;
    }
  };

  // --- Fetch ONLY Valid Moves --- (Helper Function)
  const fetchValidMoves = useCallback(async (currentState) => {
      if (!currentState || currentState.status !== GameStatus.NORMAL) {
          console.log("Skipping move fetch: game over or no state.");
          setValidMoves([]);
          return;
      }
      console.log("Fetching valid moves for new state...");
      try {
          const movesResponse = await fetch(`${API_URL}/moves`);
          if (!movesResponse.ok) {
              console.error(`Failed to fetch moves: ${movesResponse.status}`);
              setValidMoves([]);
          } else {
              const movesData = await movesResponse.json();
              setValidMoves(Array.isArray(movesData) ? movesData : []);
              console.log(`Fetched ${movesData?.length ?? 0} moves.`);
          }
      } catch (error) {
           console.error("Error fetching valid moves:", error);
           setValidMoves([]);
      }
  }, []); // No dependencies needed


  // --- Fetch Initial Game State (Only fetches state, moves fetched separately) ---
  const fetchInitialGameState = useCallback(async (caller = "unknown") => {
    if (isLoading) { console.log(`fetchInitialGameState skipped by ${caller}: isLoading=true`); return null; }
    console.log(`Fetching initial game state (called by ${caller})...`);
    setIsLoading(true);
    setStatusMessage("Fetching state...");

    let newState = null;
    try {
      const stateResponse = await fetch(`${API_URL}/state`);
      if (!stateResponse.ok) throw new Error(`State fetch failed: ${stateResponse.status}`);
      newState = await stateResponse.json();
      if (!newState || typeof newState !== 'object') throw new Error("Received invalid state from server");
      console.log("Fetched Initial State:", newState);

      setBoardState(newState); // Set initial state
      setStatusMessage(getGameStatusMessage(newState)); // Update status message

      // Fetch moves separately after setting initial state
      await fetchValidMoves(newState);

    } catch (error) {
      console.error("Error fetching initial game state:", error);
      setStatusMessage(`Error fetching state: ${error.message}`);
      setBoardState(null);
      setValidMoves([]);
    } finally {
      setIsLoading(false);
    }
    return newState;
  }, [isLoading, fetchValidMoves]); // Add fetchValidMoves dependency


  // --- Reset Game ---
  const returnToModeSelect = useCallback(async () => {
       // ... (Keep existing implementation, maybe call fetchInitialGameState at end if needed, or just reset) ...
       setIsLoading(true);
       setStatusMessage("Resetting game...");
       try {
           console.log("Calling /api/reset...");
           const response = await fetch(`${API_URL}/reset`, { method: 'POST' });
           if (!response.ok) { /* ... error handling ... */ }
           console.log("Reset successful on backend.");
           // Reset frontend state fully
           setGameMode(GameMode.SELECT);
           setBoardState(null);
           setValidMoves([]);
           setSelectedSquare(null);
           setPlayerColor(null);
           setStatusMessage("Select Game Mode");
       } catch (error) { /* ... error handling ... */ }
       finally { setIsLoading(false); }
  }, []);


  // --- Start Game ---
  const handleGameModeSelect = useCallback((mode) => {
    setGameMode(mode);
    setStatusMessage("Loading Game...");
    setBoardState(null);
    setValidMoves([]);
    setSelectedSquare(null);

    if (mode === GameMode.PLAYER_VS_AI_WHITE) setPlayerColor(Player.WHITE);
    else if (mode === GameMode.PLAYER_VS_AI_BLACK) setPlayerColor(Player.BLACK);
    else setPlayerColor(null);

    // Fetch initial state (which now also fetches initial moves)
    fetchInitialGameState("handleGameModeSelect");
  }, [fetchInitialGameState]);


  // --- Trigger AI Move ---
  const triggerAiMove = useCallback(async () => {
    // --- Condition Check (Good) ---
    if (isLoading || !boardState || boardState.status !== GameStatus.NORMAL || gameMode === GameMode.SELECT) {
        console.log("triggerAiMove skipped: conditions not met."); return; }
    const isAIsTurn =
      (gameMode === GameMode.AI_VS_AI) ||
      (gameMode === GameMode.PLAYER_VS_AI_WHITE && boardState.current_player === Player.BLACK) ||
      (gameMode === GameMode.PLAYER_VS_AI_BLACK && boardState.current_player === Player.WHITE);
    if (!isAIsTurn) { console.log("triggerAiMove skipped: not AI's turn."); return; }
    // --- End Condition Check ---

    console.log("Triggering AI move...");
    setIsLoading(true);
    setStatusMessage("AI is thinking...");
    setValidMoves([]); // Clear old moves
    setSelectedSquare(null); // Clear selection

    try {
      const response = await fetch(`${API_URL}/ai_move`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) }); // Pass difficulty if needed
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `AI move failed: ${response.status}`);
      }

      // --- FIX: Use the response directly ---
      const newState = await response.json();
      console.log("Received new state directly from POST /api/ai_move:", newState);
      setBoardState(newState); // Update state with the direct response
      setStatusMessage(getGameStatusMessage(newState)); // Update status message

      // Fetch valid moves for the *next* player based on the newState
      await fetchValidMoves(newState);
      // --- END FIX ---

    } catch (error) {
      console.error("Error during AI move:", error);
      setStatusMessage(`Error during AI turn: ${error.message}`);
      // Optionally try fetching state again on error to recover?
      // fetchInitialGameState("triggerAiMove-error");
    } finally {
      setIsLoading(false); // Ensure loading is set to false
    }
  }, [isLoading, boardState, gameMode, fetchValidMoves]); // Added fetchValidMoves


  // --- Effect to Automatically Trigger AI Move (Keep as is) ---
  useEffect(() => {
    // Initial guard conditions
    if (!boardState || gameMode === GameMode.SELECT || boardState.status !== GameStatus.NORMAL || isLoading) {
      // console.log("useEffect skipped: Initial guards"); // Optional debug log
      return;
    }

    // Determine if it *should* be AI's turn based on the current boardState
    const isAIsTurn =
      (gameMode === GameMode.AI_VS_AI) ||
      (gameMode === GameMode.PLAYER_VS_AI_WHITE && boardState.current_player === Player.BLACK) ||
      (gameMode === GameMode.PLAYER_VS_AI_BLACK && boardState.current_player === Player.WHITE);

    // console.log(`useEffect Check: gameMode=${gameMode}, currentPlayer=${boardState.current_player}, isAIsTurn=${isAIsTurn}, isLoading=${isLoading}`); // Optional debug log

    if (isAIsTurn) {
      // Use a timeout (keeps the delay, also helps separate trigger logic)
      const timeoutId = setTimeout(() => {
        // --- Double Check Inside Timeout ---
        // Check isLoading AGAIN, as it might have changed between the effect running and the timeout firing.
        // This ensures we don't trigger if another action started loading in the meantime.
        if (!isLoading) {
             console.log(`setTimeout: Triggering AI move. Current player in state should be AI's turn (${boardState.current_player}).`);
             triggerAiMove();
        } else {
            console.log("setTimeout: Skipped AI trigger because isLoading became true.");
        }
        // --- End Double Check ---
      }, 500); // Keep or adjust delay as needed

      // Cleanup function for the timeout
      return () => {
          // console.log("useEffect cleanup: Clearing timeout"); // Optional debug log
          clearTimeout(timeoutId);
      }
    }
  // Dependencies: boardState, gameMode, isLoading determine WHEN the effect runs.
  // triggerAiMove is stable due to useCallback, so it can usually be omitted.
  // If you still face issues, you might need useRef to access the absolute latest state inside setTimeout.
  }, [boardState, gameMode, isLoading, triggerAiMove]); // Keep triggerAiMove if useCallback isn't perfect or ESLint insists

  // --- Calculate Highlight Squares ---
  const getHighlightSquares = useCallback(() => {
     if (!selectedSquare || !Array.isArray(validMoves) || isLoading) return [];
     return validMoves
       .filter(m => m.start.rank === selectedSquare.rank && m.start.file === selectedSquare.file)
       .map(m => ({ rank: m.target.rank, file: m.target.file }));
  }, [selectedSquare, validMoves, isLoading]);


  // --- Render Logic ---
  return (
    // This div is now always present and applies the centering styles
    <div className="App">
      {gameMode === GameMode.SELECT ? (
        // Render the Selector *inside* the centered App container
        <GameModeSelector onSelectMode={handleGameModeSelect} />
      ) : (
        // Render the Game View *inside* the centered App container
        // Use a React Fragment <>...</> to group multiple elements
        <>
          <h1>React Chess</h1>
          <div className={`game-info ${isLoading ? 'loading-active' : ''}`}>{statusMessage}</div>
          {boardState && Array.isArray(boardState.pieces) ? (
            <Board
              boardPieces={boardState.pieces}
              onSquareClick={handleSquareClick}
              selectedSquare={selectedSquare}
              highlightSquares={getHighlightSquares()} // Make sure this function is available
            />
          ) : (
            <div className="loading">{isLoading ? 'Loading...' : 'Waiting for Server...'}</div>
          )}
          <button onClick={returnToModeSelect} disabled={isLoading} style={{ marginTop: '15px' }}>
            Change Mode / Reset
          </button>
        </>
      )}
    </div>
  );
}

export default App;
