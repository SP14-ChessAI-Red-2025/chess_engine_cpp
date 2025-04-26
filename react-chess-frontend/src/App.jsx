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
    // --- Condition Check ---
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


  // --- Effect to Automatically Trigger AI Move ---
  useEffect(() => {
    // Guard conditions - should prevent running if loading or game over, or if no player color assigned
    if (!boardState || gameMode === GameMode.SELECT || boardState.status !== GameStatus.NORMAL || isLoading) {
      return;
    }

    // Determine if it should be the AI's turn to move
    let shouldAiMove = false;
    if (gameMode === GameMode.AI_VS_AI) {
        // In AI vs AI mode, AI always moves if the game is ongoing
        shouldAiMove = true;
    } else if (playerColor !== null) {
        // In Player vs AI modes, AI moves if the current player in the state
        // is NOT the human player's chosen color.
        shouldAiMove = (boardState.current_player !== playerColor);
    } else {
        // Should not happen if gameMode is not SELECT, but good to handle
        console.error("AI Effect: Game mode requires AI but playerColor is null!");
    }


    // If it's determined to be the AI's turn...
    if (shouldAiMove) {
      // console.log(`AI Effect: Triggering AI move for player ${boardState.current_player}`);
      const timeoutId = setTimeout(triggerAiMove, 500); // triggerAiMove sets isLoading=true
      return () => clearTimeout(timeoutId); // Cleanup timeout on unmount/re-run
    }
  // Dependencies: Check if isLoading still needed if handled solely within triggerAiMove/handleSquareClick
  }, [boardState, gameMode, playerColor, isLoading, triggerAiMove]); // Added playerColor dependency


  // --- Handle Square Click Logic ---
  const handleSquareClick = useCallback(async (rank, file) => {
    // --- Condition Checks ---
    if (isLoading || !boardState || boardState.status !== GameStatus.NORMAL || gameMode === GameMode.AI_VS_AI) return;
    const isPlayerTurn = playerColor !== null && boardState.current_player === playerColor;
    if (!isPlayerTurn) return;
    // --- End Condition Checks ---

    const clickedPiece = boardState.pieces[rank][file];

    if (selectedSquare) { // A piece is already selected
      const sourceSq = selectedSquare;
      const targetSq = { rank, file };

      // Find the move object based on current valid moves
      const move = validMoves.find(m =>
        m.start.rank === sourceSq.rank && m.start.file === sourceSq.file &&
        m.target.rank === targetSq.rank && m.target.file === targetSq.file
      );

      if (move) { // Clicked on a valid target square for the selected piece
        setIsLoading(true);
        setStatusMessage("Applying move...");
        setSelectedSquare(null); // Deselect piece
        setValidMoves([]); // Clear old moves

        try {
          // Prepare payload, include current_player for conditional check on backend
          const movePayload = {
              start: move.start,
              target: move.target,
              promotion: PieceType.NONE, // Default, override if needed
              current_player: boardState.current_player // Send player making the move
          };
          // Handle promotion (assuming default to Queen for now)
          // TODO: Implement promotion piece selection UI if needed
          if (move.type === 4 /* Promotion */ && boardState.pieces[sourceSq.rank][sourceSq.file]?.type === PieceType.PAWN) {
              if (targetSq.rank === 7 || targetSq.rank === 0) {
                  movePayload.promotion = PieceType.QUEEN; // Default to Queen
                  console.log("Applying default Queen promotion for move:", move);
              }
          }

          // Call the backend API
          const response = await fetch(`${API_URL}/apply_move`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(movePayload) });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `Move application failed: ${response.status}`);
          }

          // --- FIX: Use the response directly ---
          const newState = await response.json();
          console.log("Received new state directly from POST /api/apply_move:", newState);
          setBoardState(newState); // Update state with the direct response
          setStatusMessage(getGameStatusMessage(newState)); // Update status message

          // Fetch valid moves for the *next* player based on the newState
          await fetchValidMoves(newState);
          // --- END FIX ---

        } catch (error) {
          console.error("Error applying move:", error);
          setStatusMessage(`Error applying move: ${error.message}`);
          // Attempt to recover by fetching current state/moves
          fetchInitialGameState("handleSquareClick-error"); // Fetch both state and moves on error
        } finally {
          setIsLoading(false); // Ensure loading is set to false
        }
      } else { // Clicked on a square that is NOT a valid target
        if (clickedPiece.type !== PieceType.NONE && clickedPiece.player === playerColor) {
          // Clicked on another of the player's pieces, select it instead
          setSelectedSquare({ rank, file });
        } else {
          // Clicked on empty square or opponent piece, deselect
          setSelectedSquare(null);
        }
      }
    } else { // No piece was selected previously
      if (clickedPiece.type !== PieceType.NONE && clickedPiece.player === playerColor) {
        // Clicked on one of the player's pieces, select it
        setSelectedSquare({ rank, file });
      }
      // Do nothing if clicked empty square or opponent piece when nothing selected
    }
  }, [isLoading, boardState, gameMode, playerColor, selectedSquare, validMoves, fetchValidMoves, fetchInitialGameState]); // Added dependencies


  // --- Calculate Highlight Squares ---
  const getHighlightSquares = useCallback(() => {
     if (!selectedSquare || !Array.isArray(validMoves) || isLoading) return [];
     return validMoves
       .filter(m => m.start.rank === selectedSquare.rank && m.start.file === selectedSquare.file)
       .map(m => ({ rank: m.target.rank, file: m.target.file }));
  }, [selectedSquare, validMoves, isLoading]);


  // --- Render Logic ---
  const highlightSquares = getHighlightSquares(); // Assumes getHighlightSquares is defined above

  return (
    // This div is now the single top-level element, always rendered
    <div className="App">
      {gameMode === GameMode.SELECT ? (
        // Render Selector when gameMode is SELECT
        <GameModeSelector onSelectMode={handleGameModeSelect} />
      ) : (
        // Otherwise, render the Game View elements
        // Use a React Fragment <>...</> to group multiple elements
        <>
          <h1>React Chess</h1>

          {/* Timer Display REMOVED - Assuming this is intentional */}

          {/* Game Status */}
          <div className={`game-info ${isLoading ? 'loading-active' : ''}`}>{statusMessage}</div>

          {/* Board or Loading Message */}
          {boardState && Array.isArray(boardState.pieces) ? (
            <Board
              boardPieces={boardState.pieces}
              onSquareClick={handleSquareClick}
              selectedSquare={selectedSquare}
              // Pass highlightSquares calculated above
              highlightSquares={highlightSquares}
            />
          ) : (
            <div className="loading">{isLoading ? 'Loading...' : 'Waiting for Server...'}</div>
          )}

          {/* Change Mode Button */}
          <button onClick={returnToModeSelect} disabled={isLoading} style={{ marginTop: '15px' }}>
            Change Mode / Reset
          </button>
        </>
      )}
    </div>
  );
}

export default App;