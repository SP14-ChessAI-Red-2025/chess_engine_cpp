// src/App.jsx (Synchronous AI Mode - No Timer)
import React, { useState, useEffect, useCallback } from 'react';
import Board from './Board';
import GameModeSelector from './GameModeSelector';
import './App.css';

// --- Constants ---
const Player = { WHITE: 0, BLACK: 1 };
const PieceType = { NONE: 0, PAWN: 1, KNIGHT: 2, BISHOP: 3, ROOK: 4, QUEEN: 5, KING: 6 };
const GameStatus = { NORMAL: 0, DRAW: 1, CHECKMATE: 2, RESIGNED: 3, DRAW_BY_REPETITION: 4 };
const GameMode = { SELECT: 0, AI_VS_AI: 1, PLAYER_VS_AI_WHITE: 2, PLAYER_VS_AI_BLACK: 3 };

const API_URL = 'https://chess-engine-cpp.onrender.com';

function App() {
  // --- State Variables ---
  const [boardState, setBoardState] = useState(null);
  const [validMoves, setValidMoves] = useState([]);
  const [selectedSquare, setSelectedSquare] = useState(null);
  const [statusMessage, setStatusMessage] = useState("Select Game Mode");
  const [isLoading, setIsLoading] = useState(false);
  const [gameMode, setGameMode] = useState(GameMode.SELECT);
  const [playerColor, setPlayerColor] = useState(null);
  // Timer state removed

  // --- Helper to Get Status String ---
  const getGameStatusMessage = (state) => {
    if (!state) return "Loading State...";
    if (isLoading) return statusMessage; // Keep loading messages

    // Timeout checks removed

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

  // --- Fetch Initial/Current Game State and Moves ---
  const fetchGameState = useCallback(async (caller = "unknown") => {
    if (isLoading) { console.log(`fetchGameState skipped by ${caller}: isLoading=true`); return null; }
    console.log(`Fetching game state (called by ${caller})...`);
    setIsLoading(true);
    setStatusMessage("Fetching state...");

    let newState = null;
    try {
      const stateResponse = await fetch(`${API_URL}/state`);
      if (!stateResponse.ok) throw new Error(`State fetch failed: ${stateResponse.status}`);
      newState = await stateResponse.json();
      if (!newState || typeof newState !== 'object') throw new Error("Received invalid state from server");
      console.log("Fetched State:", newState);

      setBoardState(newState);
      setStatusMessage(getGameStatusMessage(newState)); // Update status based on new state

      if (newState.status === GameStatus.NORMAL) {
        console.log("Fetching valid moves...");
        const movesResponse = await fetch(`${API_URL}/moves`);
        if (!movesResponse.ok) {
           console.error(`Failed to fetch moves: ${movesResponse.status}`);
           setValidMoves([]);
        } else {
           const movesData = await movesResponse.json();
           setValidMoves(Array.isArray(movesData) ? movesData : []);
           console.log(`Fetched ${movesData?.length ?? 0} moves.`);
        }
      } else {
        setValidMoves([]); // Game is over
      }

    } catch (error) {
      console.error("Error fetching game state:", error);
      setStatusMessage(`Error fetching state: ${error.message}`);
      setBoardState(null);
      setValidMoves([]);
    } finally {
      setIsLoading(false);
    }
    return newState;
  }, [isLoading]);


  // --- Reset Game and Return to Mode Select ---
  const returnToModeSelect = useCallback(async () => {
      setIsLoading(true);
      setStatusMessage("Resetting game...");
      // Timer clearing removed

      try {
          console.log("Calling /api/reset...");
          const response = await fetch(`${API_URL}/reset`, { method: 'POST' });
          if (!response.ok) {
              const errorData = await response.json().catch(() => ({}));
              throw new Error(errorData.error || `Reset failed: ${response.status}`);
          }
          console.log("Reset successful on backend.");

          // Reset relevant frontend state (no timers)
          setGameMode(GameMode.SELECT);
          setBoardState(null);
          setValidMoves([]);
          setSelectedSquare(null);
          setPlayerColor(null);
          setStatusMessage("Select Game Mode");

      } catch (error) {
          console.error("Error resetting game:", error);
          setStatusMessage(`Error resetting: ${error.message}. Please select mode again.`);
          // Force back to select mode even if reset failed
          setGameMode(GameMode.SELECT);
          setBoardState(null);
          setValidMoves([]);
          setSelectedSquare(null);
          setPlayerColor(null);
      } finally {
          setIsLoading(false);
      }
  }, []); // Removed timerIntervalId dependency


  // --- Start Game (Called from GameModeSelector) ---
  const handleGameModeSelect = useCallback((mode) => {
    setGameMode(mode);
    setStatusMessage("Loading Game...");
    // Reset state immediately
    setBoardState(null);
    setValidMoves([]);
    setSelectedSquare(null);
    // Timer reset removed

    if (mode === GameMode.PLAYER_VS_AI_WHITE) setPlayerColor(Player.WHITE);
    else if (mode === GameMode.PLAYER_VS_AI_BLACK) setPlayerColor(Player.BLACK);
    else setPlayerColor(null);

    fetchGameState("handleGameModeSelect");
  }, [fetchGameState]); // Removed timerIntervalId dependency


  // --- Trigger AI Move ---
  const triggerAiMove = useCallback(async () => {
    if (isLoading || !boardState || boardState.status !== GameStatus.NORMAL || gameMode === GameMode.SELECT) {
        console.log("triggerAiMove skipped: conditions not met."); return; }
    const isAIsTurn =
      (gameMode === GameMode.AI_VS_AI) ||
      (gameMode === GameMode.PLAYER_VS_AI_WHITE && boardState.current_player === Player.BLACK) ||
      (gameMode === GameMode.PLAYER_VS_AI_BLACK && boardState.current_player === Player.WHITE);
    if (!isAIsTurn) { console.log("triggerAiMove skipped: not AI's turn."); return; }

    console.log("Triggering AI move...");
    setIsLoading(true);
    setStatusMessage("AI is thinking...");
    setValidMoves([]);
    setSelectedSquare(null);

    try {
      const response = await fetch(`${API_URL}/ai_move`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `AI move failed: ${response.status}`);
      }
      const newState = await response.json();
      console.log("Received new state after AI move:", newState);
      setBoardState(newState);
      if (newState.status === GameStatus.NORMAL) {
          await fetchGameState("triggerAiMove-after"); // Fetch state AND moves for next player
      } else {
           setValidMoves([]); // Game ended
           setStatusMessage(getGameStatusMessage(newState)); // Update final status
      }
    } catch (error) {
      console.error("Error during AI move:", error);
      setStatusMessage(`Error during AI turn: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, boardState, gameMode, fetchGameState]);


  // --- Effect to Automatically Trigger AI Move ---
  useEffect(() => {
    if (!boardState || gameMode === GameMode.SELECT || boardState.status !== GameStatus.NORMAL || isLoading) return;
    const isAIsTurn =
      (gameMode === GameMode.AI_VS_AI) ||
      (gameMode === GameMode.PLAYER_VS_AI_WHITE && boardState.current_player === Player.BLACK) ||
      (gameMode === GameMode.PLAYER_VS_AI_BLACK && boardState.current_player === Player.WHITE);
    if (isAIsTurn) {
      const timeoutId = setTimeout(triggerAiMove, 100);
      return () => clearTimeout(timeoutId);
    }
  }, [boardState, gameMode, isLoading, triggerAiMove]);

  // --- Timer useEffect REMOVED ---


  // --- Handle Square Click Logic ---
  const handleSquareClick = useCallback(async (rank, file) => {
    if (isLoading || !boardState || boardState.status !== GameStatus.NORMAL || gameMode === GameMode.AI_VS_AI) return;
    const isPlayerTurn = playerColor !== null && boardState.current_player === playerColor;
    if (!isPlayerTurn) return;

    const clickedPiece = boardState.pieces[rank][file];

    if (selectedSquare) {
      const sourceSq = selectedSquare;
      const targetSq = { rank, file };
      const move = validMoves.find(m =>
        m.start.rank === sourceSq.rank && m.start.file === sourceSq.file &&
        m.target.rank === targetSq.rank && m.target.file === targetSq.file
      );

      if (move) {
        setIsLoading(true);
        setStatusMessage("Applying move...");
        setSelectedSquare(null);
        setValidMoves([]);

        try {
          const movePayload = { start: move.start, target: move.target };
          if (move.type === 4) { // Use direct integer value if MoveType enum isn't imported
            movePayload.promotion = PieceType.QUEEN; // Default to Queen
            console.warn("Applying default Queen promotion!");
          }
          const response = await fetch(`${API_URL}/apply_move`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(movePayload) });
          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `Move failed: ${response.status}`);
          }
          const newState = await response.json();
          setBoardState(newState);
          await fetchGameState("handleSquareClick-after");
        } catch (error) {
          console.error("Error applying move:", error);
          setStatusMessage(`Error applying move: ${error.message}`);
          fetchGameState("handleSquareClick-error");
        } finally {
          setIsLoading(false);
        }
      } else {
        if (clickedPiece.type !== PieceType.NONE && clickedPiece.player === playerColor) {
          setSelectedSquare({ rank, file });
        } else {
          setSelectedSquare(null);
        }
      }
    } else {
      if (clickedPiece.type !== PieceType.NONE && clickedPiece.player === playerColor) {
        setSelectedSquare({ rank, file });
      }
    }
  }, [isLoading, boardState, gameMode, playerColor, selectedSquare, validMoves, fetchGameState]);


  // --- Calculate Highlight Squares ---
  const getHighlightSquares = useCallback(() => {
    if (!selectedSquare || !Array.isArray(validMoves) || isLoading) return [];
    return validMoves
      .filter(m => m.start.rank === selectedSquare.rank && m.start.file === selectedSquare.file)
      .map(m => ({ rank: m.target.rank, file: m.target.file }));
  }, [selectedSquare, validMoves, isLoading]);


  // --- Render Logic ---
  if (gameMode === GameMode.SELECT) {
    return <GameModeSelector onSelectMode={handleGameModeSelect} />;
  }

  const highlightSquares = getHighlightSquares();

  return (
    <div className="App">
      <h1>React Chess</h1>

      {/* Timer Display REMOVED */}

      {/* Game Status */}
      <div className={`game-info ${isLoading ? 'loading-active' : ''}`}>{statusMessage}</div>

      {/* Board or Loading Message */}
      {boardState && Array.isArray(boardState.pieces) ? (
        <Board
          boardPieces={boardState.pieces}
          onSquareClick={handleSquareClick}
          selectedSquare={selectedSquare}
          highlightSquares={highlightSquares}
        />
      ) : (
        <div className="loading">{isLoading ? 'Loading...' : 'Waiting for Server...'}</div>
      )}

      {/* Change Mode Button */}
      <button onClick={returnToModeSelect} disabled={isLoading} style={{ marginTop: '15px' }}>
        Change Mode / Reset
      </button>
    </div>
  );
}

export default App;