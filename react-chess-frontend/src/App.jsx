// src/App.jsx
import React, { useState, useEffect, useCallback, useRef } from 'react';
import Board from './Board';
import CoverPage from './CoverPage';
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
  const [boardStateHistory, setBoardStateHistory] = useState(new Map());
  const [fiftyMoveCounter, setFiftyMoveCounter] = useState(0);
  const [moveHistory, setMoveHistory] = useState([]); // Tracks moves in chess notation
  const [boardEvaluation, setBoardEvaluation] = useState(null); // Stores the evaluation score
  const [showCoverPage, setShowCoverPage] = useState(true);

  // --- Helper to Get Status String ---
  const getGameStatusMessage = (state) => {
    if (!state) return "Loading State...";
    if (isLoading && (statusMessage.startsWith("AI is") || statusMessage.startsWith("Applying"))) return statusMessage;

    switch (state.status) {
      case GameStatus.NORMAL:
        const player = state.current_player === Player.WHITE ? "White" : "Black";
        const check = state.in_check && state.in_check[state.current_player] ? " (Check!)" : "";
        return `${player}'s Turn${check}`;
      case GameStatus.CHECKMATE:
        const winner = state.current_player === Player.BLACK ? "White" : "Black";
        return `Checkmate! ${winner} Wins.`;
      case GameStatus.DRAW:
        return "Draw (Stalemate/50-Move/Material)";
      case GameStatus.DRAW_BY_REPETITION:
        return "Draw by Threefold Repetition";
      case GameStatus.RESIGNED:
        return "Game Over: Player Resigned";
      default:
        return `Unknown Status (${state.status})`;
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
        if (!Array.isArray(movesData)) throw new Error("Invalid moves data from server.");

        // Filter out special-case moves like claim_draw or resign
        const filteredMoves = movesData.filter(
          (move) => move.type !== 5 && move.type !== 6 // Assuming 5 = resign, 6 = claim_draw
        );

        setValidMoves(filteredMoves);
        console.log(`Fetched ${filteredMoves.length} valid moves.`);
      }
    } catch (error) {
      console.error("Error fetching valid moves:", error);
      setValidMoves([]);
    }
  }, []); // No dependencies needed


  // --- Fetch Initial Game State ---
  const initialStateRef = useRef(null);

  const fetchInitialGameState = useCallback(async (caller = "unknown") => {
    if (isLoading) {
      console.log(`fetchInitialGameState skipped by ${caller}: isLoading=true`);
      return null;
    }

    console.log(`Fetching initial game state (called by ${caller})...`);
    setIsLoading(true);
    setStatusMessage("Fetching state...");

    let newState = null;
    try {
      const stateResponse = await fetch(`${API_URL}/state`);
      if (!stateResponse.ok) throw new Error(`State fetch failed: ${stateResponse.status}`);
      newState = await stateResponse.json();
      if (!newState || typeof newState !== "object") throw new Error("Received invalid state from server");
      console.log("Fetched Initial State:", newState);

      updateBoardStateWithHistory(newState); // Set initial state
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
  }, [isLoading, fetchValidMoves]);

  const fetchBoardEvaluation = useCallback(async () => {
    if (!boardState) {
      console.log("fetchBoardEvaluation skipped: boardState is null.");
      return;
    }
    try {
      console.log("Fetching board evaluation...");
      const response = await fetch(`${API_URL}/evaluate`);
      if (!response.ok) throw new Error(`Failed to fetch evaluation: ${response.status}`);
      const data = await response.json();
      console.log("Fetched board evaluation:", data.evaluation);
      setBoardEvaluation(data.evaluation); // Update the evaluation score
    } catch (error) {
      console.error("Error fetching board evaluation:", error);
      setBoardEvaluation(null); // Reset evaluation on error
    }
  }, [boardState]);

  const getTurnHistory = (history) => {
    const turns = [];
    for (let i = 0; i < history.length; i += 2) {
      const whiteMove = history[i];
      const blackMove = history[i + 1] || null; // Black move might not exist yet
      turns.push({ white: whiteMove, black: blackMove });
    }
    // console.log("Processed Turn History:", turns);
    return turns;
  };

  const updateBoardStateWithHistory = (newState) => {
    setBoardState((prevState) => {
      if (!newState || !newState.pieces) return prevState;

      // Serialize the board state for comparison (e.g., FEN-like string)
      const serializedState = JSON.stringify(newState.pieces);

      // Update the history map
      setBoardStateHistory((prevHistory) => {
        const newHistory = new Map(prevHistory);
        const count = newHistory.get(serializedState) || 0;
        newHistory.set(serializedState, count + 1);

        // Check for threefold repetition
        if (newHistory.get(serializedState) === 3) {
          console.log("Threefold repetition detected. Declaring a draw.");
          newState.status = GameStatus.DRAW_BY_REPETITION;
          setStatusMessage("Draw by Threefold Repetition");
        }

        return newHistory;
      });

      // Update the move history
      if (newState.last_move && newState.last_move.notation) {
        console.log("Adding move to history:", newState.last_move.notation);
        setMoveHistory((prevHistory) => [...prevHistory, newState.last_move.notation]);
      }

      // Use `turns_since_last_capture_or_pawn` to check for the fifty-move rule
      const turnsSinceLastCaptureOrPawn = newState.turns_since_last_capture_or_pawn || 0;
      if (turnsSinceLastCaptureOrPawn >= 50) {
        console.log("Fifty-move rule triggered. Declaring a draw.");
        newState.status = GameStatus.DRAW;
        setStatusMessage("Draw by Fifty-Move Rule");
      }

      return newState;
    });
  };

  const getCastlingRights = (state) => {
    if (!state || !state.can_castle) return "Unavailable";

    // Assuming `can_castle` is an array of 4 booleans:
    // [White Kingside, White Queenside, Black Kingside, Black Queenside]
    const [whiteKingside, whiteQueenside, blackKingside, blackQueenside] = state.can_castle;

    const whiteRights = [];
    if (whiteKingside) whiteRights.push("O-O");
    if (whiteQueenside) whiteRights.push("O-O-O");

    const blackRights = [];
    if (blackKingside) blackRights.push("O-O");
    if (blackQueenside) blackRights.push("O-O-O");

    return `White: ${whiteRights.length > 0 ? whiteRights.join(", ") : "None"}\nBlack: ${blackRights.length > 0 ? blackRights.join(", ") : "None"}`;
  };

  // Render with <br /> tags
  <p>
    {getCastlingRights(boardState).split("\n").map((line, index) => (
      <React.Fragment key={index}>
        {line}
        <br />
      </React.Fragment>
    ))}
  </p>

  // --- Reset Game ---
  const handleResetGame = useCallback(async () => {
    setIsLoading(true);
    setStatusMessage("Resetting board...");
    try {
      console.log("Sending reset request to backend...");
      const response = await fetch(`${API_URL}/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_mode: gameMode }), // Include gameMode in the request
      });
      if (!response.ok) throw new Error(`Reset request failed: ${response.status}`);
      
      const { initial_state, message } = await response.json();
      if (!initial_state || typeof initial_state !== "object") throw new Error("Received invalid state from server");
      console.log("Received reset state from backend:", initial_state);

      // Update the board state with the reset state
      updateBoardStateWithHistory(initial_state);
      setValidMoves([]); // Clear valid moves temporarily
      setSelectedSquare(null);
      setFiftyMoveCounter(0); // Reset fifty-move counter
      setBoardStateHistory(new Map()); // Clear the board state history
      setMoveHistory([]); // Clear move history
      setStatusMessage(message || "Board reset successfully.");

      // Fetch valid moves for the new state
      await fetchValidMoves(initial_state);
    } catch (error) {
      console.error("Error resetting board:", error);
      setStatusMessage(`Error resetting board: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  }, [fetchValidMoves, gameMode]);

  const returnToModeSelect = useCallback(async () => {
    setIsLoading(true);
    setStatusMessage("Returning to mode selection...");
    try {
      console.log("Resetting board before returning to mode selection...");
      await handleResetGame(); // Trigger the reset method

      console.log("Fetching initial game state for mode selection...");
      const newState = await fetchInitialGameState("returnToModeSelect");
      if (!newState) throw new Error("Failed to fetch initial game state.");

      // Reset the board state while switching to mode selection
      updateBoardStateWithHistory(newState);
      setValidMoves([]);
      setSelectedSquare(null);
      setFiftyMoveCounter(0); // Reset fifty-move counter
      setPlayerColor(null); // Reset player color
      setGameMode(GameMode.SELECT); // Switch to mode selection
      setStatusMessage("Select Game Mode");
    } catch (error) {
      console.error("Error returning to mode selection:", error);
      setStatusMessage(`Error returning to mode selection: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  }, [handleResetGame, fetchInitialGameState]);

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

  const handleResign = useCallback(() => {
    if (isLoading || !boardState || boardState.status !== GameStatus.NORMAL) {
      console.log("Resign action skipped: conditions not met.");
      return;
    }
  
    console.log("Player resigned.");
    setBoardState((prevState) => ({
      ...prevState,
      status: GameStatus.RESIGNED,
    }));
    setStatusMessage("Game Over: Player Resigned");
  }, [isLoading, boardState]);

  // --- Trigger AI Move ---
  const triggerAiMove = useCallback(async () => {
    if (isLoading || !boardState || boardState.status !== GameStatus.NORMAL || gameMode === GameMode.SELECT) {
      console.log("triggerAiMove skipped: conditions not met.");
      return;
    }

  const isAIsTurn =
    (gameMode === GameMode.AI_VS_AI) ||
    (gameMode === GameMode.PLAYER_VS_AI_WHITE && boardState.current_player === Player.BLACK) ||
    (gameMode === GameMode.PLAYER_VS_AI_BLACK && boardState.current_player === Player.WHITE);

  if (!isAIsTurn) {
    console.log("triggerAiMove skipped: not AI's turn.");
    return;
  }

  console.log("Triggering AI move...");
  setIsLoading(true);
  setStatusMessage("AI is thinking...");
  setValidMoves([]);
  setSelectedSquare(null);

  try {
    const response = await fetch(`${API_URL}/ai_move`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `AI move failed: ${response.status}`);
    }

    const newState = await response.json();
    console.log("Received new state directly from POST /api/ai_move:", newState);

    // Check if the game is over
    if (newState.status === GameStatus.CHECKMATE || newState.status === GameStatus.DRAW) {
      console.log("Game over detected.");
      updateBoardStateWithHistory(newState);
      setStatusMessage(getGameStatusMessage(newState));
      return;
    }

    // Update current player and fetch valid moves
    updateBoardStateWithHistory(newState);
    setStatusMessage(getGameStatusMessage(newState));
    if (newState) {
      await fetchValidMoves(newState);
    }
  } catch (error) {
    console.error("Error during AI move:", error);
    setStatusMessage(`Error during AI turn: ${error.message}`);
  } finally {
    setIsLoading(false);
  }
}, [isLoading, boardState, gameMode, fetchValidMoves]);


  // --- Effect to Automatically Trigger AI Move (Keep as is) ---
  useEffect(() => {
    // Guard conditions - should prevent running if loading or game over
    if (!boardState || gameMode === GameMode.SELECT || boardState.status !== GameStatus.NORMAL || isLoading) return;

    // Determine if AI should move based on mode and current player in the state
    const isAIsTurn =
      (gameMode === GameMode.AI_VS_AI) ||
      (gameMode === GameMode.PLAYER_VS_AI_WHITE && boardState.current_player === Player.BLACK) || // AI is Black (1)
      (gameMode === GameMode.PLAYER_VS_AI_BLACK && boardState.current_player === Player.WHITE); // AI is White (0)

    // console.log("DEBUG AI Effect: GameMode = ", gameMode); // Debug log
    // console.log("DEBUG AI Effect: Player Color =", playerColor); // Debug log
    // console.log("DEBUG AI Effect: Current Player =", boardState.current_player); // Debug log
    // console.log("DEBUG AI Effect: isAIsTurn =", isAIsTurn); // Debug log

    // If it's determined to be the AI's turn...
    if (isAIsTurn) {
      // Trigger AI after a short delay
      const timeoutId = setTimeout(triggerAiMove, 500); // 500ms delay
      return () => clearTimeout(timeoutId); // Cleanup timeout on unmount/re-run
    }
  // Dependencies: This effect re-runs if any of these change
  }, [boardState, gameMode, isLoading, triggerAiMove]);

  useEffect(() => {
    if (boardState) {
      const timeoutId = setTimeout(() => {
        console.log("Board state changed, fetching evaluation...");
        fetchBoardEvaluation();
      }, 300); // Delay by 300ms

      return () => clearTimeout(timeoutId); // Cleanup timeout on re-renders
    }
  }, [boardState, fetchBoardEvaluation]);

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
        const movePayload = {
          start: move.start,
          target: move.target,
          promotion: PieceType.NONE,
          current_player: boardState.current_player,
        };

        const response = await fetch(`${API_URL}/apply_move`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(movePayload),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.error || `Move application failed: ${response.status}`);
        }

        const newState = await response.json();
        console.log("Received new state directly from POST /api/apply_move:", newState);
        newState.current_player = newState.current_player === Player.WHITE ? Player.BLACK : Player.WHITE;
        updateBoardStateWithHistory(newState);
        setStatusMessage(getGameStatusMessage(newState));
        await fetchValidMoves(newState);
      } catch (error) {
        console.error("Error applying move:", error);
        setStatusMessage(`Error applying move: ${error.message}`);
        await fetchInitialGameState("handleSquareClick-error");
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
}, [isLoading, boardState, gameMode, playerColor, selectedSquare, validMoves, fetchValidMoves, fetchInitialGameState]);


  // --- Calculate Highlight Squares ---
  const getHighlightSquares = useCallback(() => {
     if (!selectedSquare || !Array.isArray(validMoves) || isLoading) return [];
     return validMoves
       .filter(m => m.start.rank === selectedSquare.rank && m.start.file === selectedSquare.file)
       .map(m => ({ rank: m.target.rank, file: m.target.file }));
  }, [selectedSquare, validMoves, isLoading]);


  // --- Render Logic ---
  const highlightSquares = getHighlightSquares(); // Assumes getHighlightSquares is defined above

  const handleProceed = () => {
    setGameMode(GameMode.SELECT); // Ensure gameMode is reset to SELECT
    setShowCoverPage(false); // Hide cover page and show game mode selector
  };

  const handleReturnToCoverPage = () => {
    setShowCoverPage(true); // Show Cover Page
    setGameMode(null); // Reset game mode
  };

  return (
    <div className="App">
      {showCoverPage ? (
        <CoverPage onProceed={handleProceed} />
      ) : gameMode === GameMode.SELECT ? (
        <GameModeSelector 
          onSelectMode={handleGameModeSelect} 
          onReturnToCoverPage={handleReturnToCoverPage} 
        />
      ) : (
        <div className="game-container">
          <div className="board-container">
            <h1>Chess AI</h1>

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

            {/* Resign Button */}
            <button
              onClick={handleResign}
              disabled={isLoading || !boardState || boardState.status !== GameStatus.NORMAL}
              style={{ marginTop: '15px', marginRight: '10px' }}
            >
              Resign
            </button>

            {/* Reset Board Button */}
            <button
              onClick={handleResetGame}
              disabled={isLoading}
              style={{ marginTop: '15px', marginRight: '10px' }}
            >
              Reset Board
            </button>

            {/* Change Mode Button */}
            <button
              onClick={returnToModeSelect}
              disabled={isLoading}
              style={{ marginTop: '15px' }}
            >
              Change Mode
            </button>
          </div>

          {/* Game Info Sidebar */}
          <div className="sidebar">
            <h2>Game Info</h2>
            <div className="game-info-section">
              <strong>Castling Rights:</strong>
              <p>{getCastlingRights(boardState)}</p>
            </div>
            <div className="game-info-section">
              <strong>Board Evaluation:</strong>
              <p>
                {typeof boardEvaluation === "number"
                  ? `${boardEvaluation.toFixed(2)}`
                  : "Evaluating..."}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;