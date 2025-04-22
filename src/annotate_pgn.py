import chess
import chess.pgn
import chess.engine
import argparse
import os
import sys
from tqdm import tqdm
import multiprocessing


def evaluate_position(board: chess.Board, engine_path: str = "stockfish", depth: int = 4) -> float:
    """
    Evaluate a chess position using Stockfish.
    :param board: A chess.Board object representing the position.
    :param engine_path: Path to the Stockfish executable.
    :param depth: Depth for Stockfish evaluation.
    :return: Evaluation score in centipawns (positive for White, negative for Black).
    """
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        result = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = result["score"].white().score(mate_score=10000)  # Mate scores are capped
        return score if score is not None else 0  # Return 0 if evaluation is unavailable


def evaluate_positions(boards, engine_path, depth):
    """
    Evaluate multiple chess positions using Stockfish.
    :param boards: A list of chess.Board objects representing the positions.
    :param engine_path: Path to the Stockfish executable.
    :param depth: Depth for Stockfish evaluation.
    :return: A list of evaluation scores in centipawns.
    """
    evaluations = []
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        for board in boards:
            try:
                result = engine.analyse(board, chess.engine.Limit(depth=depth))
                score = result["score"].white().score(mate_score=10000)  # Mate scores are capped
                evaluations.append(score if score is not None else 0)
            except Exception as e:
                print(f"[ERROR] Failed to evaluate position: {e}", file=sys.stderr)
                evaluations.append(0)  # Default to 0 if evaluation fails
    return evaluations


def annotate_game(game, engine_path, depth, min_ply, batch_size):
    """
    Annotate a single chess.pgn.Game object with Stockfish evaluations.
    :param game: The chess.pgn.Game object.
    :param engine_path: Path to the Stockfish executable.
    :param depth: Depth for Stockfish evaluation.
    :param min_ply: Minimum ply to evaluate positions.
    :param batch_size: Number of positions to evaluate in a batch.
    :return: Annotated chess.pgn.Game object.
    """
    position_batch = []
    node_batch = []
    annotated_positions = 0

    for node in game.mainline():
        board = node.board()

        # Skip positions before the minimum ply
        if board.ply() >= min_ply:
            position_batch.append(board)
            node_batch.append(node)

            # Evaluate the batch when it reaches the batch size
            if len(position_batch) >= batch_size:
                print(f"[DEBUG] Evaluating batch of {len(position_batch)} positions")
                evaluations = evaluate_positions(position_batch, engine_path, depth)
                for eval_score, target_node in zip(evaluations, node_batch):
                    print(f"[DEBUG] Adding evaluation {eval_score / 100:.2f} to move {target_node.move}", file=sys.stderr)
                    target_node.comment = f"Stockfish eval: {eval_score / 100:.2f} (centipawns)"
                    annotated_positions += 1
                position_batch.clear()
                node_batch.clear()

    # Evaluate any remaining positions in the batch
    if position_batch:
        print(f"[DEBUG] Evaluating final batch of {len(position_batch)} positions")
        evaluations = evaluate_positions(position_batch, engine_path, depth)
        for eval_score, target_node in zip(evaluations, node_batch):
            print(f"[DEBUG] Adding evaluation {eval_score / 100:.2f} to move {target_node.move}", file=sys.stderr)
            target_node.comment = f"Stockfish eval: {eval_score / 100:.2f} (centipawns)"
            annotated_positions += 1

    print(f"[DEBUG] Annotated {annotated_positions} positions in the game.")
    return game


def write_annotated_game(game, output_pgn):
    """
    Write an annotated game to the output PGN file.
    :param game: The chess.pgn.Game object.
    :param output_pgn: The output file handle.
    """
    exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
    pgn_string = game.accept(exporter)
    output_pgn.write(pgn_string + "\n\n")


def annotate_pgn_file(input_pgn_path, output_pgn_path, engine_path, depth, min_ply, batch_size):
    """
    Annotate a PGN file with Stockfish evaluations.
    :param input_pgn_path: Path to the input PGN file.
    :param output_pgn_path: Path to the output PGN file.
    :param engine_path: Path to the Stockfish executable.
    :param depth: Depth for Stockfish evaluation.
    :param min_ply: Minimum ply to evaluate positions.
    :param batch_size: Number of positions to evaluate in a batch.
    """
    print(f"[DEBUG] Starting annotation for {input_pgn_path}")
    with open(input_pgn_path, "r", encoding="utf-8") as input_pgn, open(output_pgn_path, "w", encoding="utf-8") as output_pgn:
        game_count = 0
        pbar = tqdm(desc=f"Annotating {os.path.basename(input_pgn_path)}", unit="game", leave=False)

        while True:
            try:
                game = chess.pgn.read_game(input_pgn)
                if game is None:
                    break  # End of PGN file
                game_count += 1
                pbar.update(1)

                print(f"[DEBUG] Processing game {game_count}")
                annotated_game = annotate_game(game, engine_path, depth, min_ply, batch_size)
                write_annotated_game(annotated_game, output_pgn)

            except Exception as e:
                print(f"[ERROR] Failed to process game {game_count}: {e}", file=sys.stderr)

        pbar.close()
        print(f"[DEBUG] Annotated {game_count} games.")
        print(f"[DEBUG] Saved annotated PGN to {output_pgn_path}.")


def annotate_pgn_directory(input_dir, output_dir, engine_path, depth, min_ply, batch_size):
    """
    Annotate all PGN files in a directory with Stockfish evaluations.
    :param input_dir: Path to the directory containing input PGN files.
    :param output_dir: Path to the directory to save annotated PGN files.
    :param engine_path: Path to the Stockfish executable.
    :param depth: Depth for Stockfish evaluation.
    :param min_ply: Minimum ply to evaluate positions.
    :param batch_size: Number of positions to evaluate in a batch.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pgn_files = [f for f in os.listdir(input_dir) if f.endswith(".pgn")]
    print(f"[DEBUG] Found {len(pgn_files)} PGN files in {input_dir}")

    for pgn_file in tqdm(pgn_files, desc="Annotating PGN files", unit="file"):
        input_pgn_path = os.path.join(input_dir, pgn_file)
        output_pgn_path = os.path.join(output_dir, pgn_file)

        print(f"[DEBUG] Annotating {pgn_file}")
        annotate_pgn_file(input_pgn_path, output_pgn_path, engine_path, depth, min_ply, batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate PGN files with Stockfish evaluations.")
    parser.add_argument("input", help="Path to the input PGN file or directory.")
    parser.add_argument("output", help="Path to the output annotated PGN file or directory.")
    parser.add_argument("--engine_path", default="stockfish", help="Path to the Stockfish executable.")
    parser.add_argument("--depth", type=int, default=4, help="Depth for Stockfish evaluation (default: 4).")
    parser.add_argument("--min_ply", type=int, default=8, help="Minimum ply to evaluate positions (default: 8).")
    parser.add_argument("--batch_size", type=int, default=50000, help="Number of positions to evaluate in a batch (default: 100).")

    args = parser.parse_args()

    if os.path.isdir(args.input):
        annotate_pgn_directory(
            input_dir=args.input,
            output_dir=args.output,
            engine_path=args.engine_path,
            depth=args.depth,
            min_ply=args.min_ply,
            batch_size=args.batch_size
        )
    else:
        annotate_pgn_file(
            input_pgn_path=args.input,
            output_pgn_path=args.output,
            engine_path=args.engine_path,
            depth=args.depth,
            min_ply=args.min_ply,
            batch_size=args.batch_size
        )