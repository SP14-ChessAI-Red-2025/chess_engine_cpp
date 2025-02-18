import sys

from python.chess import *


def main():
    if len(sys.argv) < 2:
        print("Must specify library path on the command line")

    library_path = sys.argv[1]

    with ChessEngine(library_path) as chess_engine:
        valid_moves = chess_engine.get_valid_moves(chess_engine.board_state)

        print(f"{len(valid_moves)} moves found")

        for move in valid_moves:
            print(chess_engine.move_to_str(move))


if __name__ == "__main__":
    main()
